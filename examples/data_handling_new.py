from typing import Any, Dict, Tuple, Optional
import tensorflow as tf
import json

TELESCOPE = {
    "LUVOIR": dict(prefix="B",
                   region_bins={"UV": 8,  "Vis": 94,  "NIR": 49}),
    "HABEX" : dict(prefix="SS",
                   region_bins={"UV": 7,  "Vis":109,  "NIR": 25}),
}

def parse_example(example_proto: tf.Tensor,
                  input_stats: Dict[str, Tuple[float, float]],
                  output_stats: Dict[str, Any],
                  telescope: str = "LUVOIR"):

    meta   = TELESCOPE[telescope]
    pfx    = meta["prefix"]
    bins   = meta["region_bins"]

    # 1) Define raw feature schemas.
    raw_input_features = {f'NOISY_ALBEDO_{pfx}-{r}': tf.io.VarLenFeature(tf.float32)
              for r in ['NIR','UV','Vis']}

    raw_output_features = {
        # Planetary parameters.
        "OBJECT-RADIUS-REL-EARTH": tf.io.FixedLenFeature([], tf.float32),
        "OBJECT-GRAVITY":           tf.io.FixedLenFeature([], tf.float32),
        "ATMOSPHERE-TEMPERATURE":   tf.io.FixedLenFeature([], tf.float32),
        "ATMOSPHERE-PRESSURE":      tf.io.FixedLenFeature([], tf.float32),

        # Chemical abundances.
        'CH4':  tf.io.FixedLenFeature([], tf.float32),
        # 'CO':   tf.io.FixedLenFeature([], tf.float32),
        'CO2':  tf.io.FixedLenFeature([], tf.float32),
        'H2O':  tf.io.FixedLenFeature([], tf.float32),
        'N2':   tf.io.FixedLenFeature([], tf.float32),
        # 'N2O':  tf.io.FixedLenFeature([], tf.float32),
        'O2':   tf.io.FixedLenFeature([], tf.float32),
        'O3':   tf.io.FixedLenFeature([], tf.float32),

        'Earth_type': tf.io.FixedLenFeature([], tf.string)
    }

    clean_albedo_features = {f'ALBEDO_{pfx}-{r}': tf.io.VarLenFeature(tf.float32)
                         for r in ['NIR', 'UV', 'Vis']}

    # 2) Parse the example.
    all_features = {**raw_input_features, **clean_albedo_features, **raw_output_features}
    parsed = tf.io.parse_single_example(example_proto, all_features)

    # 3) Normalize input spectra.
    normalized_inputs = {}
    for region in ['UV', 'Vis', 'NIR']:
        spectrum = parsed[f'NOISY_ALBEDO_{pfx}-{region}']
        if isinstance(spectrum, tf.SparseTensor):
            spectrum = tf.sparse.to_dense(spectrum, default_value=0.0)

        mean = input_stats[region][0]
        std  = input_stats[region][1]
        normalized = (spectrum - mean) / std

        normalized   = tf.reshape(normalized, [bins[region], 1])
        normalized_inputs[f'NOISY_ALBEDO_{pfx}-{region}'] = normalized

    # Just to visualize in the same units
    clean_inputs = {}
    for region in ['UV', 'Vis', 'NIR']:
        spectrum_clean = parsed[f'ALBEDO_{pfx}-{region}']
        if isinstance(spectrum_clean, tf.SparseTensor):
            spectrum_clean = tf.sparse.to_dense(spectrum_clean, default_value=0.0)

        # mean = input_stats[region][0]
        # std  = input_stats[region][1]
        # normalized = (spectrum_clean - mean) / std

        spectrum_clean   = tf.reshape(spectrum_clean, [bins[region], 1])
        clean_inputs[f'ALBEDO_{pfx}-{region}'] = spectrum_clean

    # 4) Process outputs.
    physical_outputs = []
    main_chemical_outputs = []
    other_chemical_outputs = []

    # 4a) Process planetary parameters (physical outputs).
    for param in ["OBJECT-RADIUS-REL-EARTH", "OBJECT-GRAVITY",
                    "ATMOSPHERE-TEMPERATURE", "ATMOSPHERE-PRESSURE"]:
        val = parsed[param]
        # To avoid a very ponctual error we found
        # the floating point precision error generates a single NaN value here :(
        if val == 273.15:
            val += 0.00002
        min_  = output_stats['physical_output_stats'][param][0]
        max_  = output_stats['physical_output_stats'][param][1]
        best_n  = output_stats['physical_output_stats'][param][2]
        processed = (val - min_) / (max_ - min_)
        processed = tf.math.pow(float(processed), float(1/best_n))
        physical_outputs.append(processed)

    # 4b) Process main chemical abundances.
    for chem in ['O2', 'O3']:
        val = parsed[chem]
        best_n = output_stats['chemical_output_stats'][chem]["best_n"]
        processed = tf.math.pow(float(val), float(1/best_n))
        main_chemical_outputs.append(processed)

    # 4c) Process other chemical abundances.
    for chem in ['CH4', 'CO2', 'H2O', 'N2']:
        val = parsed[chem]
        best_n = output_stats['chemical_output_stats'][chem]["best_n"]
        processed = tf.math.pow(float(val), float(1 / best_n))
        other_chemical_outputs.append(processed)

    grouped_outputs = {
        'physical_output': physical_outputs,              # shape (4,)
        'main_chemical_output': main_chemical_outputs,    # shape (2,)
        'other_chemical_output': other_chemical_outputs,  # shape (7,)
        'earth_type': parsed['Earth_type']
    }

    return normalized_inputs, grouped_outputs, clean_inputs


def read_tfrecord(file_path: str,
                  telescope: str = "LUVOIR",
                  batch_size: int = 1024,
                  shuffle_buffer: Optional[int] = None,
                  repeat: bool = False):

    meta  = TELESCOPE[telescope]
    pfx   = meta["prefix"]

    with open('/content/drive/MyDrive/CNN-MODELS-JUN-VERSION/DATA/normalization_stats.json') as f:
        stats = json.load(f)

    # Create lookup dictionaries for input normalization statistics.
    input_stats = {r: (stats['inputs'][f'{pfx}-{r}']['mean'],
                       stats['inputs'][f'{pfx}-{r}']['std'])
                   for r in ['UV','Vis','NIR']}

    physical_keys = ["OBJECT-RADIUS-REL-EARTH", "OBJECT-GRAVITY",
                    "ATMOSPHERE-TEMPERATURE", "ATMOSPHERE-PRESSURE"]
    physical_output_stats = {
        key: (stats['outputs'][key]['min'], stats['outputs'][key]['max'], stats['outputs'][key]['best_n'])
        for key in physical_keys
    }

    chemical_keys = ['CH4', 'CO2', 'H2O', 'N2', 'O2', 'O3']
    chemical_output_stats = {
        key: stats['outputs'][key] for key in chemical_keys
    }

    output_stats: Dict[str, Any] = {
        'physical_output_stats': physical_output_stats,
        'chemical_output_stats': chemical_output_stats
    }

    dataset = tf.data.TFRecordDataset(file_path, num_parallel_reads=tf.data.AUTOTUNE)

    if shuffle_buffer is not None and shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Parse the dataset using the parse_example function.
    parsed_dataset = dataset.map(
        lambda x: parse_example(x, input_stats, output_stats, telescope=telescope),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if repeat:
        parsed_dataset = parsed_dataset.repeat()

    parsed_dataset = parsed_dataset.batch(batch_size, drop_remainder=False)
    parsed_dataset = parsed_dataset.prefetch(tf.data.AUTOTUNE)

    return parsed_dataset


def build_era_dataset(telescope_ds, era_label=b"modern", batch=1):
    base = telescope_ds.unbatch()
    ds = base.filter(lambda x, y: tf.equal(y['earth_type'], era_label))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def load_stats(file):
    with open('DATA/normalization_stats.json') as f:
        stats = json.load(f)

    return stats