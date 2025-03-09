import tensorflow as tf
import numpy as np
import json
from typing import Any, Dict, Tuple, List, Optional

def parse_example(
    example_proto: tf.Tensor, 
    input_stats: Dict[str, Tuple[float, float]], 
    output_stats: Dict[str, Any]
) -> Tuple[Dict[str, tf.Tensor], Dict[str, List[tf.Tensor]]]:
    """
    Parse a single tf.train.Example proto using provided normalization statistics.

    Parameters
    ----------
    example_proto : tf.Tensor
        A serialized Example proto containing raw features.
    input_stats : dict
        Normalization statistics for input features. Keys are regions ('UV', 'Vis', 'NIR') 
        with corresponding tuples (mean, std).
    output_stats : dict
        Normalization and transformation statistics for output features.

    Returns
    -------
    tuple
        A tuple (normalized_inputs, grouped_outputs) where:
          - normalized_inputs is a dict mapping feature names to normalized tf.Tensor inputs.
          - grouped_outputs is a dict with keys 'physical_output', 'main_chemical_output', 
            and 'other_chemical_output', each containing a list of processed output tensors.
    """
    # 1) Define raw feature schemas.
    raw_input_features = {
        'NOISY_ALBEDO_B-NIR': tf.io.VarLenFeature(tf.float32),
        'NOISY_ALBEDO_B-UV':  tf.io.VarLenFeature(tf.float32),
        'NOISY_ALBEDO_B-Vis': tf.io.VarLenFeature(tf.float32),
    }

    raw_output_features = {
        # Planetary parameters.
        "OBJECT-RADIUS-REL-EARTH": tf.io.FixedLenFeature([], tf.float32),
        "OBJECT-GRAVITY":           tf.io.FixedLenFeature([], tf.float32),
        "ATMOSPHERE-TEMPERATURE":   tf.io.FixedLenFeature([], tf.float32),
        "ATMOSPHERE-PRESSURE":      tf.io.FixedLenFeature([], tf.float32),

        # Chemical abundances.
        'CH4':  tf.io.FixedLenFeature([], tf.float32),
        'CO':   tf.io.FixedLenFeature([], tf.float32),
        'CO2':  tf.io.FixedLenFeature([], tf.float32),
        'H2O':  tf.io.FixedLenFeature([], tf.float32),
        'N2':   tf.io.FixedLenFeature([], tf.float32),
        'N2O':  tf.io.FixedLenFeature([], tf.float32),
        'O2':   tf.io.FixedLenFeature([], tf.float32),
        'O3':   tf.io.FixedLenFeature([], tf.float32),

        # 'Earth_type': tf.io.FixedLenFeature([], tf.string)
    }

    # 2) Parse the example.
    all_features = {**raw_input_features, **raw_output_features}
    parsed = tf.io.parse_single_example(example_proto, all_features)

    # 3) Normalize input spectra.
    normalized_inputs = {}
    for region in ['UV', 'Vis', 'NIR']:
        spectrum = parsed[f'NOISY_ALBEDO_B-{region}']
        if isinstance(spectrum, tf.SparseTensor):
            spectrum = tf.sparse.to_dense(spectrum, default_value=0.0)

        mean = input_stats[region][0]
        std  = input_stats[region][1]
        normalized = (spectrum - mean) / std

        # Reshape for model (example shape)
        if region == 'UV':
            normalized = tf.reshape(normalized, [8, 1])
        elif region == 'Vis':
            normalized = tf.reshape(normalized, [94, 1])
        elif region == 'NIR':
            normalized = tf.reshape(normalized, [49, 1])

        normalized_inputs[f'NOISY_ALBEDO_B-{region}'] = normalized

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
    for chem in ['CH4', 'CO', 'CO2', 'H2O', 'N2', 'N2O']:
        val = parsed[chem]
        best_n = output_stats['chemical_output_stats'][chem]["best_n"]
        processed = tf.math.pow(float(val), float(1 / best_n))
        other_chemical_outputs.append(processed)

    grouped_outputs = {
        'physical_output': physical_outputs,              # shape (4,)
        'main_chemical_output': main_chemical_outputs,    # shape (2,)
        'other_chemical_output': other_chemical_outputs,  # shape (7,)
        #'earth_type': parsed['Earth_type']               # shape (1,)
    }

    return normalized_inputs, grouped_outputs


def read_tfrecord(
    file_path: str, 
    batch_size: int = 256, 
    shuffle_buffer: Optional[int] = None, 
    repeat: bool = False
) -> tf.data.Dataset:
    """
    Read a TFRecord file, parse its examples, and return a batched tf.data.Dataset.

    Parameters
    ----------
    file_path : str
        Path to the TFRecord file.
    batch_size : int, optional
        Number of samples per batch, by default 256.
    shuffle_buffer : int, optional
        Buffer size for shuffling. If None or <= 0, shuffling is not applied.
    repeat : bool, optional
        Whether to repeat the dataset indefinitely, by default False.

    Returns
    -------
    tf.data.Dataset
        A dataset of parsed examples, batched and prefetched.
    """
    with open('../data/normalization_stats.json') as f:
        stats = json.load(f)

    # Create lookup dictionaries for input normalization statistics.
    input_stats = {
        'UV': (stats['inputs']['B-UV']['mean'], stats['inputs']['B-UV']['std']),
        'Vis': (stats['inputs']['B-Vis']['mean'], stats['inputs']['B-Vis']['std']),
        'NIR': (stats['inputs']['B-NIR']['mean'], stats['inputs']['B-NIR']['std'])
    }

    physical_keys = ["OBJECT-RADIUS-REL-EARTH", "OBJECT-GRAVITY",
                    "ATMOSPHERE-TEMPERATURE", "ATMOSPHERE-PRESSURE"]
    physical_output_stats = {
        key: (stats['outputs'][key]['min'], stats['outputs'][key]['max'], stats['outputs'][key]['best_n'])
        for key in physical_keys
    }

    chemical_keys = ['CH4', 'CO', 'CO2', 'H2O', 'N2', 'N2O', 'O2', 'O3']
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
        lambda x: parse_example(x, input_stats, output_stats), 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if repeat:
        parsed_dataset = parsed_dataset.repeat()

    parsed_dataset = parsed_dataset.batch(batch_size, drop_remainder=False)
    parsed_dataset = parsed_dataset.prefetch(tf.data.AUTOTUNE)
    
    return parsed_dataset
