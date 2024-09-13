import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

def combine_parquet(folder: str, keyword: str, output_file: bool = False) -> pd.DataFrame:
    file_pattern = os.path.join(folder, f'*{keyword}*.parquet')
    files = glob.glob(file_pattern)

    if not files:
        print(f"No files found with keyword '{keyword}' in the folder '{folder}'")
        return pd.DataFrame()

    # Extract Earth type from filenames and combine data
    dataframes = []
    for file in files:
        if "modern" in file:
            earth_type = "modern"
        elif "proterozoic" in file:
            earth_type = "proterozoic"
        elif "archean" in file:
            earth_type = "archean"
        else:
            earth_type = "random"

        df = pd.read_parquet(file)
        df["Earth_type"] = earth_type  # Add Earth type label
        dataframes.append(df)

    combineddf = pd.concat(dataframes, ignore_index=True)

    # Optionally, save the combined DataFrame as a new .parquet file
    if output_file:
        output_dir = "../data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{keyword}_data.parquet')
        combineddf.to_parquet(output_path)
        print(f"Combined DataFrame saved to {output_path}")

    return combineddf

def _extract(row, sorted_molecules):
    """
    Function to map molecules to abundances for each row.
    """
    molecule_list = row["ATMOSPHERE-LAYERS-MOLECULES"].split(',')

    # ALL ATMOSPHERE LAYER ARE EQUAL!!
    abundance_list = row["ATMOSPHERE-LAYER-1"].split(',')[2:]  # Skip the TP
    molecule_abundance = dict(zip(molecule_list, abundance_list))
    return {molecule: float(molecule_abundance.get(molecule, 0)) for molecule in sorted_molecules}

def extract_abundances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts abundances from `'ATMOSPHERE-LAYER'` and maps them to corresponding molecules in
    `'ATMOSPHERE-LAYERS-MOLECULES'`, adding each molecule as a new column in the DataFrame.
    """
    molecules = set()
    df["ATMOSPHERE-LAYERS-MOLECULES"].apply(lambda x: molecules.update(x.split(',')))
    sorted_molecules = sorted(molecules)
    abun_df = df.apply(lambda row: _extract(row, sorted_molecules), axis=1, result_type="expand")
    abun_df.columns = [f"{molecule}" for molecule in sorted_molecules]
    return pd.concat([df, abun_df], axis=1)

def normalize_data(X_train, X_test, y_train_abundance, y_test_abundance,
                   y_train_planetary, y_test_planetary,
                   x_scaler=None, y_scalers_abundance=None, y_scalers_planetary=None):
    """
    Normalizes input data (X_train and X_test) and handles separate outputs for 
    abundances and planetary parameters.
    """
    # Input Scaler
    if x_scaler is None:
        x_scaler = StandardScaler()
    
    # Flatten X for scaling
    num_train_samples, num_features = X_train.shape
    num_test_samples = X_test.shape[0]
    X_train_flat = X_train.reshape(num_train_samples, -1)
    X_test_flat = X_test.reshape(num_test_samples, -1)
    
    # Fit and transform X
    X_train_scaled = x_scaler.fit_transform(X_train_flat)
    X_test_scaled = x_scaler.transform(X_test_flat)
    
    # Reshape X back to original shape for Conv1D layers
    X_train_scaled = X_train_scaled.reshape(num_train_samples, num_features, 1)
    X_test_scaled = X_test_scaled.reshape(num_test_samples, num_features, 1)
    
    # Output Scalers for Abundances
    num_abundance_features = y_train_abundance.shape[1]
    if y_scalers_abundance is None:
        y_scalers_abundance = []
        y_train_abundance_scaled = np.zeros_like(y_train_abundance)
        y_test_abundance_scaled = np.zeros_like(y_test_abundance)
        for i in range(num_abundance_features):
            scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_abundance_scaled[:, i] = scaler.fit_transform(y_train_abundance[:, i].reshape(-1, 1)).flatten()
            y_test_abundance_scaled[:, i] = scaler.transform(y_test_abundance[:, i].reshape(-1, 1)).flatten()
            y_scalers_abundance.append(scaler)
    else:
        y_train_abundance_scaled = np.zeros_like(y_train_abundance)
        y_test_abundance_scaled = np.zeros_like(y_test_abundance)
        for i in range(num_abundance_features):
            scaler = y_scalers_abundance[i]
            y_train_abundance_scaled[:, i] = scaler.transform(y_train_abundance[:, i].reshape(-1, 1)).flatten()
            y_test_abundance_scaled[:, i] = scaler.transform(y_test_abundance[:, i].reshape(-1, 1)).flatten()
    
    # Output Scalers for Planetary Parameters
    num_planetary_features = y_train_planetary.shape[1]
    if y_scalers_planetary is None:
        y_scalers_planetary = []
        y_train_planetary_scaled = np.zeros_like(y_train_planetary)
        y_test_planetary_scaled = np.zeros_like(y_test_planetary)
        for i in range(num_planetary_features):
            scaler = StandardScaler()
            y_train_planetary_scaled[:, i] = scaler.fit_transform(y_train_planetary[:, i].reshape(-1, 1)).flatten()
            y_test_planetary_scaled[:, i] = scaler.transform(y_test_planetary[:, i].reshape(-1, 1)).flatten()
            y_scalers_planetary.append(scaler)
    else:
        y_train_planetary_scaled = np.zeros_like(y_train_planetary)
        y_test_planetary_scaled = np.zeros_like(y_test_planetary)
        for i in range(num_planetary_features):
            scaler = y_scalers_planetary[i]
            y_train_planetary_scaled[:, i] = scaler.transform(y_train_planetary[:, i].reshape(-1, 1)).flatten()
            y_test_planetary_scaled[:, i] = scaler.transform(y_test_planetary[:, i].reshape(-1, 1)).flatten()
    
    # Print shapes for debugging
    print(f"Train input shape: {X_train_scaled.shape}")
    print(f"Test input shape: {X_test_scaled.shape}")
    print(f"Train abundance labels shape: {y_train_abundance_scaled.shape}")
    print(f"Test abundance labels shape: {y_test_abundance_scaled.shape}")
    print(f"Train planetary labels shape: {y_train_planetary_scaled.shape}")
    print(f"Test planetary labels shape: {y_test_planetary_scaled.shape}")
    
    return (X_train_scaled, X_test_scaled,
            y_train_abundance_scaled, y_test_abundance_scaled,
            y_train_planetary_scaled, y_test_planetary_scaled,
            x_scaler, y_scalers_abundance, y_scalers_planetary)

def _float_feature(value):
    """Returns a float_list from a float / double or iterable of floats."""
    if isinstance(value, (list, np.ndarray)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(input_albedo, abundance_labels, planetary_labels):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'input_albedo': _float_feature(input_albedo.flatten()),
        'abundance_labels': _float_feature(abundance_labels),
        'planetary_labels': _float_feature(planetary_labels),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecords(filename, X_data, y_abundance_data, y_planetary_data):
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(X_data)):
            example = serialize_example(X_data[i], y_abundance_data[i], y_planetary_data[i])
            writer.write(example)
    print(f"Data successfully written to {filename}")

