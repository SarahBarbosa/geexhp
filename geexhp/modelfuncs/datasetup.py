import os
import glob
from itertools import chain
from typing import Dict, List

import pandas as pd
import tensorflow as tf

def combine_parquet(folder: str, keyword: str, output_file: bool = False) -> pd.DataFrame:
    """
    Combine multiple Parquet files from a specified folder into a single DataFrame.

    This function searches for Parquet files in the given `folder` whose filenames
    contain the specified `keyword`. It then reads and concatenates all matching 
    files into a single DataFrame, and adds an "Earth_type" column based on certain 
    keywords in each filename (e.g., "modern", "proterozoic", "archean"). If 
    `output_file` is set to True, the combined data is optionally saved as a new 
    Parquet file.

    `WARNING`: This function loads all matching Parquet files into memory at once. If the 
    files are very large or if there are many files, this process can be 
    resource-intensive and may cause memory or performance issues.

    Parameters
    ----------
    folder : str
        The directory containing the Parquet files.
    keyword : str
        The keyword to filter Parquet filenames.
    output_file : bool, optional
        Whether to save the combined DataFrame as a new Parquet file (default is False).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with all matching Parquet files concatenated and an 
        additional "Earth_type" column indicating the file source.
    """
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

    if output_file:
        output_dir = "../data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{keyword}_data.parquet')
        combineddf.to_parquet(output_path)
        print(f"Combined DataFrame saved to {output_path}")

    return combineddf


def _extract(row: pd.Series, sorted_molecules: list[str]) -> dict[str, float]:
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
    Expand molecule abundances in a DataFrame into individual columns.

    This function identifies all unique molecules specified in 
    'ATMOSPHERE-LAYERS-MOLECULES' for each row of the DataFrame, and extracts 
    their corresponding abundance values (from 'ATMOSPHERE-LAYER-1'). It 
    then adds one column per molecule to the original DataFrame, where each 
    column contains the abundance for that molecule.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the atmospheric layers columns. It must have
        the following columns:
        - 'ATMOSPHERE-LAYERS-MOLECULES': a comma-separated list of molecule names
        - 'ATMOSPHERE-LAYER-1': a comma-separated list of values, where the 
          first two are T and P, and subsequent values correspond to the 
          abundances of the molecules.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional columns (one per unique molecule)
        containing the abundance values for each molecule.
    """
    molecules = set()
    df["ATMOSPHERE-LAYERS-MOLECULES"].apply(lambda x: molecules.update(x.split(',')))
    sorted_molecules = sorted(molecules)

    # Row-wise expand
    abun_df = df.apply(lambda row: _extract(row, sorted_molecules), axis=1, result_type="expand")
    abun_df.columns = [f"{molecule}" for molecule in sorted_molecules]
    
    return pd.concat([df, abun_df], axis=1)

def concatenate_all_tfrecords(root_folder: str,output_tfrecord_file: str) -> None:
    """
    Concatenate all TFRecord files found under a given directory into a single TFRecord file.

    Parameters
    ----------
    root_folder : str
        The path to the directory containing TFRecord files.
    output_tfrecord_file : str
        The output file path (e.g. "../data/geexhp_samples.tfrecord") where all
        concatenated records will be stored.

    Returns
    -------
    None
        Writes a single TFRecord file at the specified output location.
    """
    # Recursively gather all TFRecord files under root_folder
    files_array = [
        [file for file in files if file.endswith('.tfrecord')]
        for _, _, files in os.walk(root_folder)
    ]
    all_files = list(chain.from_iterable(files_array))

    # Write concatenated records to a new TFRecord file
    with tf.io.TFRecordWriter(output_tfrecord_file) as writer:
        for tfrecord_file in all_files:
            tfrecord_file_path = os.path.join(root_folder, tfrecord_file)
            # Read each TFRecord file and write its records to the output
            for record in tf.data.TFRecordDataset(tfrecord_file_path):
                writer.write(record.numpy())

    print(f"Concatenated TFRecord file saved to '{output_tfrecord_file}'")

def _load_dataset_paths(root_folder: str) -> Dict[str, Dict[str, List]]:
    """
    Scan a folder for TFRecord files named in the format "<era>_..._<number_of_samples>.tfrecord",
    and organize them into a dictionary for each era.
    """
    files = os.listdir(root_folder)

    data = {
        'modern': {
            'file_paths': [],
            'file_numbers_of_samples': []
        },
        'proterozoic': {
            'file_paths': [],
            'file_numbers_of_samples': []
        },
        'archean': {
            'file_paths': [],
            'file_numbers_of_samples': []
        },
    }

    for file in files:
        # Example filename format: "modern_..._100.tfrecord"
        #  era = "modern"
        #  number_of_samples = 100
        era = file.split('/')[-1].split('_')[0]  # get 'modern', 'proterozoic', or 'archean'
        number_of_samples_str = file.split('_')[-1].split('.')[0]  # e.g. "100"

        if era in data:
            data[era]['file_paths'].append(os.path.join(root_folder, file))
            data[era]['file_numbers_of_samples'].append(int(number_of_samples_str))

    return data

def _split_era(root_folder: str, train_split: float, val_split: float) -> Dict[str, Dict[str, List]]:
    """
    Determine the index boundaries for train, validation, and test splits
    within each era, based on the total sample counts from each era's TFRecord files.
    """
    tf_data = _load_dataset_paths(root_folder)

    # Process each era separately.
    for era in list(tf_data.keys()):
        total_number_of_samples = sum(tf_data[era]['file_numbers_of_samples'])
        count = 0
        
        # Initialize a new split_indexes dict for this era.
        split_indexes = {'train': None, 'val': None}

        for index, num_samples in enumerate(tf_data[era]['file_numbers_of_samples']):
            if split_indexes['train'] is None and count >= total_number_of_samples * train_split:
                split_indexes['train'] = index

            if count >= total_number_of_samples * (train_split + val_split):
                split_indexes['val'] = index
                break

            count += num_samples

        tf_data[era]['split_indexes'] = split_indexes

    return tf_data


def train_val_test_split(root_folder: str, train_split: float = 0.8, val_split: float = 0.1) -> None:
    """
    Split the TFRecord dataset into train/val/test sets by era, then concatenate
    the files for each split across all eras into single TFRecord files.

    Parameters
    ----------
    root_folder : str
        The path to the directory containing TFRecord files.

    Returns
    -------
    None
        Creates three TFRecord files (train, val, test) at the same directory level.
    """
    tf_data = _split_era(root_folder, train_split, val_split)

    # All splits must be the same for all eras.
    train_split = tf_data['modern']['split_indexes']['train']
    val_split = tf_data['modern']['split_indexes']['val']

    data_types_paths = {
        'train': (
            tf_data['modern']['file_paths'][:train_split]
            + tf_data['proterozoic']['file_paths'][:train_split]
            + tf_data['archean']['file_paths'][:train_split]
        ),
        'val': (
            tf_data['modern']['file_paths'][train_split:val_split]
            + tf_data['proterozoic']['file_paths'][train_split:val_split]
            + tf_data['archean']['file_paths'][train_split:val_split]
        ),
        'test': (
            tf_data['modern']['file_paths'][val_split:]
            + tf_data['proterozoic']['file_paths'][val_split:]
            + tf_data['archean']['file_paths'][val_split:]
        ),
    }

    for data_type, file_paths_list in data_types_paths.items():
        # Output TFRecord file
        output_tfrecord_file = f"../data/{data_type}.tfrecord"

        # Write concatenated records to a new TFRecord file
        with tf.io.TFRecordWriter(output_tfrecord_file) as writer:
            for tfrecord_file in file_paths_list:
                for record in tf.data.TFRecordDataset(tfrecord_file):
                    writer.write(record.numpy())

        print(f"Concatenated TFRecord file saved to '{output_tfrecord_file}'")