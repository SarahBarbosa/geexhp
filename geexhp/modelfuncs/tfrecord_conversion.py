import os
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from astropy.constants import R_earth
from geexhp.modelfuncs import datasetup as dset

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

COLUMNS_OF_INTEREST = [

    "ALBEDO_B-NIR",
    "ALBEDO_B-UV",
    "ALBEDO_B-Vis",

    "ALBEDO_SS-NIR",
    "ALBEDO_SS-UV",
    "ALBEDO_SS-Vis",

    "NOISY_ALBEDO_B-NIR",
    "NOISY_ALBEDO_B-UV",
    "NOISY_ALBEDO_B-Vis",

    "NOISY_ALBEDO_SS-NIR",
    "NOISY_ALBEDO_SS-UV",
    "NOISY_ALBEDO_SS-Vis",

    "NOISE_B-NIR",
    "NOISE_B-UV",
    "NOISE_B-Vis",

    "NOISE_SS-NIR",
    "NOISE_SS-UV",
    "NOISE_SS-Vis",

    "OBJECT-DIAMETER",
    "OBJECT-GRAVITY",
    "ATMOSPHERE-TEMPERATURE",
    "ATMOSPHERE-PRESSURE",
    "Earth_type",
]

MOLECULES = [
    "C2H6",
    "CH4",
    "CO",
    "CO2",
    "H2O",
    "N2",
    "N2O",
    "O2",
    "O3",
]

SPECTRA = [
    "NOISY_ALBEDO_B-NIR",
    "NOISY_ALBEDO_B-UV",
    "NOISY_ALBEDO_B-Vis",

    "NOISY_ALBEDO_SS-NIR",
    "NOISY_ALBEDO_SS-UV",
    "NOISY_ALBEDO_SS-Vis",
]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _float_feature(value):
    """Returns a float_list from a float / list of floats."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_feature_list(value):
    """Returns a float_list from a float / list of floats."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _serialize_sample(row: Dict[str, Union[str, float, List[float]]]) -> bytes:
    """
    Serialize a single sample (row) into a tf.train.Example.
    """
    feature = {
        # SPECTRA
        "ALBEDO_B-NIR": _float_feature_list(row["ALBEDO_B-NIR"]),
        "ALBEDO_B-UV": _float_feature_list(row["ALBEDO_B-UV"]),
        "ALBEDO_B-Vis": _float_feature_list(row["ALBEDO_B-Vis"]),

        "ALBEDO_SS-NIR": _float_feature_list(row["ALBEDO_SS-NIR"]),
        "ALBEDO_SS-UV": _float_feature_list(row["ALBEDO_SS-UV"]),
        "ALBEDO_SS-Vis": _float_feature_list(row["ALBEDO_SS-Vis"]),

        "NOISY_ALBEDO_B-NIR": _float_feature_list(row["NOISY_ALBEDO_B-NIR"]),
        "NOISY_ALBEDO_B-UV": _float_feature_list(row["NOISY_ALBEDO_B-UV"]),
        "NOISY_ALBEDO_B-Vis": _float_feature_list(row["NOISY_ALBEDO_B-Vis"]),

        "NOISY_ALBEDO_SS-NIR": _float_feature_list(row["NOISY_ALBEDO_SS-NIR"]),
        "NOISY_ALBEDO_SS-UV": _float_feature_list(row["NOISY_ALBEDO_SS-UV"]),
        "NOISY_ALBEDO_SS-Vis": _float_feature_list(row["NOISY_ALBEDO_SS-Vis"]),

        # "NOISY_ALBEDO_B-NIR_norm": _float_feature_list(row["NOISY_ALBEDO_B-NIR_norm"]),
        # "NOISY_ALBEDO_B-UV_norm": _float_feature_list(row["NOISY_ALBEDO_B-UV_norm"]),
        # "NOISY_ALBEDO_B-Vis_norm": _float_feature_list(row["NOISY_ALBEDO_B-Vis_norm"]),

        # "NOISY_ALBEDO_SS-NIR_norm": _float_feature_list(row["NOISY_ALBEDO_SS-NIR_norm"]),
        # "NOISY_ALBEDO_SS-UV_norm": _float_feature_list(row["NOISY_ALBEDO_SS-UV_norm"]),
        # "NOISY_ALBEDO_SS-Vis_norm": _float_feature_list(row["NOISY_ALBEDO_SS-Vis_norm"]),

        "NOISE_B-NIR": _float_feature_list(row["NOISE_B-NIR"]),
        "NOISE_B-UV": _float_feature_list(row["NOISE_B-UV"]),
        "NOISE_B-Vis": _float_feature_list(row["NOISE_B-Vis"]),

        "NOISE_SS-NIR": _float_feature_list(row["NOISE_SS-NIR"]),
        "NOISE_SS-UV": _float_feature_list(row["NOISE_SS-UV"]),
        "NOISE_SS-Vis": _float_feature_list(row["NOISE_SS-Vis"]),

        # Main Features
        "OBJECT-RADIUS-REL-EARTH": _float_feature(row["OBJECT-RADIUS-REL-EARTH"]),
        # "LOG-OBJECT-GRAVITY": _float_feature(row["LOG-OBJECT-GRAVITY"]),
        # "LOG-ATMOSPHERE-TEMPERATURE": _float_feature(row["LOG-ATMOSPHERE-TEMPERATURE"]),
        # "BAR-ATMOSPHERE-PRESSURE": _float_feature(row["BAR-ATMOSPHERE-PRESSURE"]),

        "OBJECT-DIAMETER": _float_feature(row["OBJECT-DIAMETER"]),
        "OBJECT-GRAVITY": _float_feature(row["OBJECT-GRAVITY"]),
        "ATMOSPHERE-TEMPERATURE": _float_feature(row["ATMOSPHERE-TEMPERATURE"]),
        "ATMOSPHERE-PRESSURE": _float_feature(row["ATMOSPHERE-PRESSURE"]),

        "Earth_type": _bytes_feature(row["Earth_type"]),

        # Molecules
        # "C2H6": _float_feature(row["C2H6"]),
        # "CH4": _float_feature(row["CH4"]),
        # "CO": _float_feature(row["CO"]),
        # "CO2": _float_feature(row["CO2"]),
        # "H2O": _float_feature(row["H2O"]),
        # "N2": _float_feature(row["N2"]),
        # "N2O": _float_feature(row["N2O"]),
        # "O2": _float_feature(row["O2"]),
        # "O3": _float_feature(row["O3"]),

        "log_C2H6": _float_feature(row["log_C2H6"]),
        "log_CH4": _float_feature(row["log_CH4"]),
        "log_CO": _float_feature(row["log_CO"]),
        "log_CO2": _float_feature(row["log_CO2"]),
        "log_H2O": _float_feature(row["log_H2O"]),
        "log_N2": _float_feature(row["log_N2"]),
        "log_N2O": _float_feature(row["log_N2O"]),
        "log_O2": _float_feature(row["log_O2"]),
        "log_O3": _float_feature(row["log_O3"]),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# def _zscore_scaler(data: np.ndarray) -> np.ndarray:
#     """
#     Standard Scaling.
#     """
#     return (data - np.mean(data)) / np.std(data)


def _convert_to_earth_radius(data: float) -> float:
    """
    Convert diameter (in km) to Earth"s radii (relative).
    """
    return data / (2 * R_earth.to("km").value)


def _log_data(data: float) -> float:
    """
    Compute the base-10 logarithm of a value.
    """
    return np.log10(data)

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def create_tfrecords(root_folder: str, save_root: str) -> None:
    """
    Traverse a root folder containing subfolders of .parquet files, filter/transform
    the data, and write each filtered DataFrame to TFRecord files.

    Parameters
    ----------
    root_folder : str
        The path to the root directory containing subfolders with .parquet files.
    save_root : str
        The path to the directory where the TFRecord files will be saved.
    """

    # Count the total number of `.parquet` files for progress bar
    file_count = sum(
        len([file for file in files if file.endswith(".parquet")])
        for _, _, files in os.walk(root_folder)
    )

    # Create the destination folder if it doesn't exist
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    with tqdm(
        total=file_count,
        desc="🌍 Progress",
        dynamic_ncols=True,
        colour="cyan",
        bar_format="{desc}: |{bar:30}| {percentage:3.0f}% ({n_fmt}/{total_fmt} files) ⏳ [{elapsed} elapsed]"
    ) as pbar:
        # Iterate through each subfolder
        for folder in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder)
            if not os.path.isdir(folder_path):
                continue  # Skip if not a directory

            files = os.listdir(folder_path)
            for file in files:
                if not file.endswith(".parquet"):
                    continue  # Skip non-parquet files

                file_path = os.path.join(folder_path, file)

                # Extract metadata from filename
                earth_type = file.split("_")[0]
                original_parquet_range = file.split("_")[1].split(".")[0]

                # Read parquet file
                df = pd.read_parquet(file_path)
                df["Earth_type"] = earth_type

                # Filter out rows with noise > 5
                noise_columns = [col for col in df.columns if "NOISE_" in col]
                mask = ~df[noise_columns].applymap(lambda x: any(value > 3 for value in x)).any(axis=1)
                df = df[mask]

                # Keep only columns of interest
                filtered_df = df.copy()
                filtered_df = filtered_df[COLUMNS_OF_INTEREST]

                # Extract abundances using custom function from geexhp
                df = dset.extract_abundances(df)

                # Additional derived features
                filtered_df["OBJECT-RADIUS-REL-EARTH"] = df["OBJECT-DIAMETER"].apply(_convert_to_earth_radius)
                # filtered_df["LOG-OBJECT-GRAVITY"] = df["OBJECT-GRAVITY"].apply(_log_data)
                # filtered_df["LOG-ATMOSPHERE-TEMPERATURE"] = df["ATMOSPHERE-TEMPERATURE"].apply(_log_data)
                # filtered_df["BAR-ATMOSPHERE-PRESSURE"] = df["ATMOSPHERE-PRESSURE"].apply(lambda x: x / 1000)

                # Normalize spectra columns
                # for spectrum in SPECTRA:
                #     filtered_df[f"{spectrum}_norm"] = filtered_df[spectrum].apply(_zscore_scaler)

                # Create log abundance columns for each molecule
                for molecule in MOLECULES:
                    if molecule in df.columns:
                        filtered_df[f"log_{molecule}"] = df[molecule].apply(_log_data)
                    else:
                        # If the molecule column doesn't exist, fill with zero
                        filtered_df[f"log_{molecule}"] = 0

                # Prepare dictionary records
                record_dict = filtered_df.to_dict(orient="records")

                # TFRecord filename
                tfrecord_file = f"{earth_type}_{folder}_{original_parquet_range}_{len(record_dict)}.tfrecord"
                save_path_file = os.path.join(save_root, tfrecord_file)

                # Serialize and write records
                with tf.io.TFRecordWriter(save_path_file) as writer:
                    for sample in record_dict:
                        serialized_sample = _serialize_sample(sample)
                        writer.write(serialized_sample)

                pbar.update(1)  # Update progress
