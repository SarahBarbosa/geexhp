import os
from tqdm import tqdm
from typing import List, Dict, Union

import numpy as np
import pandas as pd

import tensorflow as tf

from astropy.constants import R_earth
from geexhp.modelfuncs import datasetup as dset


class TFRecordConfig:
    """
    Configuration for TFRecord conversion.
    """

    COLUMNS_OF_INTEREST: List[str] = [
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
        "OBJECT-STAR-TYPE",
        "GEOMETRY-OBS-ALTITUDE",
        "OBJECT-INCLINATION",
        "OBJECT-SEASON",
        "OBJECT-SOLAR-LONGITUDE",
        "OBJECT-SOLAR-LATITUDE",
        "OBJECT-PHASE-ANGLE",
    ]

    MOLECULES: List[str] = [
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

    # SPECTRA: List[str] = [
    #     "NOISY_ALBEDO_B-NIR",
    #     "NOISY_ALBEDO_B-UV",
    #     "NOISY_ALBEDO_B-Vis",
    #     "NOISY_ALBEDO_SS-NIR",
    #     "NOISY_ALBEDO_SS-UV",
    #     "NOISY_ALBEDO_SS-Vis",
    # ]


def _bytes_feature(value: str) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _float_feature(value: float) -> tf.train.Feature:
    """Returns a float_list from a float / list of floats."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value: List[float]) -> tf.train.Feature:
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
        "NOISE_B-NIR": _float_feature_list(row["NOISE_B-NIR"]),
        "NOISE_B-UV": _float_feature_list(row["NOISE_B-UV"]),
        "NOISE_B-Vis": _float_feature_list(row["NOISE_B-Vis"]),
        "NOISE_SS-NIR": _float_feature_list(row["NOISE_SS-NIR"]),
        "NOISE_SS-UV": _float_feature_list(row["NOISE_SS-UV"]),
        "NOISE_SS-Vis": _float_feature_list(row["NOISE_SS-Vis"]),
        "SNR_B-NIR": _float_feature_list(row["SNR_B-NIR"]),
        "SNR_B-UV": _float_feature_list(row["SNR_B-UV"]),
        "SNR_B-Vis": _float_feature_list(row["SNR_B-Vis"]),
        "SNR_SS-NIR": _float_feature_list(row["SNR_SS-NIR"]),
        "SNR_SS-UV": _float_feature_list(row["SNR_SS-UV"]),
        "SNR_SS-Vis": _float_feature_list(row["SNR_SS-Vis"]),
        "SNR_FEATURE_B-NIR": _float_feature(row["SNR_FEATURE_B-NIR"]),
        "SNR_FEATURE_B-UV": _float_feature(row["SNR_FEATURE_B-UV"]),
        "SNR_FEATURE_B-Vis": _float_feature(row["SNR_FEATURE_B-Vis"]),
        "SNR_FEATURE_SS-NIR": _float_feature(row["SNR_FEATURE_SS-NIR"]),
        "SNR_FEATURE_SS-UV": _float_feature(row["SNR_FEATURE_SS-UV"]),
        "SNR_FEATURE_SS-Vis": _float_feature(row["SNR_FEATURE_SS-Vis"]),
        "SNR_FEATURE_PCTL_B-NIR": _float_feature(row["SNR_FEATURE_PCTL_B-NIR"]),
        "SNR_FEATURE_PCTL_B-UV": _float_feature(row["SNR_FEATURE_PCTL_B-UV"]),
        "SNR_FEATURE_PCTL_B-Vis": _float_feature(row["SNR_FEATURE_PCTL_B-Vis"]),
        "SNR_FEATURE_PCTL_SS-NIR": _float_feature(row["SNR_FEATURE_PCTL_SS-NIR"]),
        "SNR_FEATURE_PCTL_SS-UV": _float_feature(row["SNR_FEATURE_PCTL_SS-UV"]),
        "SNR_FEATURE_PCTL_SS-Vis": _float_feature(row["SNR_FEATURE_PCTL_SS-Vis"]),
        # Main Features
        "OBJECT-RADIUS-REL-EARTH": _float_feature(row["OBJECT-RADIUS-REL-EARTH"]),
        "OBJECT-DIAMETER": _float_feature(row["OBJECT-DIAMETER"]),
        "OBJECT-GRAVITY": _float_feature(row["OBJECT-GRAVITY"]),
        "ATMOSPHERE-TEMPERATURE": _float_feature(row["ATMOSPHERE-TEMPERATURE"]),
        "ATMOSPHERE-PRESSURE": _float_feature(row["ATMOSPHERE-PRESSURE"]),
        "Earth_type": _bytes_feature(row["Earth_type"]),
        "OBJECT-STAR-TYPE": _bytes_feature(row["OBJECT-STAR-TYPE"]),
        "GEOMETRY-OBS-ALTITUDE": _float_feature(row["GEOMETRY-OBS-ALTITUDE"]),
        "OBJECT-INCLINATION": _float_feature(row["OBJECT-INCLINATION"]),
        "OBJECT-SEASON": _float_feature(row["OBJECT-SEASON"]),
        "OBJECT-SOLAR-LONGITUDE": _float_feature(row["OBJECT-SOLAR-LONGITUDE"]),
        "OBJECT-SOLAR-LATITUDE": _float_feature(row["OBJECT-SOLAR-LATITUDE"]),
        "OBJECT-PHASE-ANGLE": _float_feature(row["OBJECT-PHASE-ANGLE"]),
        # Molecules
        "C2H6": _float_feature(row["C2H6"]),
        "CH4": _float_feature(row["CH4"]),
        "CO": _float_feature(row["CO"]),
        "CO2": _float_feature(row["CO2"]),
        "H2O": _float_feature(row["H2O"]),
        "N2": _float_feature(row["N2"]),
        "N2O": _float_feature(row["N2O"]),
        "O2": _float_feature(row["O2"]),
        "O3": _float_feature(row["O3"]),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _convert_to_earth_radius(data: float) -> float:
    """
    Convert diameter (in km) to Earth"s radii (relative).
    """
    return data / (2 * R_earth.to("km").value)


def _calculate_feature_snr(signal_array, noise_array):
    s = np.array(signal_array, dtype=np.float32)
    n = np.array(noise_array, dtype=np.float32)

    amplitude = np.max(s) - np.min(s)
    avg_noise = np.mean(n)

    if avg_noise == 0:
        return 0.0

    return amplitude / avg_noise


def _calculate_feature_snr(signal_array, noise_array):
    """
    Scalar SNR based on:
    SNR = (S_max - S_min) / sqrt(NOISE_max^2 + NOISE_min^2)

    where NOISE_max and NOISE_min are the noise at the same
    wavelengths where the signal reaches maximum and minimum.
    """
    s = np.array(signal_array, dtype=np.float32)
    n = np.array(noise_array, dtype=np.float32)

    idx_max = np.argmax(s)
    idx_min = np.argmin(s)

    s_max = s[idx_max]
    s_min = s[idx_min]

    n_max = n[idx_max]
    n_min = n[idx_min]

    amplitude = s_max - s_min
    denom = np.sqrt(n_max**2 + n_min**2)

    return amplitude / denom


def _calculate_feature_snr_percentile(signal_array, noise_array, p_low=10, p_high=90):
    """
    Alternative scalar SNR using a percentile-based amplitude:
        amplitude = S_p_high - S_p_low
    where S_p are the p-th percentiles of the signal.

    As a simple approximation for the noise of a difference
    between two points, we use:
        sigma_diff ~ sqrt(2) * sigma_rms(noise).
    """
    s = np.array(signal_array, dtype=np.float32)
    n = np.array(noise_array, dtype=np.float32)

    s_low = np.percentile(s, p_low)
    s_high = np.percentile(s, p_high)
    amplitude = float(s_high - s_low)

    # RMS noise as a typical sigma
    sigma = float(np.sqrt(np.mean(n**2)))
    denom = np.sqrt(2.0) * sigma

    return amplitude / denom


def _safe_vector_snr(signal_array, noise_array):
    s = np.array(signal_array, dtype=np.float32)
    n = np.array(noise_array, dtype=np.float32)
    return np.divide(s, n, out=np.zeros_like(s), where=n != 0)


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

    file_count = sum(
        len([file for file in files if file.endswith(".parquet")])
        for _, _, files in os.walk(root_folder)
    )

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    with tqdm(
        total=file_count,
        desc="🌍 Progress",
        dynamic_ncols=True,
        colour="cyan",
        bar_format="{desc}: |{bar:30}| {percentage:3.0f}% ({n_fmt}/{total_fmt} files) ⏳ [{elapsed} elapsed]",
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
                df["OBJECT-RADIUS-REL-EARTH"] = df["OBJECT-DIAMETER"].apply(
                    _convert_to_earth_radius
                )

                df_abundances = dset.extract_abundances(df)
                for molecule in TFRecordConfig.MOLECULES:
                    if molecule in df_abundances.columns:
                        df[molecule] = df_abundances[molecule]
                    else:
                        df[molecule] = 0.0

                # Filter out rows with noise > 3
                # noise_columns = [col for col in df.columns if "NOISE_" in col]
                # mask = ~df[noise_columns].map(lambda x: any(value > 3 for value in x)).any(axis=1)
                # df = df[mask]

                telescopes = ["B", "SS"]
                bands = ["UV", "Vis", "NIR"]

                new_vector_cols = []
                new_scalar_cols = []

                for tel in telescopes:
                    for band in bands:
                        signal_col = f"ALBEDO_{tel}-{band}"
                        noise_col = f"NOISE_{tel}-{band}"

                        snr_vec_col = f"SNR_{tel}-{band}"
                        snr_feat_col = f"SNR_FEATURE_{tel}-{band}"
                        snr_feat_pctl_col = f"SNR_FEATURE_PCTL_{tel}-{band}"

                        if signal_col in df.columns and noise_col in df.columns:
                            # Vector SNR: S / N (per wavelength bin)
                            snr_vectors = [
                                _safe_vector_snr(s, n)
                                for s, n in zip(df[signal_col], df[noise_col])
                            ]
                            df[snr_vec_col] = snr_vectors
                            new_vector_cols.append(snr_vec_col)

                            # Scalar SNR feature: (S_max - S_min) / sqrt(N_max^2 + N_min^2)
                            snr_scalars = [
                                _calculate_feature_snr(s, n)
                                for s, n in zip(df[signal_col], df[noise_col])
                            ]
                            df[snr_feat_col] = snr_scalars
                            new_scalar_cols.append(snr_feat_col)

                            # Percentile-based scalar SNR feature: (P90 - P10) / (sqrt(2)*RMS(noise))
                            snr_scalars_pctl = [
                                _calculate_feature_snr_percentile(s, n)
                                for s, n in zip(df[signal_col], df[noise_col])
                            ]
                            df[snr_feat_pctl_col] = snr_scalars_pctl
                            new_scalar_cols.append(snr_feat_pctl_col)

                cols_to_keep = TFRecordConfig.COLUMNS_OF_INTEREST.copy()

                cols_to_keep.extend(new_vector_cols)
                cols_to_keep.extend(new_scalar_cols)
                cols_to_keep.extend(TFRecordConfig.MOLECULES)

                if "OBJECT-RADIUS-REL-EARTH" not in cols_to_keep:
                    cols_to_keep.append("OBJECT-RADIUS-REL-EARTH")

                final_cols = [c for c in cols_to_keep if c in df.columns]

                filtered_df = df[final_cols].copy()

                record_dict = filtered_df.to_dict(orient="records")

                tfrecord_file = f"{earth_type}_{folder}_{original_parquet_range}_{len(record_dict)}.tfrecord"
                save_path_file = os.path.join(save_root, tfrecord_file)

                with tf.io.TFRecordWriter(save_path_file) as writer:
                    for sample in record_dict:
                        serialized_sample = _serialize_sample(sample)
                        writer.write(serialized_sample)

                pbar.update(1)
