import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MeanNormalization:
    """
    Custom scaler to perform mean normalization, where the data is scaled by subtracting the mean
    and dividing by the range (max - min).
    """
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.range = self.max - self.min
        return self

    def transform(self, data):
        return (data - self.mean) / self.range

    def inverse_transform(self, data):
        return (data * self.range) + self.mean

def combine_parquet(folder: str, keyword: str, output_file: bool = False) -> pd.DataFrame:
    """
    Combine all .parquet files.

    Parameters
    ----------
    folder : str
        The path to the folder containing the .parquet files.
    keyword : str
        The keyword used to filter .parquet files (e.g., "modern", "proterozoic", "archean", "random"). 
        Only files containing this keyword in their filenames will be processed.
    output_file : bool, optional
        If True, the combined DataFrame is saved as a .parquet file in the ../data folder.
        If False, the combined DataFrame is not saved and only returned.

    Returns
    -------
    pd.DataFrame
        A combined DataFrame containing the data from all .parquet files that match the keyword.
        If no matching files are found, an empty DataFrame is returned.

    Notes
    -----
    THIS FUNCTION CAN CONSUME A LOT OF CPU, ESPECIALLY WHEN DEALING WITH LARGE PARQUET FILES.
    MAKE SURE YOUR SYSTEM HAS SUFFICIENT RESOURCES BEFORE RUNNING THIS OPERATION.
    """
    file_pattern = os.path.join(folder, f'*{keyword}*.parquet')
    files = glob.glob(file_pattern)

    if not files:
        print(f"No files found with keyword '{keyword}' in the folder '{folder}'")
        return pd.DataFrame()

    combineddf = pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)

    # Optionally, save the combined DataFrame as a new .parquet file
    if output_file:
        output_dir = '../data'
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
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing `'ATMOSPHERE-LAYER'` and `'ATMOSPHERE-LAYERS-MOLECULES'`.
    
    Returns
    -------
    pd.DataFrame
        The DataFrame with new columns for each molecule's abundance. Missing molecules are filled with 0.
    """
    molecules = set()
    df["ATMOSPHERE-LAYERS-MOLECULES"].apply(lambda x: molecules.update(x.split(',')))
    sorted_molecules = sorted(molecules)
    abun_df = df.apply(lambda row: _extract(row, sorted_molecules), axis=1, result_type='expand')
    abun_df.columns = [f"{molecule}" for molecule in sorted_molecules]
    return pd.concat([df, abun_df], axis=1)

def windowed_normalization(X_train, X_test, y_train, y_test, wavelengths, wavelength_intervals, x_scaler=None, y_scaler=None):
    """
    Normalizes the input ALBEDO data (X_train and X_test) for each spectral window defined by wavelength_intervals.
    Scales input and output data, applying mean normalization to the output (y).

    Parameters
    -----------
    X_train : array-like
        Training input ALBEDO data.
    X_test : array-like
        Test input ALBEDO data.
    y_train : array-like
        Training output labels (target values).
    y_test : array-like
        Test output labels (target values).
    wavelengths : array-like
        Global array of WAVELENGTH values corresponding to the ALBEDO data.
    wavelength_intervals : list of tuples
        A list of tuples where each tuple represents a spectral window in wavelength values, e.g., [(0.2, 0.3), (0.3, 0.4)].
    x_scaler : list of scalers or None
        If None, new MinMaxScalers will be created for each window. If a list, existing scalers are used.
    y_scaler : scaler or None
        If None, mean normalization will be applied to the output data (y).

    Returns
    --------
    X_train_scaled : array-like
        Scaled and reshaped training input data.
    X_test_scaled : array-like
        Scaled and reshaped test input data.
    y_train_scaled : array-like
        Scaled training output data.
    y_test_scaled : array-like
        Scaled test output data.
    x_scalers : list of scalers
        A list of scalers used for each spectral window.
    y_scaler : scaler
        Scaler used for the output data.
    """
    if x_scaler is None:
        x_scalers = [MinMaxScaler() for _ in wavelength_intervals]
    else:
        x_scalers = x_scaler

    # Use MeanNormalization for output data if y_scaler is None
    if y_scaler is None:
        y_scaler = StandardScaler()

    # Helper function to apply windowed normalization to ALBEDO values
    def normalize_albedo(X, wavelengths, scalers):
        albedo_normalized = []
        for albedo in X:
            albedo_scaled = np.zeros_like(albedo)

            # Convert wavelength intervals to indices and normalize each spectral window
            for j, (start_wavelength, end_wavelength) in enumerate(wavelength_intervals):
                # Get the indices corresponding to the start and end wavelength
                start_idx = np.abs(wavelengths - start_wavelength).argmin()
                end_idx = np.abs(wavelengths - end_wavelength).argmin()

                # Normalize the spectral window using MinMaxScaler to scale between 0 and 1
                albedo_window = albedo[start_idx:end_idx]
                albedo_scaled[start_idx:end_idx] = scalers[j].fit_transform(albedo_window.reshape(-1, 1)).flatten()

            albedo_normalized.append(albedo_scaled)
        
        return np.array(albedo_normalized)

    # Apply windowed normalization to X_train and X_test
    X_train_scaled = normalize_albedo(X_train, wavelengths, x_scalers)
    X_test_scaled = normalize_albedo(X_test, wavelengths, x_scalers)

    # Reshape the data to be (samples, features, 1) for Conv1D layers
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # Apply mean normalization (or other scaler) to the output data (y_train and y_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Print shapes for debugging
    print(f"Train input shape: {X_train_scaled.shape}")
    print(f"Test input shape: {X_test_scaled.shape}")
    print(f"Train labels shape: {y_train_scaled.shape}")
    print(f"Test labels shape: {y_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scalers, y_scaler

