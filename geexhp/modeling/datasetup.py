import os
import glob
import pandas as pd

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

    # Row-wise expand
    abun_df = df.apply(lambda row: _extract(row, sorted_molecules), axis=1, result_type="expand")
    abun_df.columns = [f"{molecule}" for molecule in sorted_molecules]
    
    return pd.concat([df, abun_df], axis=1)
