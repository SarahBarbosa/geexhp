import importlib.resources

import numpy as np
import pandas as pd

from typing import List

from geexhp.core import datamod as dm

LAYERS = 60

def molweightlist(era: str) -> np.ndarray:
    """
    Returns the molecular weights of main molecules based on the given era.

    Parameters
    ----------
    era : str
        The geological era for which the molecular weights are required.
        Supported values are "modern", "proterozoic", and "archean".

    Returns
    -------
    np.ndarray
        An array of molecular weights corresponding to the main molecules of the specified era.

    Notes
    -----
    Molecules included per era:

    - `"modern"` or `"proterozoic"`:
        CO2, O2, H2O, CO, O3, CH4, N2O, N2
    - `"archean"`:
        CH4, CO, H2O, C2H6, CO2, N2

    Molecular weights are sourced from [WebQC Molecular Weight Calculator](https://webqc.org/mmcalc.php).
    """
    molecular_weights_dict = {
        "modern": np.array([44.0095, 31.99880, 18.01528, 28.0101, 47.99820, 16.0425, 44.01280, 28.01340]),
        "proterozoic": np.array([44.0095, 31.99880, 18.01528, 28.0101, 47.99820, 16.0425, 44.01280, 28.01340]),
        "archean": np.array([16.0425, 28.0101, 18.01528, 30.0690, 44.0095, 28.01340])
    }
    era_lower = era.lower()
    if era_lower in molecular_weights_dict:
        return molecular_weights_dict[era_lower]
    else:
        raise ValueError(f"Unsupported era '{era}'. Please choose from 'modern', 'proterozoic', or 'archean'.")

def _process_atmosphere(config: dict, csv_filename: str, molecules: List[str], hitran_dict: dict, era: str) -> None:
    """
    Helper function to process atmospheric data from a CSV file and update the configuration.
    """
    try:
        with importlib.resources.open_text('geexhp.resources.atmos_layers', csv_filename) as f:
            atmos_df = pd.read_csv(f)
    except FileNotFoundError as e:
        print(f"{era.capitalize()} atmospheric data file '{csv_filename}' not found.")
        raise

    abundances = atmos_df.columns[2:]
    atmos_df[abundances] = atmos_df[abundances].div(atmos_df[abundances].sum(axis=1), axis=0)

    molecular_weights = molweightlist(era)
    mmw = atmos_df[abundances].apply(
        lambda row: sum(row[col] * molecular_weights[i] for i, col in enumerate(abundances)),
        axis=1
    ).mean()

    # IT'S 60 LAYERS!!!
    for i in range(len(atmos_df)):
        layer_data = [f'{atmos_df["Pressure"][i]}', f'{atmos_df["Temperature"][i]}']
        layer_data += [f'{atmos_df[molecule][i]}' for molecule in molecules]
        config[f'ATMOSPHERE-LAYER-{i + 1}'] = ','.join(layer_data)

    config['ATMOSPHERE-WEIGHT'] = mmw
    config['ATMOSPHERE-NGAS'] = len(molecules)
    config['ATMOSPHERE-GAS'] = ",".join(molecules)
    config['ATMOSPHERE-TYPE'] = ",".join([hitran_dict[mol] for mol in molecules])
    config['ATMOSPHERE-ABUN'] = "1," * (len(molecules) - 1) + '1'
    config['ATMOSPHERE-UNIT'] = "scl," * (len(molecules) - 1) + 'scl'
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(molecules)
    config["ATMOSPHERE-PRESSURE"] = atmos_df["Pressure"][0] * 1000  # in mbar
    config["SURFACE-TEMPERATURE"] = atmos_df["Temperature"][0]  # in K

def modern(config: dict) -> None:
    """
    Configures the atmosphere for a modern Earth simulation.

    Parameters
    ----------
    config : dict
        The configuration dictionary to update.
    """
    molecules = ["CO2", "O2", "H2O", "CO", "O3", "CH4", "N2O", "N2"]

    # Mapping molecules to HITRAN database indices
    # https://hitran.org/lbl/
    hitran_dict = {
        "CO2": "HIT[2]", "O2": "HIT[7]", "H2O": "HIT[1]",
        "CO": "HIT[5]", "O3": "HIT[3]", "CH4": "HIT[6]",
        "N2O": "HIT[4]", "N2": "HIT[22]"
    }
    _process_atmosphere(config, 'modern.csv', molecules, hitran_dict, era="modern")

def proterozoic(config: dict) -> None:
    """
    Processes atmospheric data for a period 2.0 Ga after the Great Oxidation Event (GOE).
    This is based on scenarios described in the study by Kawashima and Rugheimer (2019).
    
    - The atmospheric data and scenarios are based on the study:
        Kawashima, Y., & Rugheimer, S. (2019). "Spectra of Earth-like Planets Through Geological Evolution
        Around FGKM Stars." The Astronomical Journal, 157(6), 225. DOI: 10.3847/1538-3881/ab14e3
        Available at: https://iopscience.iop.org/article/10.3847/1538-3881/ab14e3

    Parameters
    ----------
    config : dict
        The configuration dictionary to update.
    """
    molecules = ["CO2", "O2", "H2O", "CO", "O3", "CH4", "N2O", "N2"]
    hitran_dict = {
        "CO2": "HIT[2]", "O2": "HIT[7]", "H2O": "HIT[1]",
        "CO": "HIT[5]", "O3": "HIT[3]", "CH4": "HIT[6]",
        "N2O": "HIT[4]", "N2": "HIT[22]"
    }
    _process_atmosphere(config, 'proterozoic.csv', molecules, hitran_dict, era="proterozoic")

def archean(config: dict) -> None:
    """
    Processes atmospheric data for a Archean Earth with CH4/CO2 = 0.2.
    This is based on scenarios described in the study by Arney et al. (2016).

    - The atmospheric data and scenarios are based on the study:
        Arney, Giada, et al. "The pale orange dot: the spectrum and habitability of hazy 
        Archean Earth." Astrobiology 16.11 (2016): 873-899.
        Available at: https://www.liebertpub.com/doi/full/10.1089/ast.2015.1422

    Parameters
    ----------
    config : dict
        The configuration dictionary to update.
    """
    molecules = ["CH4", "CO", "H2O", "C2H6", "CO2", "N2"]
    hitran_dict = {
        "CH4": "HIT[6]", "CO": "HIT[5]", "H2O": "HIT[1]",
        "C2H6": "HIT[27]", "CO2": "HIT[2]", "N2": "HIT[22]"
    }
    _process_atmosphere(config, 'archean.csv', molecules, hitran_dict, era="archean")

def random_atmosphere(config: dict) -> None:
    """
    Generates a completely random atmospheric composition for planetary modeling and 
    populates the provided configuration dictionary with the results.

    The atmosphere is assumed to be isothermal across all layers.

    Parameters
    ----------
    config : dict
        A configuration dictionary where the atmospheric parameters will be stored.
    
    Molecules
    ---------
    The following molecules are considered in the random atmospheric composition:
    
    - H2O: Water vapor
    - CO2: Carbon dioxide
    - CH4: Methane
    - O2: Oxygen
    - NH3: Ammonia
    - HCN: Hydrogen cyanide
    - PH3: Phosphine
    - H2: Molecular hydrogen
    """
    layers = LAYERS 

    dm.set_spectral_type(config)
    dm.set_stellar_parameters(config)
    dm.set_solar_coordinates(config)
    dm.set_habitable_zone_distance(config)
    dm.maintain_planetary_atmosphere(config)

    molecular_weights = {
        'H2O': 18.01528, 'CO2': 44.0095, 'CH4': 16.0425, 'O2': 31.99880,
        'NH3': 17.03052, 'HCN': 27.0253, 'PH3': 33.99758, 'H2': 2.01588
    }
    hitran_dict = {
        'H2O': 'HIT[1]', 'CO2': 'HIT[2]', 'CH4': 'HIT[6]', 
        'O2': 'HIT[7]', 'NH3': 'HIT[11]', 'HCN': 'HIT[23]',
        'PH3': 'HIT[28]', 'H2': 'HIT[45]',
    }

    molecules = list(molecular_weights.keys())

    sample = {molecule: np.random.lognormal(mean=-13, sigma=1) for molecule in molecules}
    total_concentration = sum(sample.values())
    normalized_sample = {molecule: value / total_concentration for molecule, value in sample.items()}
    layer_concentrations = {molecule: np.full(layers, concentration) for molecule, concentration in normalized_sample.items()}

    # Atmospheric parameters setup
    # 1e-11 = Earth’s upper atmosphere (Fuller-Rowell, 2014) in bar
    # Fuller-Rowell, T. (2014). Physical Characteristics and Modeling of Earth’s Thermosphere. 
    pressure_top = 1e-11  # in bar
    pressure_base = config["ATMOSPHERE-PRESSURE"] / 1000  # Convert to bar
    pressure = np.logspace(np.log10(pressure_base), np.log10(pressure_top), num=layers)
    temperature = np.full(layers, config["ATMOSPHERE-TEMPERATURE"])

    # Populate atmospheric layers
    for i in range(layers):
        layer_data = [f'{pressure[i]}', f'{temperature[i]}']
        layer_data += [f'{layer_concentrations[molecule][i]}' for molecule in layer_concentrations]
        config[f'ATMOSPHERE-LAYER-{i + 1}'] = ','.join(layer_data)

    # Additional atmosphere configurations
    average_molecular_weight = sum(normalized_sample[molecule] * molecular_weights[molecule] for molecule in normalized_sample)
    config['ATMOSPHERE-WEIGHT'] = average_molecular_weight
    config['ATMOSPHERE-NGAS'] = len(molecules)
    config['ATMOSPHERE-GAS'] = ",".join(molecules)
    config['ATMOSPHERE-TYPE'] = ",".join([hitran_dict[mol] for mol in molecules])
    config['ATMOSPHERE-ABUN'] = "1," * (len(molecules) - 1) + '1'
    config['ATMOSPHERE-UNIT'] = "scl," * (len(molecules) - 1) + 'scl'
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(molecules)