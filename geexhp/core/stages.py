import os
import numpy as np
import pandas as pd
from scipy.stats import beta
from geexhp.core import datamod as dm

def molweightlist(era: str) -> np.ndarray:
    """
    Returns the molecular weights of main molecules based on the given era.
    Source: https://pt.webqc.org/mmcalc.php
    
    Parameters
    ----------
    era : str 
        The geological era for which the molecular weights are required. 
        Supported values are "modern", "proterozoic", and "archean".
    
    Returns
    -------
    np.ndarray

    Molecules
    ---------
    - "modern" or "proterozoic":
        CO2, O2, H2O, CO, O3, CH4, N2O, N2
    - "archean":
        CH4, CO, H2O, C2H6, CO2, N2
    """
    molecular_weights_dict = {
        "modern": np.array([44.0095, 31.99880, 18.01528, 28.0101, 47.99820, 16.0425, 44.01280, 28.01340]),
        "proterozoic": np.array([44.0095, 31.99880, 18.01528, 28.0101, 47.99820, 16.0425, 44.01280, 28.01340]),
        "archean": np.array([16.0425, 28.0101, 18.01528, 30.0690, 44.0095, 28.01340])
    }
    try:
        return molecular_weights_dict[era.lower()]
    except KeyError:
        raise ValueError(f"Unsupported era '{era}'. Please choose from 'modern', 'proterozoic', or 'archean'.")


def modern(config: dict) -> None:
    """
    Calculate atmospheric parameters for a modern Earth simulation.

    Parameters
    ----------
    config : dict
        A dictionary to store the calculated atmospheric parameters.

    Returns
    -------
    None

    Notes
    -----
    This function models atmospheric parameters of a modern Earth, including pressure, 
    temperature, and the abundance of various gases in different atmospheric layers.
    Gas abundances are provided as a fraction relative to the total mixture in each layer.
    """
    layers = 60

    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "atmos_layers", "modern.csv"))
    modern = pd.read_csv(csv_path)

    molecules = ["CO2", "O2", "H2O", "CO", "O3", "CH4", "N2O", "N2"]
    
    # Mapping molecules to HITRAN database indices
    # https://hitran.org/lbl/
    HITRAN_DICT = {"CO2": "HIT[2]", "O2" : "HIT[7]", "H2O": "HIT[1]",
                    "CO": "HIT[5]", "O3":"HIT[3]", "CH4" :"HIT[6]", 
                    "N2O": "HIT[4]", "N2" : "HIT[22]"}
        
    for i in range(layers):
        layer_data = [f'{modern["Pressure"][i]}', f'{modern["Temperature"][i]}']
        layer_data += [f'{modern[molecule][i]}' for molecule in molecules]
        config[f'ATMOSPHERE-LAYER-{i + 1}'] = ','.join(layer_data)

    config['ATMOSPHERE-NGAS'] = len(molecules)
    config['ATMOSPHERE-GAS'] = ",".join(molecules)
    config['ATMOSPHERE-TYPE'] = ",".join([HITRAN_DICT[mol] for mol in molecules]) 
    config['ATMOSPHERE-ABUN'] = "1," * (len(molecules) - 1) + '1'
    config['ATMOSPHERE-UNIT'] = "scl," * (len(molecules) - 1) + 'scl' 
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(molecules)

def proterozoic(config: dict) -> None:
    """
    Processes atmospheric data for a period 2.0 Ga after the Great Oxidation Event (GOE),
    normalizes gas abundances, calculates the mean molecular weight, and updates a configuration dictionary
    with atmospheric parameters. This is based on scenarios described in the study by Kawashima and
    Rugheimer (2019).

    Parameters
    ----------
    config : dict
        A dictionary where atmospheric parameters will be stored. The dictionary is updated
        with various keys representing atmospheric properties like layer data, gas composition,
        molecular weight, and other relevant settings.

    Notes
    -----
    - The CSV file is expected to be in "../data/atmos_layers/" directory.
    - The normalization assumes that the sum of gas abundances in each layer should equal 1.
    - The mean molecular weight is computed based on predefined molecular weights associated
        with each gas.
    - The atmospheric data and scenarios are based on the study:
        Kawashima, Y., & Rugheimer, S. (2019). "Spectra of Earth-like Planets Through Geological Evolution
        Around FGKM Stars." The Astronomical Journal, 157(6), 225. DOI: 10.3847/1538-3881/ab14e3
        Available at: https://iopscience.iop.org/article/10.3847/1538-3881/ab14e3
    """
    layers = 60

    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "atmos_layers", "proterozoic.csv"))
    proterozoic = pd.read_csv(csv_path)
    
    molecules = ["CO2", "O2", "H2O", "CO", "O3", "CH4", "N2O", "N2"]
    HITRAN_DICT = {"CO2": "HIT[2]", "O2" : "HIT[7]", "H2O": "HIT[1]",
                    "CO": "HIT[5]", "O3":"HIT[3]", "CH4" :"HIT[6]", 
                    "N2O": "HIT[4]", "N2" : "HIT[22]"}
    
    abun = proterozoic.columns[2:]
    proterozoic[abun] = proterozoic[abun].div(proterozoic[abun].sum(axis=1), axis=0)

    mmw = proterozoic[abun].apply(lambda row: sum(row[col] * molweightlist("proterozoic")[i] for i, col in enumerate(abun)), axis=1).mean()
        
    for i in range(layers):
        layer_data = [f'{proterozoic["Pressure"][i]}', f'{proterozoic["Temperature"][i]}']
        layer_data += [f'{proterozoic[molecule][i]}' for molecule in molecules]
        config[f'ATMOSPHERE-LAYER-{i + 1}'] = ','.join(layer_data)

    config['ATMOSPHERE-WEIGHT'] = mmw
    config['ATMOSPHERE-NGAS'] = len(molecules)
    config['ATMOSPHERE-GAS'] = ",".join(molecules)
    config['ATMOSPHERE-TYPE'] = ",".join([HITRAN_DICT[mol] for mol in molecules]) 
    config['ATMOSPHERE-ABUN'] = "1," * (len(molecules) - 1) + '1'
    config['ATMOSPHERE-UNIT'] = "scl," * (len(molecules) - 1) + 'scl' 
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(molecules)
    config["ATMOSPHERE-PRESSURE"] = proterozoic["Pressure"][0] * 1000   # in mbar
    config["SURFACE-TEMPERATURE"] = proterozoic["Temperature"][0] # in K

def archean(config: dict) -> None:
    """
    Processes atmospheric data for a Archean Earth (without hazy) with CH4/CO2 = 0.1, normalizes gas 
    bundances, calculates the mean molecular weight, and updates a configuration dictionary
    with atmospheric parameters. This is based on scenarios described in the study by Arney et al. (2016).

    Parameters
    ----------
    config : dict
        A dictionary where atmospheric parameters will be stored. The dictionary is updated
        with various keys representing atmospheric properties like layer data, gas composition,
        molecular weight, and other relevant settings.

    Notes
    -----
    - The CSV file is expected to be in "../data/atmos_layers/" directory.
    - The normalization assumes that the sum of gas abundances in each layer should equal 1.
    - The mean molecular weight is computed based on predefined molecular weights associated
        with each gas.
    - The atmospheric data and scenarios are based on the study:
        Arney, Giada, et al. "The pale orange dot: the spectrum and habitability of hazy 
        Archean Earth." Astrobiology 16.11 (2016): 873-899.
        Available at: https://www.liebertpub.com/doi/full/10.1089/ast.2015.1422
    """
    layers = 60

    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "atmos_layers", "archean.csv"))
    archean = pd.read_csv(csv_path)
    
    molecules = ["CH4","CO","H2O","C2H6","CO2", "N2"]
    HITRAN_DICT = {"CH4" :"HIT[6]", "CO": "HIT[5]", "H2O": "HIT[1]", "C2H6": "HIT[27]", "CO2": "HIT[2]",
                    "N2" : "HIT[22]"}
    
    abun = archean.columns[2:]
    archean[abun] = archean[abun].div(archean[abun].sum(axis=1), axis=0)

    mmw = archean[abun].apply(lambda row: sum(row[col] * molweightlist("archean")[i] for i, col in enumerate(abun)), axis=1).mean()
        
    for i in range(layers):
        layer_data = [f'{archean["Pressure"][i]}', f'{archean["Temperature"][i]}']
        layer_data += [f'{archean[molecule][i]}' for molecule in molecules]
        config[f'ATMOSPHERE-LAYER-{i + 1}'] = ','.join(layer_data)

    config['ATMOSPHERE-WEIGHT'] = mmw
    config['ATMOSPHERE-NGAS'] = len(molecules)
    config['ATMOSPHERE-GAS'] = ",".join(molecules)
    config['ATMOSPHERE-TYPE'] = ",".join([HITRAN_DICT[mol] for mol in molecules]) 
    config['ATMOSPHERE-ABUN'] = "1," * (len(molecules) - 1) + '1'
    config['ATMOSPHERE-UNIT'] = "scl," * (len(molecules) - 1) + 'scl' 
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(molecules)
    config["ATMOSPHERE-PRESSURE"] = archean["Pressure"][0] * 1000   # in mbar
    config["SURFACE-TEMPERATURE"] = archean["Temperature"][0] # in K

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
    dm.set_spectral_type(config)
    dm.set_stellar_parameters(config)
    dm.set_solar_coordinates(config)
    dm.set_habitable_zone_distance(config)
    dm.maintain_planetary_atmosphere(config)

    layers = 60 
    molecular_weights = {
        'H2O': 18.01528,  
        'CO2': 44.0095,   
        'CH4': 16.0425,   
        'O2': 31.99880,   
        'NH3': 17.03052,  
        'HCN': 27.0253,   
        'PH3': 33.99758,  
        'H2': 2.01588,   
    }
    HITRAN_DICT = {
        'H2O': 'HIT[1]', 'CO2': 'HIT[2]', 'CH4': 'HIT[6]', 
        'O2': 'HIT[7]', 'NH3': 'HIT[11]', 'HCN': 'HIT[23]',
        'PH3': 'HIT[28]', 'H2': 'HIT[45]',
    }

    molecules = ['H2O', 'CO2', 'CH4', 'O2', 'NH3', 'HCN', 'PH3', 'H2']

    sample = {}
    for molecule in molecules:
        sample[molecule] = np.random.lognormal(-13, 1)

    total_concentration = sum(sample.values())
    normalized_sample = {molecule: value / total_concentration for molecule, value in sample.items()}
    layer_concentrations = {molecule: np.full(layers, concentration) for molecule, concentration in normalized_sample.items()}

    # Atmospheric parameters setup
    # 1e-11 = Earth’s upper atmosphere (Fuller-Rowell, 2014) in bar
    # Fuller-Rowell, T. (2014). Physical Characteristics and Modeling of Earth’s Thermosphere. 
    pressure = np.logspace(np.log10(config["ATMOSPHERE-PRESSURE"] / 1000), np.log10(1e-11), num=layers)
    temperature = np.ones(len(pressure)) * config["ATMOSPHERE-TEMPERATURE"]

    # Populate atmospheric layers
    for i in range(layers):
        layer_data = [f'{pressure[i]}', f'{temperature[i]}']
        layer_data += [f'{layer_concentrations[molecule][i]}' for molecule in layer_concentrations]
        config[f'ATMOSPHERE-LAYER-{i + 1}'] = ','.join(layer_data)

    # Additional atmosphere configurations
    average_molecular_weight = sum(normalized_sample[molecule] * molecular_weights[molecule] for molecule in normalized_sample)
    config['ATMOSPHERE-WEIGHT'] = average_molecular_weight
    config['ATMOSPHERE-NGAS'] = len(sample)
    config['ATMOSPHERE-GAS'] = ",".join(sample.keys())
    config['ATMOSPHERE-TYPE'] = ",".join([HITRAN_DICT[mol] for mol in sample.keys()]) 
    config['ATMOSPHERE-ABUN'] = "1," * (len(sample) - 1) + '1'
    config['ATMOSPHERE-UNIT'] = "scl," * (len(sample) - 1) + 'scl' 
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(sample.keys())