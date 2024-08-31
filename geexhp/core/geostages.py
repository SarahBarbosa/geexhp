import os
import numpy as np
import pandas as pd
from scipy.stats import beta
from geexhp.core import datamod as dm

def molweightlist() -> np.ndarray:
    """
    Returns the molecular weights of main molecules.

    The molecules included are:
    - CO2  : Carbon Dioxide
    - O2   : Oxygen
    - H2O  : Water
    - CO   : Carbon Monoxide
    - O3   : Ozone
    - CH4  : Methane
    - N2O  : Nitrous Oxide
    - N2   : Nitrogen
    """
    # Information about molecules (Reference: https://pt.webqc.org/mmcalc.php)
    molecular_weights = np.array([
        44.0095, 31.99880, 18.01528, 28.0101, 47.99820, 16.0425, 44.01280, 28.01340])
    return molecular_weights

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
    Processes atmospheric data from a CSV file for a period 2.0 Ga after the Great Oxidation Event (GOE),
    normalizes gas abundances, calculates the mean molecular weight, and updates a configuration dictionary
    with atmospheric parameters. This function is based on scenarios described in the study by Kawashima and
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

    mmw = proterozoic[abun].apply(lambda row: sum(row[col] * molweightlist()[i] for i, col in enumerate(abun)), axis=1).mean()
        
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

def random_atmosphere(config: dict) -> None:
    """
    Generates a completely random atmospheric composition for planetary modeling and 
    populates the provided configuration dictionary with the results.

    The atmosphere is assumed to be isothermal across all layers.

    Parameters
    ----------
    config : dict
        A configuration dictionary where the atmospheric parameters will be stored.

    Returns
    -------
    None
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
        'SO2': 64.0638,   
        'H2S': 34.0809   
    }
    HITRAN_DICT = {
        'H2O': 'HIT[1]', 'CO2': 'HIT[2]', 'CH4': 'HIT[6]', 
        'O2': 'HIT[7]', 'NH3': 'HIT[11]', 'HCN': 'HIT[23]',
        'PH3': 'HIT[28]', 'SO2': 'HIT[9]', 'H2S': 'HIT[31]'
    }

    molecules = ['H2O', 'CO2', 'CH4', 'O2', 'NH3', 'HCN', 'PH3', 'SO2', 'H2S']

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