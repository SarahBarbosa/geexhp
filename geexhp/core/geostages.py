import os
import numpy as np
import pandas as pd
from scipy.stats import beta
from geexhp.core import datamod as dm

def molweight_modern() -> np.ndarray:
    """
    Returns the molecular weights of elements in the modern Earth's atmosphere.

    The molecules included are:
    - CO2 : Carbon Dioxide
    - N2  : Nitrogen
    - O2  : Oxygen
    - H2O : Water
    - CO  : Carbon Monoxide
    - C2H6: Ethane
    - HCN : Hydrogen Cyanide
    - SO2 : Sulfur Dioxide
    - O3  : Ozone
    - CH4 : Methane
    - N2O : Nitrous Oxide
    - NH3 : Ammonia
    """
    # Information about molecules (Reference: https://pt.webqc.org/mmcalc.php)
    molecular_weights = np.array([
        44.0095, 28.01340, 31.99880, 18.01528, 28.0101, 30.0690, 27.0253, 
        64.0638, 47.99820, 16.0425, 44.01280, 17.03052])
    return molecular_weights

def molweight_after_goe() -> np.ndarray:
    """
    Returns the molecular weights of elements in 2.0 Ga after the 
    Great Oxidation Event.

    The molecules included are:
    - H2O : Water
    - N2O : Nitrous Oxide
    - O3  : Ozone
    - CO  : Carbon Monoxide
    - CO2 : Carbon Dioxide
    - O2  : Oxygen
    - CH4 : Methane
    - N2  : Nitrogen
    """
    molecular_weights = np.array([
        18.01528, 44.01280, 47.99820, 28.0101, 44.0095, 31.99880, 16.0425, 28.01340 
        ])
    return molecular_weights   

def isothermal_PT(layers: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates an isothermal pressure-temperature profile for a planetary atmosphere,
    using a Beta distribution to model atmospheric pressures and temperatures 
    between values typical for Mars and a theoretical upper limit less favorable than Venus.

    Parameters:
    -----------
    layers : int
        The number of atmospheric layers for which the pressure values are generated.
        This defines the vertical resolution of the model.

    Returns:
    --------
    tuple (np.ndarray, np.ndarray)
        A tuple containing two arrays:
        - The first array contains logarithmically spaced pressure values from a near vacuum
            to a scaled pressure value.
        - The second array contains a constant temperature value replicated across all layers.

    Physics Context:
    ----------------
    - The pressure values are generated to simulate a range for a terrestrial atmosphere, 
        bounded by extreme low (Mars-like) and moderate high pressures, avoiding extremes 
        like those on Venus.
    - The temperature is set using a Beta distribution to represent variability in planetary 
        atmospheric temperatures, constrained within a range that approximates Mars-like to 
        ~ 100°C 
    - This greatly simplifies the forward model and is valid because the reflected light 
    spectra are not sensitive to the specifics of the temperature (Damiano and Hu, 2022)
    > https://arxiv.org/pdf/2204.13816

    Beta Distribution Parameters (a, b):
    ------------------------------------
    - The parameters 'a' and 'b' are set to 2 and 5 respectively to create a distribution 
        that is skewed towards lower values (more common) with a longer tail towards higher 
        values (less common). This helps simulate conditions where typical temperatures and 
        pressures are low with a possibility of reaching higher values occasionally.

    Scaling Formula:
    ----------------
    - The formula used for scaling the Beta distribution output to a specific range is:
      scaled_value = min_value + (max_value - min_value) * beta_output
    - This is a standard approach for feature scaling that adjusts the data into a specified 
        range [min_value, max_value].
    - More on this can be read about feature scaling on Wikipedia: 
        https://en.wikipedia.org/wiki/Feature_scaling

    Source:
    ------
    Fuller-Rowell, T. (2014). Physical Characteristics and Modeling of Earth’s Thermosphere. 
    > https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/9781118704417.ch2
    > https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
    > https://nssdc.gsfc.nasa.gov/planetary/factsheet/venusfact.html
    """
    min_press_earth = 1e-11      # Earth’s upper atmosphere (Fuller-Rowell, 2014)
    a, b = 2, 5                  # Shape parameters for Beta distribution
    min_temp = 210               # Approximate Mars-like minimum atmosphere temperature in Kelvin
    max_temp = 370               # Thermal limits to life on Earth  (Clarke, 2014)
    min_pres = 0.006             # Approximate Mars surface pressure in bars
    max_pres = 10.0              # A practical upper limit for pressures (10x Earth), less than Venus (92 bars TO MUCH!)

    # Generate beta-distributed random values for temperature and pressure
    scaled_temp = min_temp + (max_temp - min_temp) * beta.rvs(a, b)
    scaled_pres = min_pres + (max_pres - min_pres) * beta.rvs(a, b)

    # Generate logarithmically spaced pressures from a minimum value to a scaled value
    pressure = np.logspace(np.log10(scaled_pres), np.log10(min_press_earth), num=layers)
    
    # Return pressure array and a constant temperature array
    return pressure, np.ones(len(pressure)) * scaled_temp

def modern_earth(config: dict) -> None:
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
    csv_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "atmos_layers", "modern_atm.csv"))
    modern = pd.read_csv(csv_path)

    molecules = ["CO2" , "N2" , "O2" , "H2O", "CO", "C2H6" , 
                "HCN", "SO2" , "O3" , "CH4" , "N2O", "NH3"]
    
    # Mapping molecules to HITRAN database indices
    # https://hitran.org/lbl/
    HITRAN_DICT = {"CO2": "HIT[2]", "N2" : "HIT[22]", "O2" : "HIT[7]", 
                    "H2O": "HIT[1]", "CO": "HIT[5]", "C2H6": "HIT[27]",
                    "HCN": "HIT[23]", "SO2": "HIT[9]", "O3":"HIT[3]", 
                    "CH4" :"HIT[6]", "N2O": "HIT[4]", "NH3" : "HIT[11]"}
        
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

def after_goe(config: dict) -> None:
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
    csv_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "atmos_layers", "after_goe.csv"))
    goe = pd.read_csv(csv_path)
    
    molecules = ["H2O", "N2O", "O3", "CO", "CO2", "O2", "CH4", "N2"]
    HITRAN_DICT = {"H2O": "HIT[1]", "N2O": "HIT[4]", "O3":"HIT[3]", "CO": "HIT[5]",
                    "CO2": "HIT[2]", "O2" : "HIT[7]", "CH4" :"HIT[6]", "N2" : "HIT[22]"}
    
    # Unit sum normalization at each layer
    goe["N2"] = np.full(60, 0.781)
    abun = goe.columns[2:]
    goe[abun] = goe[abun].div(goe[abun].sum(axis=1), axis=0)

    # That's right, right? Mean Molecular Weight Calculation
    mmw = goe[abun].apply(lambda row: sum(row[col] * molweight_after_goe()[i] for i, col in enumerate(abun)), axis=1).mean()
        
    for i in range(layers):
        layer_data = [f'{goe["Pressure"][i]}', f'{goe["Temperature"][i]}']
        layer_data += [f'{goe[molecule][i]}' for molecule in molecules]
        config[f'ATMOSPHERE-LAYER-{i + 1}'] = ','.join(layer_data)

    config['ATMOSPHERE-WEIGHT'] = mmw
    config['ATMOSPHERE-NGAS'] = len(molecules)
    config['ATMOSPHERE-GAS'] = ",".join(molecules)
    config['ATMOSPHERE-TYPE'] = ",".join([HITRAN_DICT[mol] for mol in molecules]) 
    config['ATMOSPHERE-ABUN'] = "1," * (len(molecules) - 1) + '1'
    config['ATMOSPHERE-UNIT'] = "scl," * (len(molecules) - 1) + 'scl' 
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(molecules)
    config["ATMOSPHERE-PRESSURE"] = goe["Pressure"][0] * 1000 # in mbar
    config["SURFACE-TEMPERATURE"] = goe["Temperature"][0] # in K

def random_atmosphere(config: dict) -> None:
    """
    Generates a completely random atmospheric composition for planetary modeling and 
    populates the provided configuration dictionary with the results. This function
    autonomously calculates random molecular concentrations, normalizes them, and
    distributes these across specified atmospheric layers without using additional
    functions from `datamod`.

    Parameters
    ----------
    config : dict
        A configuration dictionary where the atmospheric parameters will be stored.

    Returns
    -------
    None

    Notes
    -----
    The atmosphere is assumed to be isothermal across all layers, and molecular
    concentrations are randomly assigned with a special consideration for "known" and
    "foreign" molecules. "Known" molecules have a 40% chance to be zero to simulate
    absence in some scenarios. The function also sets up other necessary atmospheric
    parameters such as average molecular weight and spectral type configurations.
    """
    layers = 60 
    molecular_weights = {
        'H2O': 18.01528,  # Water vapor
        'CO2': 44.0095,   # Carbon dioxide
        'CH4': 16.0425,   # Methane
        'O2': 31.99880,   # Oxygen
        'NH3': 17.03052,  # Ammonia
        'HCN': 27.0253,   # Hydrogen cyanide
        'PH3': 33.99758,  # Phosphine
        'SO2': 64.0638,   # Sulfur dioxide
        'H2S': 34.0809    # Hydrogen sulfide
    }
    HITRAN_DICT = {
        'H2O': 'HIT[1]', 'CO2': 'HIT[2]', 'CH4': 'HIT[6]', 
        'O2': 'HIT[7]', 'NH3': 'HIT[11]', 'HCN': 'HIT[23]',
        'PH3': 'HIT[28]', 'SO2': 'HIT[9]', 'H2S': 'HIT[31]'
    }

    # Generate random molecule concentrations
    known_molecules = ['H2O', 'CO2', 'CH4', 'O2']
    foreign_molecules = ['NH3', 'HCN', 'PH3', 'SO2', 'H2S']

    # Calculate random values for each molecule
    sample = {}
    for molecule in known_molecules:
        sample[molecule] = np.random.rand() / 2 if np.random.random() < 0.25 else 0
    for molecule in foreign_molecules:
        sample[molecule] = np.random.rand()

    # Normalize molecule concentrations
    total_concentration = sum(sample.values())
    normalized_sample = {molecule: value / total_concentration for molecule, value in sample.items()}

    # Replicate concentrations across all layers
    layer_concentrations = {molecule: np.full(layers, concentration) for molecule, concentration in normalized_sample.items()}

    # Atmospheric parameters setup
    pressure, temperature = isothermal_PT(layers)

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
    config["ATMOSPHERE-PRESSURE"] = pressure[0] * 1000 # in mbar

    dm.set_spectral_type(config)
    dm.set_stellar_parameters(config)
    dm.set_solar_coordinates(config)
    dm.set_habitable_zone_distance(config)
    dm.maintain_planetary_atmosphere(config)



