import numpy as np
import astropy.units as u
from astropy.constants import R_sun, L_sun, sigma_sb, G, M_earth, R_earth

def mixing_ratio_constant(config: dict, layers: int) -> None:
    """
    Modifies the input configuration dictionary to set a constant 
    abundance value across atmospheric layers.

    Parameters
    ----------
    config : dict
        A dictionary containing configuration data for atmospheric layers.
        Each key is a string in the format 'ATMOSPHERE-LAYER-{index}' and the corresponding 
        value is a string containing comma-separated data where the abundance data starts 
        from the third element.
    layers : int
        The number of atmospheric layers to process.

    Notes
    -----
    This function directly modifies the 'config' dictionary to update the abundance values
    for each layer based on the first layer's abundance values. This approach assumes all layers
    will have the same abundance values as the first layer.
    This function DOES NOT MODIFY the average molecular weight.
    """
    # Preallocate a numpy array for performance and memory efficiency
    abundances = np.empty((layers, len(config["ATMOSPHERE-LAYER-1"].split(",")[2:])))

    # Fill the numpy array directly from the config
    for i in range(layers):
        layer_abundances = config[f"ATMOSPHERE-LAYER-{i + 1}"].split(",")[2:]
        abundances[i, :] = np.array(layer_abundances, dtype=float)

    # Sums up the abundances across all layers and normalizes
    total_abundances = np.sum(abundances, axis=0)
    normalized_abundances = total_abundances / np.sum(total_abundances)

    # Update Layer Information with vectorized string operations
    for i in range(layers):
        PT = config[f"ATMOSPHERE-LAYER-{i + 1}"].split(",")[:2]
        abundance_values = ",".join([f"{value}" for value in normalized_abundances])
        config[f"ATMOSPHERE-LAYER-{i + 1}"] = ",".join(PT + [abundance_values])

def random_atmospheric_layers(config: dict, layers: int) -> None:
    """
    Modify the atmospheric layers configuration by multiplying each column (starting 
    from the third element in each entry) with a random factor. Each column has a 
    25% chance to get a random factor of zero, otherwise a random factor between 0 
    and 2.

    Parameters
    ----------
    config : dict
        A dictionary where keys are strings formatted as 'ATMOSPHERE-LAYER-{i + 1}' 
        and values are strings of comma-separated numbers. The first two numbers of 
        each value are not modified, while the subsequent numbers are multiplied by 
        a random factor.

    Returns
    -------
    None
        The function modifies the dictionary in-place, adding modified values as 
        strings concatenated to the initial unmodified parts.
    """

    # Generate random factors for each type of gas with 25% chance of being zero
    num_gases = len(config[f"ATMOSPHERE-LAYER-1"].split(",")) - 2
    random_factors = [0 if np.random.random() < 0.25 else np.random.uniform(0, 2) for _ in range(num_gases)]

    # Iterate over each key in the dictionary and modify the values accordingly
    for i in range(layers):
        PT = config[f"ATMOSPHERE-LAYER-{i + 1}"].split(",")[:2]
        original_values = config[f"ATMOSPHERE-LAYER-{i + 1}"].split(",")[2:]
        modified_values = [str(float(value) * random_factors[index]) for index, value in enumerate(original_values)]
        config[f"ATMOSPHERE-LAYER-{i + 1}"] = ",".join(PT + modified_values)

def normalize_layer(config: dict, layers: int, molweight: list) -> None:
    """
    Normalize the values from the third element onward of the first atmospheric 
    layer based on their sum and apply this normalization to all other layers. 
    Each layer's values are replaced by the normalized values of the first layer.

    Parameters
    ----------
    config : dict
        A dictionary where keys are strings formatted as 'ATMOSPHERE-LAYER-{i + 1}' 
        and values are strings of comma-separated numbers. The first two numbers of 
        each value are not modified.
    molweight : list
        A list of molecular weights for the molecules in the order specified by 
        `config["ATMOSPHERE-LAYERS-MOLECULES"]`.
        
    Returns
    -------
    None
        The function modifies the dictionary in-place, updating each layer with the 
        normalized values from the first layer and setting a new key 'ATMOSPHERE-WEIGHT' 
        with the computed average molecular weight.
    """
    # Retrieve the abundance data from the configuration for the first layer and 
    # convert to float
    first_layer_abundances = np.array(config['ATMOSPHERE-LAYER-1'].split(',')[2:], dtype=float)
    
    # Calculate the normalized values based on their sum
    normalized_values = first_layer_abundances / np.sum(first_layer_abundances)

    # Apply the normalized values to all layers
    for i in range(layers):
        key = f"ATMOSPHERE-LAYER-{i + 1}"
        values = config[key].split(',')
        PT = values[:2]  # Preserve the first two values
        # Update the dictionary with the normalized values
        config[key] = ','.join(PT + normalized_values.astype(str).tolist())

    # Compute the weighted average molecular weight and store it in the configuration
    config["ATMOSPHERE-WEIGHT"] = np.sum(molweight * normalized_values)

def set_spectral_type(config: dict) -> None:
    """
    Sets the spectral type of the star and updates the dictionary with star and 
    occultation class.
    """
    spectral_type = ['F', 'G', 'K', 'M']
    class_star = np.random.choice(spectral_type)
    config['OBJECT-STAR-TYPE'] = class_star
    config['GEOMETRY-STELLAR-TYPE'] = class_star

def set_stellar_parameters(config: dict) -> None:
    """
    Sets key stellar parameters such as radius, temperature, and magnitude 
    based on the star's spectral type. The function updates the 'config' 
    dictionary with randomized values for these parameters, within the ranges 
    associated with the spectral type provided. 
    """
    class_star = config['OBJECT-STAR-TYPE']
    params = {
        'F': {'temp_range': (6000, 7220), 'radius_range': (1.18, 1.79), 'mag_range': (2.50, 4.22)},
        'G': {'temp_range': (5340, 5920), 'radius_range': (0.876, 1.12), 'mag_range': (4.40, 5.34)},
        'K': {'temp_range': (3940, 5280), 'radius_range': (0.552, 0.817), 'mag_range': (5.54, 7.59)},
        'M': {'temp_range': (2320, 3870), 'radius_range': (0.104, 0.559), 'mag_range': (7.75,  13.62)}
    }

    star_temperature = round(np.random.uniform(*params[class_star]['temp_range']), 3)
    star_radius = round(np.random.uniform(*params[class_star]['radius_range']), 3)
    star_mag = round(np.random.uniform(*params[class_star]['mag_range']), 3)

    config['OBJECT-STAR-RADIUS'] = star_radius
    config['OBJECT-STAR-TEMPERATURE'] = star_temperature
    config['GEOMETRY-STELLAR-TEMPERATURE'] = star_temperature
    config["GEOMETRY-STELLAR-MAGNITUDE"] = star_mag

    # Set the planet's distance from the observer to 3 parsecs (fixed value)
    config["GEOMETRY-OBS-ALTITUDE"] = 3

    # (REMOVED) Motivation and source: High metallicity and non-equilibrium chemistry... 
    # (Madhusudhan1 and Seager 2011)
    # https://iopscience.iop.org/article/10.1088/0004-637X/729/1/41/meta
    # 10x greater and lesser the metallicity of the sun (in dex)
    #config["OBJECT-STAR-METALLICITY"] = round(np.random.uniform(-1, 1), 3)

def set_solar_coordinates(config: dict) -> None:
    """
    Randomly sets the sub-solar longitude and latitude.
    """
    config['OBJECT-SOLAR-LONGITUDE'] = np.random.uniform(-360, 360)
    config['OBJECT-SOLAR-LATITUDE'] =  np.random.uniform(-90, 90)

def calculate_luminosity(config: dict) -> float:
    """
    Calculate luminosity using the Stefan-Boltzmann Law to calculate luminosity.
    """
    star_radius = config['OBJECT-STAR-RADIUS']
    temperature = config['OBJECT-STAR-TEMPERATURE']
    return 4 * np.pi * (star_radius * R_sun.value) ** 2 * sigma_sb.value * temperature ** 4

def set_habitable_zone_distance(config: dict) -> None:
    """
    Calculates and sets the habitable zone distance based on the star's 
    luminosity and temperature.

    Notes
    -----
    Source: Habitable zones around main-sequence stars... 
    ([Kopparapu et al. (2014)](https://iopscience.iop.org/article/10.1088/2041-8205/787/2/L29/pdf)). 
    See Equation 4 and Table 1 from Kopparapu et al. (2013)
    """
    temp = config['OBJECT-STAR-TEMPERATURE'] - 5780
    luminosity_star = calculate_luminosity(config)

    # Recent Venus (lower limit)
    S_eff_odot = 1.776
    a, b, c, d = 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15
    S_eff_lower = S_eff_odot + a * temp + b * temp**2 + c * temp**3 + d * temp**4

    # Early Mars (upper limit)
    S_eff_odot = 0.32
    a, b, c, d = 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16
    S_eff_upper = S_eff_odot + a * temp + b * temp**2 + c * temp**3 + d * temp**4

    # Distance of the habitable zone
    lower_dist = np.sqrt((luminosity_star / L_sun.value) / S_eff_lower)
    upper_dist = np.sqrt((luminosity_star / L_sun.value) / S_eff_upper)
    config['OBJECT-STAR-DISTANCE'] = np.random.uniform(lower_dist, upper_dist)

def maintain_planetary_atmosphere(config: dict, attempts: int = 5) -> None:
    """
    Simulates to find a planet size that can maintain an atmosphere. 
    References are included for each scientific principle used.

    The number of attempts to try generating a planet with atmosphere. 
    Recommended between 3 to 5 attempts for balancing efficiency and success rate.
    More attempts may slow execution without significantly improving success.
    """
    if attempts == 0:
        raise ValueError("Failed to find a suitable planet configuration to maintain an atmosphere.")

    semi_major_axis = config['OBJECT-STAR-DISTANCE']
    star_luminosity = calculate_luminosity(config)

    # Define the planet's radius in Earth radii; targeting terrestrial planets 
    # up to about 1.23 Earth radii. Based on findings from "A rocky exoplanet classification method 
    # and its application to calculating surface pressure and surface temperature"
    # Reference: https://arxiv.org/abs/2301.03348
    planet_radius = np.random.uniform(0.3, 1.23)

    # # Calculate planet mass based on planet radius using the relationship
    # derived from Equation (2) of the paper "A rocky exoplanet classification method 
    # and its application to calculating surface pressure and surface temperature" 
    # by McIntyre et al. (2023)
    planet_mass = planet_radius ** (1 / (0.279 + np.random.uniform(-0.009, 0.009)))
        
    # Calculate planetary gravity (g = GM/r²) in m/s² using the mass and radius 
    # estimates.
    gravity = G.value * (planet_mass * M_earth.value) / (planet_radius * R_earth.value) ** 2

    # Calculate escape velocity from the planet's surface in m/s
    escape_velocity = np.sqrt(2 * gravity * planet_radius * R_earth.value)

    # Compute the insolation, considering the star-planet 
    # distance and stellar luminosity. Reference for insolation calculations: 
    # Zahnle and Catling (2017), particularly their Equation (4)
    # https://arxiv.org/pdf/1702.03386
    real_insolation = (1 ** 2 / semi_major_axis ** 2) * (star_luminosity / L_sun.value)

    # Estimate the critical insolation for atmospheric retention based on escape 
    # velocity, using an empirically derived relationship from Zahnle and 
    # Catling (2017), with approximation from graph analysis.
    # See equation (7) by McIntyre et al. (2023)
    cosmic_shoreline = (5 * 1e-16) * (escape_velocity) ** 4

    # Check if the real insolation is less than the calculated critical insolation for the planet
    # This condition ensures that the planet is below the Cosmic Shoreline.
    # If the planet receives less energy (insolation) than the critical threshold defined by 
    # the Cosmic Shoreline, it means the planet has enough gravitational pull to retain its atmosphere.
    # In this case, we proceed to store the planet's properties in the config dictionary.
    if real_insolation < cosmic_shoreline:
        # According to McIntyre et al. (2023), Equation (3) in mbar
        config["ATMOSPHERE-PRESSURE"] = 1013.25 * (planet_radius ** (3.168 + np.random.uniform(-0.232, 0.232)))
        config['OBJECT-DIAMETER'] = 2 * planet_radius * R_earth.to(u.km).value
        config['OBJECT-GRAVITY'] = gravity

        # See equation (25) by McIntyre et al. (2023) for surface temperature
        temperature_analogue = 41.9 * cosmic_shoreline ** (1 / 4) + 33.85
        config["ATMOSPHERE-TEMPERATURE"] = temperature_analogue
        config["SURFACE-TEMPERATURE"] = temperature_analogue
        config["SURFACE-ALBEDO"] = np.random.uniform(0.1, 0.8)
        config["SURFACE-EMISSIVITY"] = 1 - config["SURFACE-ALBEDO"]
    else:
        # If the real insolation is greater than or equal to the critical insolation,
        # the planet is above the Cosmic Shoreline and cannot retain a significant atmosphere.
        # Therefore, we call the function again recursively to generate a new planet configuration
        # that might meet the criteria for atmosphere retention.
        # Recursively call the function, reducing the number of remaining attempts
        maintain_planetary_atmosphere(config, attempts - 1)

def set_instrument(config: dict, instrument: str) -> None:
    """
    Adjust the telescope settings in the provided dictionary based 
    on the selected instrument. If not used, the default setting will be 'SS-Vis'.

    Parameters
    ----------
    config : dict
        The dictionary of settings to be modified.
    instrument : str
        The telescope instrument for which settings need to be modified. 
        Valid options are 'HWC', 'SS-NIR', 'SS-UV', and 'SS-Vis'.

    Notes
    -----
    The function updates the settings dictionary directly based on the instrument 
    specification, applying predefined configurations for optical and infrared 
    channels, including noise characteristics and other operational parameters.
    """
    if instrument == "SS-Vis":
        pass

    valid_instruments = ['HWC', 'SS-NIR', 'SS-UV', 'SS-Vis']
    if instrument not in valid_instruments:
        raise ValueError(f"Instrument must be one of {valid_instruments}.")
    
    if instrument == "HWC":
        config['GENERATOR-INSTRUMENT'] = """
        HabEx_HWC-Spec: The HabEx Workforce Camera (HWC) has two channels that can simultaneously observe the same field of view: 
        an optical channel using delta-doped CCD detectors providing access from 370 nm to 950 nm (QE:0.9), and a near-IR channel 
        using Hawaii-4RG HgCdTe (QE:0.9) arrays providing good throughput from 950 µm to 1.8 µm. The imaging mode can provide 
        spectroscopy (RP<10) via filters at high-throughput, while a grating delivering RP=1000 is assumed to reduce the throughput 
        by 50%.
        """
        config['GENERATOR-RANGE1'] = 0.2    # From UV! (Just a modification...)
        config['GENERATOR-RANGE2'] = 2.00   # Another modification!
        config['GENERATOR-RESOLUTION'] = 1000
        config['GENERATOR-TELESCOPE'] = "SINGLE"
        config['GENERATOR-TELESCOPE1'] = 1
        config['GENERATOR-TELESCOPE2'] = 2.0
        config['GENERATOR-TELESCOPE3'] = 1.0
        config['GENERATOR-NOISEOTEMP'] = 250
        config['GENERATOR-NOISEOEFF'] = '0.000@0.325,0.003@0.337,0.016@0.348,0.067@0.353,0.183@0.365,0.222@0.370,0.240@0.381,\
            0.251@0.401,0.273@0.421,0.302@0.454,0.312@0.508,0.302@0.620,0.283@0.714,0.258@0.793,0.248@0.836,0.261@0.905,0.280@0.955, \
                0.287@1.004,0.295@1.131,0.302@1.291,0.314@1.426,0.321@1.561,0.330@1.693,0.335@1.800'
        config['GENERATOR-NOISEFRAMES'] = 10
        config['GENERATOR-NOISETIME'] = 3600  
        config['GENERATOR-NOISEPIXELS'] = 8
        config['GENERATOR-CONT-STELLAR'] = 'Y'
    
    elif instrument == "SS-NIR":
        config['GENERATOR-INSTRUMENT'] = """
        HabEx_SS-NIR: The HabEx StarShade (SS) will provide extraordinary high-contrast capabilities from the UV (0.2 to 0.45 um), to the 
        visible (0.45 to 1um), and to the infrared (0.975 to 1.8 um). By limiting the number of optical surfaces, this configuration provides 
        high optical throughput (0.2 to 0.4) across this broad of wavelengths, while the quantum efficiency (QE) is expected to be 0.9 for the 
        VU and visible detectors and 0.6 for the infrared detector. The UV channel provides a resolution (RP) of 7, visible channel a maximum 
        of 140 and the infrared 40.
        """
        config['GENERATOR-RANGE1'] = 0.975
        config['GENERATOR-RANGE2'] = 1.80
        config['GENERATOR-RESOLUTION'] = 40
        config['GENERATOR-TELESCOPE3'] = '7e-11@-0.000e+00,7e-11@-1.995e-02,7e-11@-3.830e-02,3.544e-03@-5.439e-02,1.949e-02@-6.791e-02,\
            3.367e-02@-7.434e-02,6.734e-02@-7.982e-02,1.241e-01@-8.561e-02,2.091e-01@-9.108e-02,2.818e-01@-9.526e-02,3.332e-01@-9.752e-02,\
                3.987e-01@-1.014e-01,4.661e-01@-1.052e-01,5.352e-01@-1.075e-01,6.008e-01@-1.110e-01,6.344e-01@-1.130e-01,6.699e-01@-1.155e-01,\
                    6.911e-01@-1.184e-01,7.000e-01@-1.278e-01,7.000e-01@-1.561e-01,7.000e-01@-1.950e-01,7.000e-01@-2.224e-01,7.000e-01@-2.349e-01'
        config['GENERATOR-NOISEFRAMES'] = 1
        config['GENERATOR-NOISETIME'] = 1000

    elif instrument == "SS-UV":
        config['GENERATOR-INSTRUMENT'] = """
        HabEx_SS-UV: The HabEx StarShade (SS) will provide extraordinary high-contrast capabilities from the UV (0.2 to 0.45 um), to the visible 
        (0.45 to 1um), and to the infrared (0.975 to 1.8 um). By limiting the number of optical surfaces, this configuration provides high optical 
        throughput (0.2 to 0.4) across this broad of wavelengths, while the quantum efficiency (QE) is expected to be 0.9 for the VU and visible 
        detectors and 0.6 for the infrared detector. The UV channel provides a resolution (RP) of 7, visible channel a maximum of 140 and 
        the infrared 40.
        """
        config['GENERATOR-RANGE1'] = 0.2
        config['GENERATOR-RANGE2'] = 0.45
        config['GENERATOR-RESOLUTION'] = 7
        config['GENERATOR-TELESCOPE3'] = '7e-11@-0.000e+00,7e-11@-7.483e-03,7e-11@-1.436e-02,3.544e-03@-2.040e-02,1.949e-02@-2.547e-02,\
            3.367e-02@-2.788e-02,6.734e-02@-2.993e-02,1.241e-01@-3.210e-02,2.091e-01@-3.416e-02,2.818e-01@-3.572e-02,3.332e-01@-3.657e-02,\
                3.987e-01@-3.802e-02,4.661e-01@-3.947e-02,5.352e-01@-4.031e-02,6.008e-01@-4.164e-02,6.344e-01@-4.236e-02,6.699e-01@-4.333e-02,\
                    6.911e-01@-4.441e-02,7.000e-01@-4.791e-02,7.000e-01@-5.853e-02,7.000e-01@-7.314e-02,7.000e-01@-8.340e-02,7.000e-01@-8.810e-02'
        config['GENERATOR-NOISEFRAMES'] = 1
        config['GENERATOR-NOISETIME'] = 1000

def random_planet(config: dict, molweight: list, layers: int = 60) -> None:
    """
    Configures a random planet by setting various parameters and calculating 
    necessary environmental and physical attributes. This function orchestrates 
    the setup of atmospheric conditions, stellar characteristics, and planetary 
    habitability factors.

    Parameters
    ----------
    config : dict
        A dictionary where all the planetary and stellar configuration settings are stored.
    molweight : list
        A list of molecular weights used to normalize atmospheric layers.
    layers : int, optional
        The number of atmospheric layers to generate and configure, default is 60.

    Steps:
    1. Set a constant mixing ratio for the atmosphere.
    2. Generate random atmospheric layers based on the defined mixing ratio.
    3. Normalize the atmospheric layer values to ensure consistency.
    4. Determine and set the spectral type of the star associated with the planet.
    5. Calculate and set the stellar parameters including radius and temperature 
        based on the spectral type.
    6. Set random solar coordinates for the planet relative to its star.
    7. Calculate the distance of the habitable zone based on the stellar parameters.
    8. Simulate planetary characteristics to ensure the planet can maintain an 
        atmosphere.

    """
    mixing_ratio_constant(config, layers)
    random_atmospheric_layers(config, layers)
    normalize_layer(config, layers, molweight)
    set_spectral_type(config)
    set_stellar_parameters(config)
    set_solar_coordinates(config)
    set_habitable_zone_distance(config)
    maintain_planetary_atmosphere(config)

