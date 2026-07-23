import numpy as np
import astropy.units as u
from astropy.constants import R_sun, L_sun, sigma_sb, G, M_earth, R_earth, R

# Frozen subset of the Pecaut-Mamajek mean dwarf sequence (version
# 2022.04.16).  Values are ordered by increasing effective temperature.
# Columns: spectral subtype, Teff [K], radius [R_sun].
# https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.dat
_MAIN_SEQUENCE = {
    "F": {
        "spt": np.array(
            [
                "F9.5V",
                "F9V",
                "F8V",
                "F7V",
                "F6V",
                "F5V",
                "F4V",
                "F3V",
                "F2V",
                "F1V",
                "F0V",
            ]
        ),
        "teff": np.array(
            [5990, 6050, 6180, 6280, 6350, 6550, 6670, 6750, 6820, 7020, 7220],
            dtype=float,
        ),
        "radius": np.array(
            [
                1.142,
                1.167,
                1.221,
                1.324,
                1.359,
                1.473,
                1.533,
                1.578,
                1.622,
                1.679,
                1.728,
            ],
            dtype=float,
        ),
    },
    "G": {
        "spt": np.array(
            ["G9V", "G8V", "G7V", "G6V", "G5V", "G4V", "G3V", "G2V", "G1V", "G0V"]
        ),
        "teff": np.array(
            [5380, 5480, 5550, 5600, 5660, 5680, 5720, 5770, 5860, 5930], dtype=float
        ),
        "radius": np.array(
            [0.853, 0.914, 0.927, 0.949, 0.977, 0.991, 1.002, 1.012, 1.060, 1.100],
            dtype=float,
        ),
    },
    # "K": {
    #     "spt": np.array(
    #         ["K9V", "K8V", "K7V", "K6V", "K5V", "K4V", "K3V", "K2V", "K1V", "K0V"]
    #     ),
    #     "teff": np.array(
    #         [3930, 3990, 4100, 4300, 4440, 4600, 4830, 5100, 5170, 5270], dtype=float
    #     ),
    #     "radius": np.array(
    #         [0.608, 0.615, 0.630, 0.669, 0.701, 0.713, 0.755, 0.783, 0.797, 0.813],
    #         dtype=float,
    #     ),
    # },
}


def _layer_values(config: dict, index: int) -> list:
    """Return one PSG atmospheric layer split into its comma-separated fields."""
    key = f"ATMOSPHERE-LAYER-{index + 1}"
    if key not in config:
        raise KeyError(f"Missing atmospheric layer: {key}")
    values = config[key].split(",")
    if len(values) < 3:
        raise ValueError(
            f"{key} must contain pressure, temperature, and at least one gas."
        )
    return values


def mixing_ratio_constant(config: dict, layers: int) -> None:
    """
     For each species, the function calculates a representative volume
    mixing ratio by integrating its vertical profile over pressure using
    the trapezoidal rule,

        <Y_i> = integral(Y_i dp) / integral(dp),

    which approximates an atmospheric-column-weighted mean under
    hydrostatic equilibrium. The resulting abundances are normalized so
    that their sum is unity and are then assigned uniformly to every
    atmospheric layer.

    Parameters
    ----------
    config : dict
        Atmospheric configuration modified in place. Each layer must be
        stored under a key of the form ``ATMOSPHERE-LAYER-{index}``, with
        a comma-separated value containing pressure, temperature, and the
        species volume mixing ratios, in that order.
    layers : int
        Number of atmospheric layers to process. At least two layers with
        distinct, positive pressures are required.

    Notes
    -----
    The pressure and temperature values of the individual layers are
    preserved. Only the chemical abundances are replaced.

    Pressure may be provided in any consistent unit because it cancels
    when computing the weighted mean. The calculation assumes that the
    listed pressures represent layer-centre pressures.

    This function does not update ``ATMOSPHERE-WEIGHT`` or otherwise
    recalculate the mean molecular weight.

    Replace each vertical abundance profile with its pressure-weighted
    mean and use N2 as the balance gas so that the composition sums to one.
    """
    first = _layer_values(config, 0)
    num_gases = len(first) - 2

    molecules = [
        molecule.strip()
        for molecule in config["ATMOSPHERE-LAYERS-MOLECULES"].split(",")
    ]

    if len(molecules) != num_gases:
        raise ValueError("The number of molecule names does not match the gas columns.")

    if "N2" not in molecules:
        raise ValueError("N2 must be present as the atmospheric balance gas.")

    pressures = np.empty(layers)
    abundances = np.empty((layers, num_gases))

    for i in range(layers):
        values = _layer_values(config, i)
        pressures[i] = float(values[0])
        abundances[i] = np.asarray(values[2:], dtype=float)

    order = np.argsort(pressures)
    pressure = pressures[order]
    abundance = abundances[order]

    pressure_span = pressure[-1] - pressure[0]

    if pressure_span <= 0.0:
        raise ValueError("Atmospheric pressure grid must contain distinct values.")

    dp = np.diff(pressure)

    column_mean = (
        np.sum(
            0.5 * (abundance[1:] + abundance[:-1]) * dp[:, None],
            axis=0,
        )
        / pressure_span
    )

    n2_index = molecules.index("N2")
    non_n2_indices = [i for i in range(num_gases) if i != n2_index]

    non_n2_total = float(np.sum(column_mean[non_n2_indices]))

    if not 0.0 <= non_n2_total < 1.0:
        raise ValueError(
            "The pressure-weighted non-N2 abundances must sum to less than one."
        )

    # N2 fills the compositional remainder.
    column_mean[n2_index] = 1.0 - non_n2_total

    composition = column_mean.astype(str).tolist()

    for i in range(layers):
        values = _layer_values(config, i)

        config[f"ATMOSPHERE-LAYER-{i + 1}"] = ",".join(values[:2] + composition)


def random_atmospheric_layers(
    config: dict,
    layers: int,
    log_ratio_half_width: float = 2.5,
    minimum_n2_fraction: float = 0.5,
    max_attempts: int = 10_000,
) -> None:
    """
    Perturb atmospheric composition symmetrically in log-ratio space.

    For every positive species i other than N2, the template abundance
    ratio relative to N2 is perturbed according to

        log(Y_i / Y_N2)_new
            = log(Y_i / Y_N2)_template + U(-a, a),

    where ``a`` is ``log_ratio_half_width``.

    The resulting ratios are transformed back into mixing ratios using

        Y_N2 = 1 / (1 + sum_i r_i)

    and

        Y_i = r_i * Y_N2,

    where ``r_i`` is the perturbed ratio Y_i/Y_N2.

    This transformation guarantees positive abundances and exact
    compositional closure. Species with zero abundance in the geological
    template remain exactly zero.

    A geological prior is imposed on N2: its final abundance cannot fall
    below ``minimum_n2_fraction`` times its template abundance.

    Parameters
    ----------
    config : dict
        PSG atmospheric configuration modified in place. Each layer must
        contain pressure, temperature, and gas mixing ratios, in that
        order.
    layers : int
        Number of atmospheric layers.
    log_ratio_half_width : float, optional
        Half-width of the uniform perturbation in natural-log ratio
        space. The default is 2.5, corresponding to ratio multipliers
        between exp(-2.5) approximately 0.082 and
        exp(2.5) approximately 12.18.
    minimum_n2_fraction : float, optional
        Minimum allowed N2 abundance as a fraction of the template N2
        abundance. The default of 0.5 prevents N2 from falling below
        half of its template value.
    max_attempts : int, optional
        Maximum number of compositional proposals before raising an
        error. Rejection occurs only when the N2 geological prior is
        violated.

    Notes
    -----
    The same final composition is applied to every atmospheric layer,
    preserving the isoabundance assumption. Pressure and temperature
    are not modified.
    """
    # Gas names must follow the same order as the abundance columns
    molecules = [
        molecule.strip()
        for molecule in config["ATMOSPHERE-LAYERS-MOLECULES"].split(",")
    ]

    first_layer = _layer_values(config, 0)
    template = np.asarray(first_layer[2:], dtype=float)

    closure = float(np.sum(template))
    template = template / closure

    molecule_names_upper = [molecule.upper() for molecule in molecules]

    n2_index = molecule_names_upper.index("N2")
    template_n2 = float(template[n2_index])

    minimum_n2 = minimum_n2_fraction * template_n2

    # Select only positive non-N2 gases. Gases that are exactly zero remain zero.
    positive_non_n2_indices = np.asarray(
        [
            index
            for index, abundance in enumerate(template)
            if index != n2_index and abundance > 0.0
        ],
        dtype=int,
    )

    if positive_non_n2_indices.size == 0:
        final_composition = np.zeros_like(template)
        final_composition[n2_index] = 1.0

    else:
        # Template gas/N2 ratios
        template_ratios = template[positive_non_n2_indices] / template_n2

        for _ in range(max_attempts):
            # Symmetric perturbation in natural-log space
            log_ratio_offsets = np.random.uniform(
                low=-log_ratio_half_width,
                high=log_ratio_half_width,
                size=positive_non_n2_indices.size,
            )

            perturbed_ratios = template_ratios * np.exp(log_ratio_offsets)

            # Transform the ratios back into a closed composition
            candidate_n2 = float(1.0 / (1.0 + np.sum(perturbed_ratios)))

            # Apply the geological N2 prior
            if candidate_n2 < minimum_n2:
                continue

            final_composition = np.zeros_like(template)

            final_composition[n2_index] = candidate_n2

            final_composition[positive_non_n2_indices] = perturbed_ratios * candidate_n2

            break

        else:
            raise ValueError(
                "Could not generate a composition satisfying "
                f"Y_N2 >= {minimum_n2:.6g} after "
                f"{max_attempts} attempts."
            )

    # Correct only possible floating-point closure error
    final_composition /= np.sum(final_composition)

    composition_strings = final_composition.astype(str).tolist()

    # Preserve P and T and apply the same composition to all layers
    for layer_index in range(layers):
        values = _layer_values(config, layer_index)
        config[f"ATMOSPHERE-LAYER-{layer_index + 1}"] = ",".join(
            values[:2] + composition_strings
        )


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
    abundances = np.asarray(_layer_values(config, 0)[2:], dtype=float)
    molecular_weights = np.asarray(molweight, dtype=float)

    closure = float(np.sum(abundances))
    normalized = abundances / closure

    composition = normalized.astype(str).tolist()
    for i in range(layers):
        values = _layer_values(config, i)
        config[f"ATMOSPHERE-LAYER-{i + 1}"] = ",".join(values[:2] + composition)

    config["ATMOSPHERE-WEIGHT"] = float(np.dot(molecular_weights, normalized))


def set_spectral_type(config: dict) -> None:
    """
    Sets the spectral type of the star and updates the dictionary with star and
    occultation class.

    [REVISION 5 -- drop K-type hosts]
    ``spectral_types`` defaults to ``SPECTRAL_TYPES`` (see the module header),
    which excludes K.  K-type hosts do not survive the SNR > 3 cut -- the 3000
    spectrum test run retained roughly 235 F and 70 G and no K at all, which is
    also what Section 5 of the manuscript reports -- so every K draw is a full
    set of PSG calls thrown away after the fact.
    """
    # spectral_type = ['F', 'G', 'K', 'M']
    # spectral_type = ["F", "G", "K"]
    spectral_type = ["F", "G"]
    class_star = np.random.choice(spectral_type)
    config["OBJECT-STAR-TYPE"] = class_star
    # config["GEOMETRY-STELLAR-TYPE"] = class_star


def set_stellar_parameters(config: dict, max_inclination_deg: float = 50.0) -> None:
    """
    Sample joint main-sequence stellar and observing parameters.

    Effective temperature is uniform within the selected F, G, or K class.
    Radius is interpolated conditionally along the frozen Pecaut--Mamajek main-sequence locus.

    [REVISION 5 -- truncated distance prior]
    ``distance_pc_range`` defaults to ``DISTANCE_PC_RANGE`` (see the module
    header).  The previous version hard-coded 5-20 pc, but nothing beyond about
    16 pc survives the SNR > 3 cut because the planet flux falls as 1/d^2, so
    those draws are full sets of PSG calls discarded after the fact.

    [REVISION 4 -- record the draw]
    Returns the sampled quantities, including the nearest tabulated spectral
    subtype, which the previous version computed and then discarded.
    """
    star_class = config.get("OBJECT-STAR-TYPE")

    sequence = _MAIN_SEQUENCE[star_class]
    teff_min = float(sequence["teff"][0])
    teff_max = min(float(sequence["teff"][-1]), 7200.0)
    teff = float(np.random.uniform(teff_min, teff_max))

    radius = float(
        np.exp(np.interp(teff, sequence["teff"], np.log(sequence["radius"])))
    )

    distance_pc = float(np.random.uniform(5.0, 16.0))

    if not 0.0 < max_inclination_deg <= 90.0:
        raise ValueError("max_inclination_deg must lie in (0, 90].")

    cos_i = np.random.uniform(np.cos(np.radians(max_inclination_deg)), 1.0)
    inclination = float(np.degrees(np.arccos(cos_i)))
    season = float(np.random.uniform(0.0, 360.0))

    config["OBJECT-STAR-RADIUS"] = radius
    config["OBJECT-STAR-TEMPERATURE"] = teff
    config["GEOMETRY-OBS-ALTITUDE"] = distance_pc
    config["GEOMETRY-ALTITUDE-UNIT"] = "pc"
    config["OBJECT-INCLINATION"] = inclination
    config["OBJECT-SEASON"] = season

    config.pop("GEOMETRY-STELLAR-TYPE", None)
    config.pop("GEOMETRY-STELLAR-TEMPERATURE", None)
    config.pop("GEOMETRY-STELLAR-MAGNITUDE", None)

    # (REMOVED) Motivation and source: High metallicity and non-equilibrium chemistry...
    # (Madhusudhan1 and Seager 2011)
    # https://iopscience.iop.org/article/10.1088/0004-637X/729/1/41/meta
    # 10x greater and lesser the metallicity of the sun (in dex)
    # config["OBJECT-STAR-METALLICITY"] = round(np.random.uniform(-1, 1), 3)


def set_solar_coordinates(config: dict, obliquity_deg: float = 0.0) -> None:
    """
    Set the sub-solar point and record the resulting phase angle.

    [REVISION 1 -- geometry]
    PSG does not treat the sub-solar latitude, the inclination and the season as
    three independent parameters.  For exoplanets it derives the sub-observer
    point from the sub-solar point (PSG documentation, "Defining the object",
    https://psg.gsfc.nasa.gov/help.php#object):

        olat = slat - inclination + 90.0        olon = slon - season

    The previous version drew the sub-solar latitude uniformly in sine over the
    whole sphere while ``set_stellar_parameters`` caps the inclination at 50 deg.
    ``olat`` then stays inside PSG's legal [-90, +90] range only when
    ``slat <= inclination``, which fails for 23.3% of realizations and reaches
    175 deg in the worst case.

    An explicit obliquity convention is used instead, and the phase is carried by
    OBJECT-SEASON alone.  With ``obliquity_deg = 0`` the phase angle is exactly

        alpha = arccos( sin(i) * cos(season) )

    which for i <= 50 deg spans 40-140 deg with a median of 90 deg.  That is the
    quantity Referee 2 asks about in R2.6, and it is returned here so it can be
    stored per target.

    The sub-solar longitude is still randomized.  It is inert for the spatially
    uniform Lambertian surface used here (SURFACE-MODEL = Lambert, NSURF = 0),
    since it only rotates the planet's own longitude system, but it is kept for
    continuity and recorded.

    Parameters
    ----------
    config : dict
        Configuration dictionary. OBJECT-INCLINATION and OBJECT-SEASON must
        already be set, i.e. ``set_stellar_parameters`` must run first.
    obliquity_deg : float, optional
        Half-width of the sub-solar latitude interval, in degrees. The draw is
        additionally clipped to the inclination so that ``olat`` cannot leave
        PSG's legal range. Default 0.0 (zero-obliquity convention).

    Returns
    -------
    None
        Updates the config dictionary with the sub-solar and sub-observer coordinates and the phase angle.
    """
    inclination = float(config["OBJECT-INCLINATION"])
    season = float(config["OBJECT-SEASON"])

    if obliquity_deg < 0.0:
        raise ValueError("obliquity_deg must be non-negative.")

    limit = min(float(obliquity_deg), inclination)
    sub_solar_latitude = float(np.random.uniform(-limit, limit)) if limit > 0.0 else 0.0
    sub_solar_longitude = float(np.random.uniform(0.0, 360.0))

    config["OBJECT-SOLAR-LONGITUDE"] = sub_solar_longitude
    config["OBJECT-SOLAR-LATITUDE"] = sub_solar_latitude

    # PSG will derive these; computed here to validate and to record the phase.
    sub_observer_latitude = sub_solar_latitude - inclination + 90.0
    sub_observer_longitude = sub_solar_longitude - season

    if not -90.0 <= sub_observer_latitude <= 90.0:
        raise ValueError(
            f"Derived sub-observer latitude {sub_observer_latitude:.3f} deg lies "
            f"outside PSG's legal range; slat={sub_solar_latitude:.3f}, "
            f"inclination={inclination:.3f}."
        )

    slat_rad = np.radians(sub_solar_latitude)
    olat_rad = np.radians(sub_observer_latitude)
    cos_phase = np.sin(slat_rad) * np.sin(olat_rad) + np.cos(slat_rad) * np.cos(
        olat_rad
    ) * np.cos(np.radians(sub_solar_longitude - sub_observer_longitude))

    phase_angle = float(np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0))))
    config["OBJECT-PHASE-ANGLE"] = phase_angle


def calculate_luminosity(config: dict) -> float:
    """
    Calculate luminosity using the Stefan-Boltzmann Law to calculate luminosity.
    """
    star_radius = config["OBJECT-STAR-RADIUS"]
    temperature = config["OBJECT-STAR-TEMPERATURE"]
    return (
        4 * np.pi * (star_radius * R_sun.value) ** 2 * sigma_sb.value * temperature**4
    )


def set_habitable_zone_distance(config: dict) -> float:
    """
    Calculates and sets the habitable zone distance based on the star's
    luminosity and temperature.

    Notes
    -----
    Source: Habitable zones around main-sequence stars...
    ([Kopparapu et al. (2014)](https://iopscience.iop.org/article/10.1088/2041-8205/787/2/L29/pdf)).
    See Equation 4 and Table 1 from Kopparapu et al. (2013)
    """
    temp = config["OBJECT-STAR-TEMPERATURE"] - 5780
    luminosity_star = calculate_luminosity(config)

    if not 2600.0 <= float(config["OBJECT-STAR-TEMPERATURE"]) <= 7200.0:
        raise ValueError("Kopparapu HZ coefficients require 2600 <= Teff <= 7200 K.")

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
    distance = np.random.uniform(lower_dist, upper_dist)
    config["OBJECT-STAR-DISTANCE"] = distance

    return distance


def _boiling_temperature(pressure_mbar: float) -> float:
    """Approximate the pure-water boiling temperature with Clausius--Clapeyron."""
    reference_pressure_mbar = 1013.25
    reference_temperature_k = 373.15
    latent_heat_j_mol = 40.65e3
    return float(
        1.0
        / (
            1.0 / reference_temperature_k
            - (R.value / latent_heat_j_mol)
            * np.log(pressure_mbar / reference_pressure_mbar)
        )
    )


def _sample_surface_temperature(
    real_insolation: float,
    pressure_mbar: float,
) -> dict:
    """Draw a liquid-water-compatible temperature with a grey greenhouse prior."""

    # HABEX FINAL REPORT: https://www.jpl.nasa.gov/habex/pdf/HabEx-Final-Report-Public-Release-LINKED-0924.pdf
    # The albedo can reasonably be assumed to be between 0.06 and 0.96. Earth-size HZ planets with a lower albedo,
    # if they exist, would actually be impossible to detect in the first place.
    shortwave_albedo = float(np.random.uniform(0.06, 0.96))
    atmospheric_ir_emissivity = float(np.random.uniform(0.0, 1.0))

    equilibrium_temperature = float(
        ((1.0 - shortwave_albedo) * real_insolation * 1361.0 / (4.0 * sigma_sb.value))
        ** 0.25
    )

    surface_temperature = float(
        equilibrium_temperature * (2.0 / (2.0 - atmospheric_ir_emissivity)) ** 0.25
    )

    surface_pressure_pa = pressure_mbar * 100.0
    if surface_pressure_pa <= 611.657:
        return None

    boiling_temperature = _boiling_temperature(pressure_mbar)
    if not 273.15 <= surface_temperature <= boiling_temperature:
        return None

    return {
        "shortwave_albedo": shortwave_albedo,
        "atmospheric_ir_emissivity": atmospheric_ir_emissivity,
        "equilibrium_temperature_k": equilibrium_temperature,
        "surface_temperature_k": surface_temperature,
        "boiling_temperature_k": boiling_temperature,
    }


def rescale_pt_profile(
    config: dict,
    layers: int,
    pressure_mbar: float,
    surface_temperature: float,
) -> None:
    """Rescale template P--T boundary conditions while preserving vertical shape."""
    parsed = [_layer_values(config, i) for i in range(layers)]
    pressure = np.asarray([float(values[0]) for values in parsed], dtype=float)
    temperature = np.asarray([float(values[1]) for values in parsed], dtype=float)

    surface_index = int(np.argmax(pressure))
    target_surface_pressure_bar = pressure_mbar / 1000.0
    pressure_scale = target_surface_pressure_bar / pressure[surface_index]
    temperature_shift = surface_temperature - temperature[surface_index]

    new_pressure = pressure * pressure_scale
    new_temperature = temperature + temperature_shift

    for i, values in enumerate(parsed):
        values[0] = f"{new_pressure[i]:.12e}"
        values[1] = f"{new_temperature[i]:.8f}"
        config[f"ATMOSPHERE-LAYER-{i + 1}"] = ",".join(values)

    config["ATMOSPHERE-LAYERS"] = layers
    config["ATMOSPHERE-PRESSURE"] = float(pressure_mbar)
    config["ATMOSPHERE-PUNIT"] = "mbar"
    config["ATMOSPHERE-TEMPERATURE"] = float(surface_temperature)
    config["SURFACE-TEMPERATURE"] = float(surface_temperature)


def maintain_planetary_atmosphere(
    config: dict,
    attempts: int = 200,
    layers: int = 60,
) -> None:
    """
    Draw an accepted rocky planet and commit it to a PSG configuration.

    Each attempt re-samples the semi-major axis, planet, shortwave albedo, and
    effective atmospheric infrared emissivity.  The cosmic shoreline is used as
    an empirical first-order retention filter. Accepted planets also satisfy the
    pure-water liquid phase criterion.
    """
    if attempts <= 0:
        raise ValueError("attempts must be positive.")

    stellar_luminosity = calculate_luminosity(config)

    for _ in range(1, attempts + 1):
        semi_major_axis = set_habitable_zone_distance(config)

        # Define the planet's radius in Earth radii; targeting terrestrial planets
        # up to about 1.23 Earth radii. Based on findings from "A rocky exoplanet classification method
        # and its application to calculating surface pressure and surface temperature"
        # Reference: https://arxiv.org/abs/2301.03348
        planet_radius = float(np.random.uniform(0.3, 1.23))

        # # Calculate planet mass based on planet radius using the relationship
        # derived from Equation (2) of the paper "A rocky exoplanet classification method
        # and its application to calculating surface pressure and surface temperature"
        # by McIntyre et al. (2023)
        mass_radius_exponent = 0.279 + float(np.random.uniform(-0.009, 0.009))
        planet_mass = float(planet_radius ** (1.0 / mass_radius_exponent))

        # Calculate planetary gravity (g = GM/r²) in m/s² using the mass and radius  estimates.
        radius_m = planet_radius * R_earth.value
        gravity = float(G.value * (planet_mass * M_earth.value) / radius_m**2)

        # Calculate escape velocity from the planet's surface in m/s
        escape_velocity = float(np.sqrt(2.0 * gravity * radius_m))

        # Compute the insolation, considering the star-planet
        # distance and stellar luminosity. Reference for insolation calculations:
        # Zahnle and Catling (2017), particularly their Equation (4)
        # https://arxiv.org/pdf/1702.03386
        real_insolation = float((stellar_luminosity / L_sun.value) / semi_major_axis**2)

        # Estimate the critical insolation for atmospheric retention based on escape
        # velocity, using an empirically derived relationship from Zahnle and
        # Catling (2017), with approximation from graph analysis.
        # See equation (7) by McIntyre et al. (2023)
        cosmic_shoreline = float(5.0e-16 * escape_velocity**4)
        if real_insolation >= cosmic_shoreline:
            continue

        # [REVISION 2 -- mass-radius exponent]
        # The surface-pressure exponent is DERIVED from the mass-radius exponent
        # drawn above, not sampled a second time.  Combining Eq. 4 (R ~ M^q) with
        # Eq. 5 (P ~ M^2 / R^4) gives P ~ R^(2/q - 4), so for q = 0.279 the
        # exponent is 2/0.279 - 4 = 3.168, and propagating the 0.009 uncertainty
        # gives (2/0.279^2) * 0.009 = 0.231.  The previous version drew
        # 3.168 +/- 0.232 independently, which gave corr(q, exponent_P) = 0 where
        # Eq. 6 requires -1 (the exponent is decreasing in q).  Physically, that
        # allowed a planet to receive the mass, gravity, escape velocity and
        # cosmic-shoreline verdict of a denser body together with the surface
        # pressure of a lighter one.
        pressure_exponent = 2.0 / mass_radius_exponent - 4.0
        pressure_mbar = float(1013.25 * planet_radius**pressure_exponent)
        climate = _sample_surface_temperature(real_insolation, pressure_mbar)
        if climate is None:
            continue

        planet_diameter_km = float(2.0 * planet_radius * R_earth.value / 1000.0)
        surface_temperature = climate["surface_temperature_k"]

        config["SURFACE-ALBEDO"] = climate["shortwave_albedo"]
        config["SURFACE-EMISSIVITY"] = 1.0
        config["OBJECT-DIAMETER"] = planet_diameter_km
        config["OBJECT-GRAVITY"] = gravity
        config["OBJECT-GRAVITY-UNIT"] = "g"
        config["ATMOSPHERE-PUNIT"] = "mbar"
        config["GENERATOR-TRANS-APPLY"] = "N"

        rescale_pt_profile(
            config,
            layers=layers,
            pressure_mbar=pressure_mbar,
            surface_temperature=surface_temperature,
        )

        return  # Exit

    raise ValueError(
        "Exhausted all attempts to generate an atmosphere-retaining rocky "
        "planet compatible with the adopted liquid-water criterion."
    )


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
        Valid options are 'SS-NIR', 'SS-UV', 'SS-Vis', 'B-NIR', 'B-UV', 'B-Vis'

    Notes
    -----
    The function updates the settings dictionary directly based on the instrument
    specification, applying predefined configurations for optical and infrared
    channels, including noise characteristics and other operational parameters.
    """
    if instrument == "SS-Vis":
        pass

    valid_instruments = ["SS-NIR", "SS-UV", "SS-Vis", "B-NIR", "B-UV", "B-Vis"]
    if instrument not in valid_instruments:
        raise ValueError(f"Instrument must be one of {valid_instruments}.")

    elif instrument == "SS-NIR":
        config[
            "GENERATOR-INSTRUMENT"
        ] = """HabEx_SS-NIR: The HabEx StarShade (SS) will provide extraordinary high-contrast capabilities from the UV (0.2 to 0.45 um), to the\
visible (0.45 to 1um), and to the infrared (0.975 to 1.8 um). By limiting the number of optical surfaces, this configuration provides\
high optical throughput (0.2 to 0.4) across this broad of wavelengths, while the quantum efficiency (QE) is expected to be 0.9 for the\
VU and visible detectors and 0.6 for the infrared detector. The UV channel provides a resolution (RP) of 7, visible channel a maximum\
of 140 and the infrared 40."""
        config["GENERATOR-RANGE1"] = 0.975
        config["GENERATOR-RANGE2"] = 1.80
        config["GENERATOR-RESOLUTION"] = 40
        config[
            "GENERATOR-TELESCOPE3"
        ] = """7e-11@-0.000e+00,7e-11@-1.995e-02,7e-11@-3.830e-02,3.544e-03@-5.439e-02,1.949e-02@-6.791e-02,\
3.367e-02@-7.434e-02,6.734e-02@-7.982e-02,1.241e-01@-8.561e-02,2.091e-01@-9.108e-02,2.818e-01@-9.526e-02,3.332e-01@-9.752e-02,\
3.987e-01@-1.014e-01,4.661e-01@-1.052e-01,5.352e-01@-1.075e-01,6.008e-01@-1.110e-01,6.344e-01@-1.130e-01,6.699e-01@-1.155e-01,\
6.911e-01@-1.184e-01,7.000e-01@-1.278e-01,7.000e-01@-1.561e-01,7.000e-01@-1.950e-01,7.000e-01@-2.224e-01,7.000e-01@-2.349e-01"""
        config["GENERATOR-NOISEFRAMES"] = 3600
        config["GENERATOR-NOISETIME"] = 1000

    elif instrument == "SS-UV":
        config[
            "GENERATOR-INSTRUMENT"
        ] = """HabEx_SS-UV: The HabEx StarShade (SS) will provide extraordinary high-contrast capabilities from the UV (0.2 to 0.45 um), to the visible\
(0.45 to 1um), and to the infrared (0.975 to 1.8 um). By limiting the number of optical surfaces, this configuration provides high optical\
throughput (0.2 to 0.4) across this broad of wavelengths, while the quantum efficiency (QE) is expected to be 0.9 for the VU and visible\
detectors and 0.6 for the infrared detector. The UV channel provides a resolution (RP) of 7, visible channel a maximum of 140 and\
the infrared 40."""
        config["GENERATOR-RANGE1"] = 0.2
        config["GENERATOR-RANGE2"] = 0.45
        config["GENERATOR-RESOLUTION"] = 7
        config[
            "GENERATOR-TELESCOPE3"
        ] = """7e-11@-0.000e+00,7e-11@-7.483e-03,7e-11@-1.436e-02,3.544e-03@-2.040e-02,1.949e-02@-2.547e-02,\
3.367e-02@-2.788e-02,6.734e-02@-2.993e-02,1.241e-01@-3.210e-02,2.091e-01@-3.416e-02,2.818e-01@-3.572e-02,3.332e-01@-3.657e-02,\
3.987e-01@-3.802e-02,4.661e-01@-3.947e-02,5.352e-01@-4.031e-02,6.008e-01@-4.164e-02,6.344e-01@-4.236e-02,6.699e-01@-4.333e-02,\
6.911e-01@-4.441e-02,7.000e-01@-4.791e-02,7.000e-01@-5.853e-02,7.000e-01@-7.314e-02,7.000e-01@-8.340e-02,7.000e-01@-8.810e-02"""
        config["GENERATOR-NOISEFRAMES"] = 3600
        config["GENERATOR-NOISETIME"] = 1000

    elif instrument == "B-NIR":
        config[
            "GENERATOR-INSTRUMENT"
        ] = """LUVOIR_B-NIR: The Extreme Coronagraph for Living Planetary Systems (ECLIPS) delivers continuous spectral\
coverage from 200 nm to 2.5 um via three channels, UV (200 to 525 nm), VIS (515 nm to 1030 nm), and NIR (1 to 2 microns). The UV channel is effectively\
an imager and provides a maximum resolution of RP=7, while the VIS channel RP=140, and NIR=70. The core coronagraph throughput is practically twice for\
LUVOIR-B than A."""
        config["GENERATOR-RANGE1"] = 1.01
        config["GENERATOR-RANGE2"] = 2.0
        config["GENERATOR-RESOLUTION"] = 70
        config["GENERATOR-DIAMTELE"] = 8.0
        config[
            "GENERATOR-TELESCOPE3"
        ] = """4.578000e-11@0.000,4.578000e-11@0.216,4.578000e-11@0.649,2.110e-03@0.973,1.266e-02@1.459,\
8.228e-02@2.108,1.709e-01@2.973,2.658e-01@4.486,3.418e-01@6.757,3.945e-01@10.216,4.219e-01@14.108,4.409e-01@19.459,4.536e-01@23.622,\
4.578e-01@27.784,4.578e-01@29.459"""
        config["GENERATOR-NOISE1"] = "0@0.2,0@1,2.5@1.01,2.5@2.0"
        config["GENERATOR-NOISE2"] = "3e-5@0.2,3e-5@1,2e-3@1.01,2e-3@2.0"
        config[
            "GENERATOR-NOISEOEFF"
        ] = """0.0317@0.2000,0.0437@0.2261,0.0589@0.2580,0.0742@0.2986,0.0851@0.3377,0.0917@0.3667,0.0971@0.4029,\
0.1015@0.4493,0.1004@0.4971,0.1004@0.5140,0.1670@0.5150,0.1659@0.5377,0.1506@0.6304,0.1255@0.7087,0.0939@0.7986,0.0884@0.8435,0.1146@0.9058,\
0.1419@0.9594,0.1594@0.9942,0.1821@1.2200,0.1958@1.4100,0.2049@1.6200,0.2094@1.8700,0.2140@2.0000"""
        config["GENERATOR-NOISEFRAMES"] = 3600
        config["GENERATOR-NOISETIME"] = 1000
        config["GENERATOR-TRANS"] = "03-01"
        config["GENERATOR-CONT-STELLAR"] = "Y"

    elif instrument == "B-Vis":
        config[
            "GENERATOR-INSTRUMENT"
        ] = """LUVOIR_B-VIS: The Extreme Coronagraph for Living Planetary Systems (ECLIPS) delivers continuous spectral\
coverage from 200 nm to 2.5 um via three channels, UV (200 to 525 nm), VIS (515 nm to 1030 nm), and NIR (1 to 2 microns). The UV channel is effectively\
an imager and provides a maximum resolution of RP=7, while the VIS channel RP=140, and NIR=70. The core coronagraph throughput is practically twice for\
LUVOIR-B than A."""
        config["GENERATOR-RANGE1"] = 0.515
        config["GENERATOR-RANGE2"] = 1.0
        config["GENERATOR-RESOLUTION"] = 140
        config["GENERATOR-DIAMTELE"] = 8.0
        config[
            "GENERATOR-TELESCOPE3"
        ] = """4.578000e-11@0.000,4.578000e-11@0.216,4.578000e-11@0.649,2.110e-03@0.973,1.266e-02@1.459,8.228e-02@2.108,\
1.709e-01@2.973,2.658e-01@4.486,3.418e-01@6.757,3.945e-01@10.216,4.219e-01@14.108,4.409e-01@19.459,4.536e-01@23.622,4.578e-01@27.784,4.578e-01@29.459"""
        config["GENERATOR-NOISE1"] = "0@0.2,0@1,2.5@1.01,2.5@2.0"
        config["GENERATOR-NOISE2"] = "3e-5@0.2,3e-5@1,2e-3@1.01,2e-3@2.0"
        config[
            "GENERATOR-NOISEOEFF"
        ] = """0.0317@0.2000,0.0437@0.2261,0.0589@0.2580,0.0742@0.2986,0.0851@0.3377,0.0917@0.3667,0.0971@0.4029,\
0.1015@0.4493,0.1004@0.4971,0.1004@0.5140,0.1670@0.5150,0.1659@0.5377,0.1506@0.6304,0.1255@0.7087,0.0939@0.7986,0.0884@0.8435,0.1146@0.9058,\
0.1419@0.9594,0.1594@0.9942,0.1821@1.2200,0.1958@1.4100,0.2049@1.6200,0.2094@1.8700,0.2140@2.0000"""
        config["GENERATOR-NOISEFRAMES"] = 3600
        config["GENERATOR-NOISETIME"] = 1000
        config["GENERATOR-TRANS"] = "03-01"
        config["GENERATOR-CONT-STELLAR"] = "Y"

    elif instrument == "B-UV":
        config[
            "GENERATOR-INSTRUMENT"
        ] = """LUVOIR_B-UV: The Extreme Coronagraph for Living Planetary Systems (ECLIPS) delivers continuous spectral\
coverage from 200 nm to 2.5 um via three channels, UV (200 to 525 nm), VIS (515 nm to 1030 nm), and NIR (1 to 2 microns). The UV channel is effectively\
an imager and provides a maximum resolution of RP=7, while the VIS channel RP=140, and NIR=70. The core coronagraph throughput is practically twice for\
LUVOIR-B than A."""
        config["GENERATOR-RANGE1"] = 0.2
        config["GENERATOR-RANGE2"] = 0.515
        config["GENERATOR-RESOLUTION"] = 7
        config["GENERATOR-DIAMTELE"] = 8.0
        config[
            "GENERATOR-TELESCOPE3"
        ] = """4.578000e-11@0.000,4.578000e-11@0.216,4.578000e-11@0.649,2.110e-03@0.973,1.266e-02@1.459,8.228e-02@2.108,\
1.709e-01@2.973,2.658e-01@4.486,3.418e-01@6.757,3.945e-01@10.216,4.219e-01@14.108,4.409e-01@19.459,4.536e-01@23.622,4.578e-01@27.784,4.578e-01@29.459"""
        config["GENERATOR-NOISE1"] = "0@0.2,0@1,2.5@1.01,2.5@2.0"
        config["GENERATOR-NOISE2"] = "3e-5@0.2,3e-5@1,2e-3@1.01,2e-3@2.0"
        config[
            "GENERATOR-NOISEOEFF"
        ] = """0.0317@0.2000,0.0437@0.2261,0.0589@0.2580,0.0742@0.2986,0.0851@0.3377,0.0917@0.3667,0.0971@0.4029,\
0.1015@0.4493,0.1004@0.4971,0.1004@0.5140,0.1670@0.5150,0.1659@0.5377,0.1506@0.6304,0.1255@0.7087,0.0939@0.7986,0.0884@0.8435,0.1146@0.9058,\
0.1419@0.9594,0.1594@0.9942,0.1821@1.2200,0.1958@1.4100,0.2049@1.6200,0.2094@1.8700,0.2140@2.0000"""
        config["GENERATOR-NOISEFRAMES"] = 3600
        config["GENERATOR-NOISETIME"] = 1000
        config["GENERATOR-TRANS"] = "03-01"
        config["GENERATOR-CONT-STELLAR"] = "Y"


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
    config["ATMOSPHERE-LAYERS"] = layers
    config["ATMOSPHERE-PUNIT"] = "mbar"
    config["OBJECT-GRAVITY-UNIT"] = "g"
    config["GEOMETRY-ALTITUDE-UNIT"] = "pc"
    config["GENERATOR-TRANS-APPLY"] = "N"

    mixing_ratio_constant(config, layers)
    random_atmospheric_layers(config, layers)
    normalize_layer(config, layers, molweight)
    set_spectral_type(config)
    set_stellar_parameters(config)
    set_solar_coordinates(config)
    maintain_planetary_atmosphere(config, attempts=200, layers=layers)
