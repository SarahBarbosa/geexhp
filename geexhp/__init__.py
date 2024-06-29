from .core.datagen import *
from .core.datavis import *
from .core.datamod import *

__version__ = "1.0.0"
__all__ = [
    'DataGen',
    'get_config',
    'modern_earth',
    'configure_matplotlib',
    'plot_spectrum',
    'label_line',
    'mixing_ratio_constant',
    'random_atmospheric_layers',
    'normalize_layer',
    'set_spectral_type',
    'set_stellar_parameters',
    'set_solar_coordinates',
    'set_atmospheric_pressure',
    'calculate_luminosity',
    'set_habitable_zone_distance',
    'maintain_planetary_atmosphere',
    'set_instrument',
    'random_planet'
]