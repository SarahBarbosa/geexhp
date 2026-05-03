# geeXHP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15648637.svg)](https://doi.org/10.5281/zenodo.15648637)

`geeXHP` is a Python package for generating synthetic reflected-light spectra of Earth-like exoplanets and preparing machine-learning datasets for atmospheric and planetary retrieval studies.

This repository supports the work presented in:

> Barbosa, S. G. A., Estrela, R., da Silva Filho, P. C. F., Mugnai, L. V., & de Freitas, D. B. (2026). *Towards the Habitable Worlds Observatory: Retrieval of Reflection Spectra from Evolving Earth Analogues using 1D CNNs*. RASTI preprint.

## Overview

The package was developed to generate mission-relevant synthetic spectra for future direct-imaging studies of terrestrial exoplanets, especially in the context of the Habitable Worlds Observatory (HWO). The workflow combines physically motivated Earth-analogue atmospheric scenarios with NASA's Planetary Spectrum Generator (PSG) and machine-learning utilities for 1D CNN retrieval
experiments.

The paper uses `geeXHP` to build a dataset of 108,246 synthetic reflected-light spectra spanning Archean, Proterozoic, and Modern Earth atmospheres under LUVOIR-B and HabEx/SS noise models. The corresponding CNN framework retrieves six atmospheric mixing ratios (`CH4`, `CO2`, `H2O`, `N2`, `O2`, `O3`) and four planetary parameters: radius, surface gravity, surface temperature, and surface pressure.

## Features

- Generate synthetic reflected-light spectra with PSG.
- Sample Earth-like planets across Archean, Proterozoic, and Modern atmospheric compositions.
- Apply habitability and atmospheric-retention filters for physically plausible planet configurations.
- Produce spectra for LUVOIR-B and HabEx/SS instrument configurations.
- Prepare datasets for CNN-based atmospheric and planetary retrieval.
- Convert generated data into machine-learning-ready formats.
- Visualize noiseless and noisy spectra across instrument channels.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/SarahBarbosa/geexhp.git
cd geexhp
pip install -e .
```

For development tools:

```bash
pip install -e .[dev]
```

`geeXHP` requires access to PSG. For large-scale generation, the paper used a locally hosted PSG instance running inside a Docker container.

## Quick Start

```python
from geexhp import datagen

dg = datagen.DataGen(stage="modern")
dg.generator(
    start=0,
    end=8,
    random_atm=False,
    output_file="modern_0-8",
    instruments="LUVOIR",
)
```

This saves the generated dataset to `data/modern_0-8.parquet`.

To generate spectra for other geological stages:

```python
from geexhp import datagen

proterozoic = datagen.DataGen(stage="proterozoic")
archean = datagen.DataGen(stage="archean")
```

Instrument options include:

- `"all"`: all available instrument channels.
- `"LUVOIR"`: LUVOIR-B channels.
- `"SS"`: HabEx/SS channels.
- Specific channels such as `"B-UV"`, `"B-Vis"`, `"B-NIR"`, `"SS-UV"`, `"SS-Vis"`, and `"SS-NIR"`.

## Documentation

Additional usage notes are available in `docs/`:

- `docs/1-psg_installation_steps.rst`: PSG installation notes.
- `docs/2-how_to_use_geexhp.rst`: data-generation and visualization guide.

## Data Availability

The trained CNN models for the HabEx/SS and LUVOIR-B configurations, together with the corresponding training, validation, and test datasets, are available on Zenodo: [https://doi.org/10.5281/zenodo.15648637](https://doi.org/10.5281/zenodo.15648637)

## Acknowledgements

We gratefully acknowledge the financial support from the Brazilian agency CAPES (grant No. 88887.622098/2021-00), as well as the STELLAR TEAM at the Federal University of Ceará for our discussions and collaborative support. Part of the research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration. Special thanks to Geronimo Villanueva for his assistance with setting up the PSG, and to Yui Kawashima for providing data on the Proterozoic Earth.

## License

This project is distributed under the BSD 2-Clause License.
