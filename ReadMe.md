# Towards the Habitable Worlds Observatory: 1D CNN Retrieval of Reflection Spectra from Evolving Earth Analogs (Barbosa et al. in prep)

This project is currently under development and is part of my ongoing thesis work.  It includes both the generation of synthetic reflection spectra using the Planetary Spectrum Generator (PSG) and the development of a 1D Convolutional Neural Network (CNN) to retrieve atmospheric and planetary properties from those spectra.  

In future versions, we will release a **Zenodo repository** containing:

- The full training, validation, and test datasets
- Pre-trained CNN models for reproducibility and benchmarking
- Additional metadata and reproducibility scripts

# Install with Git (recommended)

Python 3.7 or later is required.

```bash
git clone https://github.com/SarahBarbosa/geexhp.git
cd geexhp
pip install -e .
```

# Acknowledgements

We gratefully acknowledge the financial support from the Brazilian agency CAPES (grant No. 88887.622098/2021-00), as well as the STELLAR TEAM at the Federal University of Ceara for our discussions and collaborative support. We also extend our thanks to Geronimo Villanueva at NASA Goddard Space Flight Center for his assistance with setting up the PSG, and to Yui Kawashima for providing data on the Proterozoic Earth.