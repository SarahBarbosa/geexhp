
Random Dataset Generation Guide
==============================

Introduction
============
This guide explains how to generate random planetary datasets based on different geological eras (Modern, Proterozoic, Archean) and random planets with specific atmospheric conditions or noise data.

Loading the Dataset Generator
=============================
To generate random planetary datasets, you need to create a ``DataGen`` object using a configuration file and specifying a URL for the PSG API. This URL can point to either a local or a remote API, depending on how you run the PSG.

If you're running PSG locally inside a Docker container, the API URL will usually be something like `http://127.0.0.1:3000/api.php`.

Once the Docker container is running, you can configure the URL and dynamically set the configuration file path in your Python script as follows:

.. code-block:: python

    import os
    from geexhp import datagen

    dg = datagen.DataGen(url="http://127.0.0.1:3000/api.php")

Era-Specific Customization
==========================
By default, ``datagen.DataGen``` uses the "modern" era if no specific era is mentioned. The "modern" era represents recent Earth-like conditions. However, you can change this to generate data for different geological eras such as "proterozoic" or "archean."

Changing to Other Eras
----------------------
To generate datasets for different eras, you must specify the ``stage`` parameter.

.. code-block:: python
    
    # Proterozoic Era
    dg_proterozoic = datagen.DataGen(url="http://127.0.0.1:3000/api.php", stage="proterozoic")

    # Archean Era
    dg_archean = datagen.DataGen(url="http://127.0.0.1:3000/api.php", stage="archean")

Generating Random Data for Different Geological Eras
=====================================================

Parameters for ``dg.generator``
-----------------------------

- ``start``: The starting index for the range of planets to generate data for.
- ``end``: The ending index for the range of planets to generate data for.
- ``random_atm``: Set to ``True`` to generate random atmospheres, or ``False`` to use a fixed configuration.
- ``verbose``: When ``True``, enables detailed output during the data generation process.
- ``output_file``: The output file name, which stores the generated dataset.
- ``instruments``: Specifies the instrument(s) to generate data for. Options are:
    - ``"all"``: Processes all instruments (default behavior).
    - ``"SS"``: Processes all "SS" instruments ("SS-NIR", "SS-UV", "SS-Vis").
    - Specific instrument name as a string (e.g., ``"HWC"``, ``"SS-NIR"``).
    - List of instrument names (e.g., [``"SS-NIR"``, ``"SS-Vis"``]).

In a multi-threaded or parallel, you can split this range ``(start,end)`` among different threads to speed up the generation process.

Example 
-------

.. code-block:: python

    # Modern Era 
    dg.generator(           # Or dg_proterozoic or dg_archean
        start=0, end=8,     # A dataset with 8 planets  
        random_atm=False,
        verbose=True,
        output_file="modern_0-8",  # Just a example
        instruments="all"   # Processes all instruments (default)
    )

Generating Random Planets with an Isothermal Profile
====================================================
For certain simulations, you may want to generate planets with a completely random atmospheric composition that is assumed to be isothermal across all layers. In this case, set the ``random_atm`` parameter to ``True``. When ``random_atm=True``, the ``molweight`` parameter is not required, as the atmospheric composition is randomly generated.

Molecules in Random Atmosphere Generation
-----------------------------------------

The random atmosphere generation includes the following molecules:

* H₂O (Water vapor)
* CO₂ (Carbon dioxide)
* CH₄ (Methane)
* O₂ (Oxygen)
* NH₃ (Ammonia)
* HCN (Hydrogen cyanide)
* PH₃ (Phosphine)
* H₂ (Hydrogen molecule)

Example Code
------------
To generate planets with an isothermal profile:

.. code-block:: python

    dg.generator(           # It doesn't matter the stage here
        start=0, end=8,
        random_atm=True,    # Random atmosphere generation enabled
        verbose=True,
        output_file="random_0-8"   # Output file
    )

Visualizing the Data
====================
After generating the datasets, use the ``datavis`` library to visualize the spectra of the generated planets.

Configuring Matplotlib for Visualizations
-----------------------------------------
Before visualizing the spectra, you can configure matplotlib parameters using the ``datavis.configure_matplotlib`` function. 
This allows you to customize the appearance of the plots. The function provides a flexible way to configure either a modern or an "old-school" style for the plots.

.. code-block:: python

    from geexhp import datavis
    datavis.configure_matplotlib(oldschool=False)

* ``oldschool`` parameter:
    * If ``oldschool=True``, it imports ``smplotlib`` for traditional plotting styles.
    * If ``oldschool=False``, it updates various ``matplotlib`` settings for a more modern appearance (my style, feel free to be an artist too)

The `datavis.plot_spectrum`` function has been enhanced to allow plotting spectra from multiple instruments.

* Parameters Explained: 
    * ``df``: The DataFrame containing the spectrum data.
    * ``label``: Optional label for the plot legend. If not provided, the instrument names are used.
    * ``index``: The index of the planet in the DataFrame. If None, assumes the DataFrame contains data for a single planet.
    * ``instruments``: A string or list of instrument names to plot. Valid instruments are "HWC", "SS-UV", "SS-Vis", and "SS-NIR". If None, the function plots HWC data on one plot and combines SS instruments on a separate plot.
    * ``ax``: An Axes object or list of Axes to plot on. If None, new figures and axes are created.
    * ``noise``: If True, plots the noisy data with error bars.
    * ``**kwargs``: Additional keyword arguments passed to the plotting functions for further customization.

.. code-block:: python

    # Assume 'data' is your DataFrame containing the spectra data
    # Plot HWC data for the planet at index 1
    datavis.plot_spectrum(data, label="Planet X", index=1, instruments="HWC");

    # Plot SS instruments data for the planet at index 1
    datavis.plot_spectrum(data, label="Planet X", index=1, noise=True, instruments=["SS-UV", "SS-Vis", "SS-NIR"]);

    # Plot HWC and combined SS instruments on separate plots
    datavis.plot_spectrum(data, index=1);

Or, if you want visualize the noise data, use ``noise=True`` parameter:

.. code-block:: python

    datavis.plot_spectrum(data, label="Planet X", index=1, noise=True)

The noise column comes from the telescope observation with a distance assumption of 3 parsecs. The noise is generated using a Gaussian distribution, where the mean is the total model and the standard deviation is the 1-sigma noise.