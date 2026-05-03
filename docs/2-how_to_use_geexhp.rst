
How to use geeXHP
=================

Introduction
============
This guide explains how to generate synthetic reflected-light spectra for Earth-like exoplanets across different geological eras: Modern, Proterozoic, and Archean. It also covers random isothermal atmospheric compositions and basic visualization utilities.

Loading the Dataset Generator
=============================
To generate spectra, create a ``DataGen`` object with the URL for the PSG API. The URL can point to either a local or a remote PSG instance, depending on how you run PSG. If you're running PSG locally inside a Docker container, the API URL will usually be something like `http://127.0.0.1:3000/api.php`. However, before proceeding, ensure you are connected by starting the container:

.. code-block:: bash

    docker start psg  

Once the container is running, you can initialize the ``DataGen`` object as follows:

.. code-block:: python

    import os
    from geexhp import datagen

    dg = datagen.DataGen()

Era-Specific Customization
==========================
By default, ``datagen.DataGen`` uses the ``"modern"`` era if no specific era is provided. The ``"modern"`` era represents recent Earth-like atmospheric conditions. You can also generate data for ``"proterozoic"`` or ``"archean"`` atmospheres.

Changing to Other Eras
----------------------
To generate datasets for different eras, you must specify the ``stage`` parameter.

.. code-block:: python
    
    # Proterozoic Era
    dg_proterozoic = datagen.DataGen(stage="proterozoic")

    # Archean Era
    dg_archean = datagen.DataGen(stage="archean")

Generating Random Data for Different Geological Eras
=====================================================

Parameters for ``dg.generator``
-------------------------------

- ``start``: The starting index for the range of planets to generate data for.
- ``end``: The ending index for the range of planets to generate data for.
- ``random_atm``: Set to ``True`` to generate random atmospheres, or ``False`` to use a fixed configuration.
- ``output_file``: The output file name without extension. The dataset is saved as ``data/<output_file>.parquet``.
- ``instruments``: Specifies the instrument(s) to generate data for. Options are:

  - ``"all"``: All instruments (default).
  - ``"SS"``: HabEx/SS instruments (``"SS-NIR"``, ``"SS-UV"``, ``"SS-Vis"``).
  - ``"LUVOIR"``: LUVOIR-B instruments (``"B-NIR"``, ``"B-UV"``, ``"B-Vis"``).
  - Specific instrument name(s) as a string or list, such as ``"B-NIR"`` or ``["SS-NIR", "SS-Vis"]``.

The ``generator`` method saves the generated rows to a Parquet file and returns ``None``. To inspect the generated data, read the saved file with pandas.

For parallel generation, split the ``start`` and ``end`` range across different processes or jobs. For example, one job can generate ``0-200`` and another can generate ``200-400``.

Example 
-------

.. code-block:: python

    # Modern Era 
    dg.generator(            # Or dg_proterozoic or dg_archean
        start=0, end=8,      # A dataset with 8 planets  
        random_atm=False,
        output_file="modern_0-8",
        instruments="all"    # Processes all instruments (default)
    )

    # The file is saved as data/modern_0-8.parquet.
    # You can load it later with:
    # import pandas as pd
    # df = pd.read_parquet("data/modern_0-8.parquet")

Generating Random Planets with an Isothermal Profile
====================================================
For certain simulations, you may want to generate planets with a completely random atmospheric composition that is assumed to be isothermal across all layers. In this case, set the ``random_atm`` parameter to ``True``. When ``random_atm=True``, the ``molweight`` parameter is not required, as the atmospheric composition is randomly generated.

Molecules in Random Atmosphere Generation
-----------------------------------------

The random atmosphere generation includes the following molecules:

* H2O
* CO2
* CH4
* O2
* NH3
* HCN
* PH3
* H2

Example Code
------------
To generate planets with an isothermal profile:

.. code-block:: python

    dg.generator(                  # It doesn't matter the stage here
        start=0, end=8,
        random_atm=True,           # Random atmosphere generation enabled
        output_file="random_0-8"    # Saves data/random_0-8.parquet
    )

Visualizing the Data
====================
After generating the datasets, use the ``datavis`` library to visualize the spectra of the generated planets.

Configuring Matplotlib for Visualizations
-----------------------------------------
Before visualizing the spectra, you can configure matplotlib parameters using the ``datavis.configure_matplotlib`` function. This allows you to customize the appearance of the plots. The function provides a way to configure either the default project style or an ``smplotlib`` style.

.. code-block:: python

    from geexhp import datavis
    datavis.configure_matplotlib(oldschool=False)

* ``oldschool`` parameter:
    * If ``oldschool=True``, it imports ``smplotlib`` for traditional plotting styles.
    * If ``oldschool=False``, it updates several ``matplotlib`` settings used by the project.

The ``datavis.plot_spectrum`` function can plot spectra from multiple instruments.

* Parameters Explained: 
    * ``df``: The DataFrame containing the spectrum data.
    * ``label``: Optional label for the plot legend. If not provided, the instrument names are used.
    * ``index``: The index of the planet in the DataFrame. If None, assumes the DataFrame contains data for a single planet.
    * ``instruments``: A string or list of instrument names to plot. Valid instruments are "B-UV", "B-Vis", "B-NIR", "SS-UV", "SS-Vis", and "SS-NIR". If None, the function plots LUVOIR data on one plot and SS on a separate plot.
    * ``ax``: An Axes object or list of Axes to plot on. If None, new figures and axes are created.
    * ``noise``: If True, plots the noisy data with error bars.
    * ``show_legend``: If True, displays the legend.
    * ``line_color``: Optional color for the albedo curve.
    * ``error_color``: Optional color for the noisy-data error bars.
    * ``**kwargs``: Additional keyword arguments passed to the plotting functions for further customization.

.. code-block:: python

    import pandas as pd
    # Plot SS instruments data for the planet at index 1
    data = pd.read_parquet("data/modern_0-8.parquet")
    datavis.plot_spectrum(data, label="Planet X", index=1, noise=True, instruments=["SS-UV", "SS-Vis", "SS-NIR"]);

    # Plot LUVOIR and SS instruments on separate plots
    datavis.plot_spectrum(data, index=1);

Or, if you want visualize the noise data, use ``noise=True`` parameter:

.. code-block:: python

    datavis.plot_spectrum(data, label="Planet X", index=1, noise=True)

The noise is generated using a Gaussian distribution, where the mean is the total model and the standard deviation is the 1-sigma noise.
