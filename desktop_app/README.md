# geeXHP Desktop

An offline PySide6 application for exploring the results of **"Towards the Habitable Worlds Observatory: Retrieval of Reflection Spectra from Evolving Earth Analogues using 1D CNNs"** (Barbosa et al. 2026, *RASTI* 000, 1–22).

The app lets you navigate the full 10,826-sample held-out test set, run the LUVOIR-B and HabEx/SS Keras models interactively, inspect $\sigma_\text{total}$ uncertainties, compare telescopes side-by-side, and explore Integrated Gradients sensitivity maps, without writing a single line of code.

## Requirements

| Dependency         | Notes                 |
| ------------------ | --------------------- |
| Python ≥ 3.10     | Tested with 3.12      |
| PySide6 ≥ 6.6     | Qt 6 bindings         |
| TensorFlow ≥ 2.15 | Keras model loading   |
| NumPy, Matplotlib  | Plotting and numerics |

Full pinned versions: [`requirements.txt`](requirements.txt)

The app also requires the data and model artifacts from the repository:

```
data/test.tfrecord              test-set spectra (10 826 samples)
data/normalization_stats.json   per-parameter normalization constants
data/bootstrap/                 pre-computed sigma_model, sigma_data, and IG heatmaps
models/luvoir_model.keras       trained LUVOIR-B retrieval model
models/habex_model.keras        trained HabEx/SS retrieval model
```

These files are managed with **Git LFS**. After cloning, pull them with:

```bash
git lfs pull
```

## Quickstart

### Option 1: double-click (Linux, recommended)

Create a Python environment once, then just double-click the `geeXHP` executable in your file manager:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r desktop_app/requirements.txt
```

`geeXHP` is a small native launcher (source: `launcher.c`) that resolves the repository root and delegates to `run_geexhp.sh`. If your file manager refuses to execute it, right-click → **Properties → Permissions → Allow executing file as program**.

### Option 2: application menu

Run the install script once to register geeXHP in your GNOME / KDE menu:

```bash
bash desktop_app/install_launcher.sh
```

This copies the icon to `~/.local/share/icons/hicolor/` and writes a  `~/.local/share/applications/geexhp.desktop` launcher. After that, searching "geeXHP" in your application menu is enough.

### Option 3: terminal

```bash
source .venv/bin/activate          # or psg-venv
python -m desktop_app.main
```

## What the app does

Six tabs walk through one complete retrieval workflow:

### 1 · Target

Choose a spectrum two ways:

- **Filter test set:** narrow the 10,826 held-out spectra by star type (F / G), geological era (Modern / Proterozoic / Archean), and distance (5-16 pc). The test set mirrors the full training distribution: 34 % Archean, 39 % Proterozoic, 33 % Modern, with F and G host stars.
- **Paste spectrum:** supply a custom apparent-albedo spectrum as a plain-text flux column, a flattened vector, or `wavelength_um, noisy_albedo, noise_1sigma` rows. The app converts it to the model's normalised input tensors and runs a live retrieval.

The selected target's spectrum is shown below the picker with each instrumental band (UV / Vis / NIR) colour-tinted.

### 2 · Retrieve

Run the LUVOIR-B or HabEx/SS Keras model and inspect the results:

- Retrieved values for all 10 parameters are denormalised and displayed with  error bars (combined from the pre-computed  `combined_uncertainty.npy`).
- A **3$\sigma$ detection panel** reports per-species significance.
- A sortable table shows truth vs. predicted vs. $\pm \sigma_{\text{total}}$ for each parameter.

Key test-set performance (Table 4 of the paper):

| Parameter           | R² (LUVOIR-B) | R² (HabEx/SS) |
| ------------------- | -------------- | -------------- |
| O₃                 | 0.996          | 0.996          |
| CH₄                | 0.983          | 0.980          |
| H₂O                | 0.973          | 0.964          |
| CO₂                | 0.951          | 0.957          |
| O₂                 | 0.982          | 0.980          |
| Radius              | 0.831          | 0.837          |
| Gravity             | 0.829          | 0.833          |
| Surface pressure    | 0.818          | 0.826          |
| Surface temperature | 0.500          | 0.377          |

Surface temperature is the least reliable parameter: reflected-light spectra encode very limited thermal information under an isothermal vertical structure assumption.

### 3 · Sensitivity

Integrated Gradients heatmap (from `ig_heatmaps.npy`): for the selected telescope and geological era, shows which wavelengths the network relies on for each chemical species.

- **Rows** = O₂, O₃, CH₄, CO₂, H₂O, N₂
- **Warm cells** = positive retrieval leverage; **cool** = suppression

### 4 · Compare

Overlay LUVOIR-B and HabEx/SS retrievals scaled by ground truth (perfect retrieval = 1.0). Intended for test-set samples where truth is known.

### 5 · Corner

For pasted custom spectra: generate a corner plot from B = 5,000 bootstrap noise realisations and N_MC = 5,000 MC Dropout samples, showing joint posterior distributions for all retrieved parameters.

### 6 · About

Project summary: authors, scientific context, dataset statistics, paper citation, and Zenodo DOI.

## Dataset & models

|                  | LUVOIR-B                | HabEx/SS                |
| ---------------- | ----------------------- | ----------------------- |
| Aperture         | 8 m (coronagraph)       | 4 m (starshade)         |
| UV band          | 0.20-0.515 μm, R = 7   | 0.20-0.45 μm, R = 7    |
| Vis band         | 0.515-1.00 μm, R = 140 | 0.45-0.975 μm, R = 140 |
| NIR band         | 1.00-2.00 μm, R = 70   | 0.975-1.80 μm, R = 40  |
| Input length     | 151 bins                | 141 bins                |
| Model parameters | 500,224                 | 456,760                 |

Training set: **108,246 spectra** generated with NASA's Planetary Spectrum Generator (Villanueva et al. 2018, 2022). Each spectrum is simulated at a random distance (5-10 pc), orbital inclination (0-50°), and orbital phase (0-360°) around an F or G star, with 1,000 h total exposure time per target.

Uncertainty is decomposed into:

- **σ_data** (aleatoric): bootstrap over B = 5,000 independent Gaussian noise realisations per test spectrum.
- **σ_model** (epistemic): Monte Carlo Dropout with N_MC = 5,000 forward passes at fixed noise.
- **σ_total**: combined in quadrature and pre-computed for all 10,826 test samples.

At the nominal mission noise level, chemical species are predominantly **model-limited** rather than photon-limited: improving the inference framework will gain as much as increasing photon collection.

## Citation

working...

## License

BSD 2-Clause. See `LICENSE` in the repository root.
