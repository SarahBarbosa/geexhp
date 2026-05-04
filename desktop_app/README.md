# geeXHP Desktop

An offline PySide6 application for exploring the results of **"Towards the Habitable Worlds Observatory: Retrieval of Reflection Spectra from Evolving Earth Analogues using 1D CNNs"** (Barbosa et al. 2026, *RASTI* 000, 1–22).

The app lets you navigate the full 10,826-sample held-out test set, run the LUVOIR-B and HabEx/SS Keras models interactively, inspect $\sigma_\text{total}$ uncertainties, compare telescopes side-by-side, explore Integrated Gradients sensitivity maps, and inspect how a spectrum moves through the neural network, without writing a single line of code.

## Interface highlights

- **Target**: filter the held-out test set or paste a custom spectrum.
- **Network**: follow a spectrum through normalization, residual Conv1D blocks, attention, global pooling, dense layers, and the final retrieval heads. Hover over the architecture to inspect tensor shapes and activation statistics, or click a layer to display the live tensor below.
- **Retrieve**: run LUVOIR-B or HabEx/SS retrievals and inspect uncertainties.
- **Sensitivity**: view era-specific Integrated Gradients heatmaps.
- **Compare**: compare LUVOIR-B and HabEx/SS with physical parameters on independent scales and chemistry in log-space.
- **Corner**: inspect custom-spectrum bootstrap and MC Dropout posteriors in the same contour style used in the paper notebooks.

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
