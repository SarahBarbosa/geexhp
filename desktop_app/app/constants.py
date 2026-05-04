import numpy as np

TELESCOPES = ("LUVOIR", "HABEX")

TELESCOPE_CONFIG = {
    "LUVOIR": {
        "label": "LUVOIR-B",
        "prefix": "B",
        "aperture_m": 8.0,
        "bins": {"UV": 8, "Vis": 94, "NIR": 49},
        "ranges_um": {
            "UV": (0.20, 0.515),
            "Vis": (0.515, 1.030),
            "NIR": (1.030, 2.000),
        },
        "model_file": "luvoir_model.keras",
        "uncertainty_key": "luvoir",
    },
    "HABEX": {
        "label": "HabEx/SS",
        "prefix": "SS",
        "aperture_m": 4.0,
        "bins": {"UV": 7, "Vis": 109, "NIR": 25},
        "ranges_um": {"UV": (0.20, 0.45), "Vis": (0.45, 0.975), "NIR": (0.975, 1.80)},
        "model_file": "habex_model.keras",
        "uncertainty_key": "habex",
    },
}

BANDS = ("UV", "Vis", "NIR")

PHYS_PARAMS = (
    "OBJECT-RADIUS-REL-EARTH",
    "OBJECT-GRAVITY",
    "ATMOSPHERE-TEMPERATURE",
    "ATMOSPHERE-PRESSURE",
)
MAIN_CHEM = ("O2", "O3")
OTHER_CHEM = ("CH4", "CO2", "H2O", "N2")
ALL_PARAMS = PHYS_PARAMS + MAIN_CHEM + OTHER_CHEM

PARAM_LABELS = (
    r"R$_\oplus$",
    r"$g$",
    r"$T_{\rm surf}$",
    r"$P_{\rm surf}$",
    r"O$_2$",
    r"O$_3$",
    r"CH$_4$",
    r"CO$_2$",
    r"H$_2$O",
    r"N$_2$",
)

PARAM_UNITS = (
    r"R$_\oplus$",
    r"m s$^{-2}$",
    "K",
    "mbar",
    "vmr",
    "vmr",
    "vmr",
    "vmr",
    "vmr",
    "vmr",
)


PARAM_UNITS_TEXT = (
    "R⊕",
    "m s⁻²",
    "K",
    "mbar",
    "vmr",
    "vmr",
    "vmr",
    "vmr",
    "vmr",
    "vmr",
)

PARAM_PLAIN = (
    "Radius",
    "Gravity",
    "Surface T",
    "Surface P",
    "O2",
    "O3",
    "CH4",
    "CO2",
    "H2O",
    "N2",
)

CHEM_IDX = {"O2": 4, "O3": 5, "CH4": 6, "CO2": 7, "H2O": 8, "N2": 9}

ERA_ORDER = ("modern", "proterozoic", "archean")
ERA_COLORS = {
    "modern": "#1f77b4",
    "proterozoic": "#2ca02c",
    "archean": "#ff7f0e",
}

TEL_COLORS = {"LUVOIR": "#223648", "HABEX": "#b2182b"}
DETECTION_SIGMA = 3.0


def make_wavelengths(telescope: str) -> dict[str, np.ndarray]:
    cfg = TELESCOPE_CONFIG[telescope]
    out = {}
    for band in BANDS:
        lo, hi = cfg["ranges_um"][band]
        n = cfg["bins"][band]
        edges = np.linspace(lo, hi, n + 1)
        out[band] = 0.5 * (edges[:-1] + edges[1:])
    return out
