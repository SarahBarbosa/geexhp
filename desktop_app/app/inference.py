from pathlib import Path

import numpy as np
import tensorflow as tf

from desktop_app.app.constants import (
    BANDS,
    PHYS_PARAMS,
    MAIN_CHEM,
    OTHER_CHEM,
    TELESCOPE_CONFIG,
)
from desktop_app.app.data import DataStore, Spectrum


class Retriever:
    def __init__(self, store: DataStore):
        self.store = store
        self._models: dict[str, tf.keras.Model] = {}

    def get_model(self, telescope: str) -> tf.keras.Model:
        if telescope not in self._models:
            path = (
                self.store.project_root
                / "models"
                / TELESCOPE_CONFIG[telescope]["model_file"]
            )
            self._models[telescope] = tf.keras.models.load_model(str(path))
        return self._models[telescope]

    def _norm_spectrum(self, spec: Spectrum) -> dict[str, np.ndarray]:
        cfg = TELESCOPE_CONFIG[spec.telescope]
        pfx = cfg["prefix"]
        out = {}
        stats = self.store.norm_stats["inputs"]
        for band in BANDS:
            mean = stats[f"{pfx}-{band}"]["mean"]
            std = stats[f"{pfx}-{band}"]["std"]
            x = (spec.noisy_albedo[band] - mean) / std
            out[f"NOISY_ALBEDO_{pfx}-{band}"] = x.reshape(1, -1, 1).astype(np.float32)
        return out

    def _denorm_phys(self, z: float, name: str) -> float:
        s = self.store.norm_stats["outputs"][name]
        return float((z ** s["best_n"]) * (s["max"] - s["min"]) + s["min"])

    def _denorm_chem(self, z: float, name: str) -> float:
        n = self.store.norm_stats["outputs"][name]["best_n"]
        return float(z**n)

    def _outputs_to_matrix(self, preds) -> np.ndarray:
        z_phys = preds["physical_output"].numpy()
        z_main = preds["main_chemical_output"].numpy()
        z_other = preds["other_chemical_output"].numpy()

        cols = []
        for j, name in enumerate(PHYS_PARAMS):
            vals = [self._denorm_phys(float(z), name) for z in z_phys[:, j]]
            cols.append(np.asarray(vals, dtype=float))
        for j, name in enumerate(MAIN_CHEM):
            n = self.store.norm_stats["outputs"][name]["best_n"]
            mr = np.clip(z_main[:, j], 0.0, None) ** n
            cols.append(np.where(mr > 0, np.log10(mr), np.nan))
        for j, name in enumerate(OTHER_CHEM):
            n = self.store.norm_stats["outputs"][name]["best_n"]
            mr = np.clip(z_other[:, j], 0.0, None) ** n
            cols.append(np.where(mr > 0, np.log10(mr), np.nan))
        return np.column_stack(cols)

    def _norm_spectrum_batch(
        self, spec: Spectrum, noisy_by_band: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        cfg = TELESCOPE_CONFIG[spec.telescope]
        pfx = cfg["prefix"]
        out = {}
        stats = self.store.norm_stats["inputs"]
        for band in BANDS:
            mean = stats[f"{pfx}-{band}"]["mean"]
            std = stats[f"{pfx}-{band}"]["std"]
            x = (noisy_by_band[band] - mean) / std
            out[f"NOISY_ALBEDO_{pfx}-{band}"] = x[:, :, np.newaxis].astype(np.float32)
        return out

    def predict(self, spec: Spectrum) -> tuple[dict[str, float], dict[str, float]]:
        model = self.get_model(spec.telescope)
        x = self._norm_spectrum(spec)
        preds = model(x, training=False)

        z_phys = preds["physical_output"].numpy().reshape(-1)
        z_main = preds["main_chemical_output"].numpy().reshape(-1)
        z_other = preds["other_chemical_output"].numpy().reshape(-1)

        physical: dict[str, float] = {}
        z_pred: dict[str, float] = {}
        for j, name in enumerate(PHYS_PARAMS):
            z_pred[name] = float(z_phys[j])
            physical[name] = self._denorm_phys(z_pred[name], name)
        for j, name in enumerate(MAIN_CHEM):
            z_pred[name] = float(z_main[j])
            physical[name] = self._denorm_chem(z_pred[name], name)
        for j, name in enumerate(OTHER_CHEM):
            z_pred[name] = float(z_other[j])
            physical[name] = self._denorm_chem(z_pred[name], name)
        return physical, z_pred

    def bootstrap_samples(
        self, spec: Spectrum, n_samples: int = 250, seed: int = 123
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noisy = {}
        for band in BANDS:
            center = spec.noisy_albedo[band][np.newaxis, :]
            noise = spec.noise[band][np.newaxis, :]
            eps = rng.standard_normal((n_samples, center.shape[1])).astype(np.float32)
            noisy[band] = center + eps * noise

        model = self.get_model(spec.telescope)
        preds = model(self._norm_spectrum_batch(spec, noisy), training=False)
        return self._outputs_to_matrix(preds)

    def mc_dropout_samples(self, spec: Spectrum, n_samples: int = 5000) -> np.ndarray:
        noisy = {
            band: np.tile(
                spec.noisy_albedo[band][np.newaxis, :],
                (n_samples, 1),
            )
            for band in BANDS
        }
        model = self.get_model(spec.telescope)
        preds = model(self._norm_spectrum_batch(spec, noisy), training=True)
        return self._outputs_to_matrix(preds)

    def sigma_z_to_phys(
        self, sigma_z: np.ndarray, z_pred: dict[str, float]
    ) -> dict[str, float]:
        names = list(PHYS_PARAMS) + list(MAIN_CHEM) + list(OTHER_CHEM)
        out: dict[str, float] = {}
        for j, name in enumerate(names):
            sd_z = float(sigma_z[j])
            z = z_pred[name]
            if name in PHYS_PARAMS:
                lo = max(z - sd_z, 0.0)
                hi = min(z + sd_z, 1.0)
                phys_lo = self._denorm_phys(lo, name)
                phys_hi = self._denorm_phys(hi, name)
            else:
                lo = max(z - sd_z, 0.0)
                hi = z + sd_z
                phys_lo = self._denorm_chem(lo, name)
                phys_hi = self._denorm_chem(hi, name)
            out[name] = 0.5 * abs(phys_hi - phys_lo)
        return out
