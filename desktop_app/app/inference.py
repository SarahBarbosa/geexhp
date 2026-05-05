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
        self._trace_models: dict[str, tf.keras.Model] = {}

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

    def _norm_spectrum_full(self, spec: Spectrum) -> np.ndarray:
        x = self._norm_spectrum(spec)
        cfg = TELESCOPE_CONFIG[spec.telescope]
        pfx = cfg["prefix"]
        parts = [
            x[f"NOISY_ALBEDO_{pfx}-{band}"].reshape(-1).astype(float)
            for band in BANDS
        ]
        return np.concatenate(parts)

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

    def _get_trace_model(self, telescope: str) -> tf.keras.Model:
        if telescope not in self._trace_models:
            model = self.get_model(telescope)
            layer_names = [
                "activation_5",
                "activation_7",
                "layer_normalization_3",
                "global_average_pooling1d_1",
                "add_9",
                "physical_output",
                "main_chemical_output",
                "other_chemical_output",
            ]
            outputs = [model.get_layer(name).output for name in layer_names]
            self._trace_models[telescope] = tf.keras.Model(
                inputs=model.inputs,
                outputs=outputs,
                name=f"{telescope.lower()}_trace",
            )
        return self._trace_models[telescope]

    def trace_network(self, spec: Spectrum) -> list[dict]:
        x = self._norm_spectrum(spec)
        trace_model = self._get_trace_model(spec.telescope)
        tensors = trace_model(x, training=False)
        arrays = [np.asarray(t.numpy()).squeeze(axis=0) for t in tensors]

        z_phys = arrays[5].reshape(-1)
        z_main = arrays[6].reshape(-1)
        z_other = arrays[7].reshape(-1)
        z_output = np.concatenate([z_phys, z_main, z_other]).astype(float)

        return [
            {
                "title": "1. Observed spectrum",
                "subtitle": "Noisy reflected-light albedo split into UV, Visible, and NIR bands.",
                "explain": (
                    "This is the measurement the network receives: a low-resolution "
                    "reflected-light spectrum with instrument noise."
                ),
                "reading": (
                    "Absorption bands appear as wavelength-dependent structure. "
                    "The network does not see labels here, only the flux pattern."
                ),
                "kind": "spectrum",
                "values": spec.noisy_full,
                "wave": spec.wave_full,
                "telescope": spec.telescope,
            },
            {
                "title": "2. Normalized model input",
                "subtitle": "Each band is standardized with the training-set mean and standard deviation.",
                "explain": (
                    "The app converts the spectrum into the exact tensors used during "
                    "training, one input array per spectral band."
                ),
                "reading": (
                    "Values near zero are typical for the training set. Strong positive "
                    "or negative excursions mark wavelengths the network can use."
                ),
                "kind": "line",
                "values": self._norm_spectrum_full(spec),
                "wave": spec.wave_full,
                "telescope": spec.telescope,
            },
            {
                "title": "3. First residual Conv1D block",
                "subtitle": "Local absorption structure is encoded into 16 feature channels.",
                "explain": (
                    "Small convolution filters scan neighboring wavelength bins and "
                    "turn local spectral shapes into learned feature channels."
                ),
                "reading": (
                    "Bright rows in the heatmap are feature channels responding to "
                    "specific local patterns in the spectrum."
                ),
                "kind": "activation",
                "values": arrays[0],
                "telescope": spec.telescope,
            },
            {
                "title": "4. Downsampled Conv1D block",
                "subtitle": "The network compresses neighboring wavelengths while keeping spectral features.",
                "explain": (
                    "The sequence becomes shorter, but each position now summarizes "
                    "a wider spectral neighborhood."
                ),
                "reading": (
                    "The heatmap is shorter along x because the model has compressed "
                    "the wavelength axis into coarser learned positions."
                ),
                "kind": "activation",
                "values": arrays[1],
                "telescope": spec.telescope,
            },
            {
                "title": "5. Attention + normalization",
                "subtitle": "Self-attention lets distant wavelengths influence the same latent representation.",
                "explain": (
                    "The attention block allows non-adjacent wavelengths to interact, "
                    "so a feature in the blue/UV can be interpreted together with NIR."
                ),
                "reading": (
                    "This map is no longer just local filtering; each position can "
                    "carry context from the rest of the spectrum."
                ),
                "kind": "activation",
                "values": arrays[2],
                "telescope": spec.telescope,
            },
            {
                "title": "6. Global spectral embedding",
                "subtitle": "Global average pooling condenses the sequence into a compact latent vector.",
                "explain": (
                    "The model collapses the wavelength sequence into one vector: a "
                    "compact fingerprint of the planet spectrum."
                ),
                "reading": (
                    "Each bar is a latent unit. It is not a gas by itself, but a "
                    "learned summary used by the retrieval heads."
                ),
                "kind": "vector",
                "values": arrays[3],
                "telescope": spec.telescope,
            },
            {
                "title": "7. Dense retrieval context",
                "subtitle": "Dense layers with dropout prepare shared information for the output heads.",
                "explain": (
                    "A shared dense trunk combines the spectral fingerprint into the "
                    "representation used for all physical and chemical predictions."
                ),
                "reading": (
                    "Positive and negative bars are internal evidence patterns. "
                    "Dropout during uncertainty runs probes how stable this context is."
                ),
                "kind": "vector",
                "values": arrays[4],
                "telescope": spec.telescope,
            },
            {
                "title": "8. Output heads",
                "subtitle": "Three heads retrieve physical parameters, O2/O3, and the remaining gases.",
                "explain": (
                    "The final branches convert the shared latent context into the "
                    "10 normalized retrieval targets."
                ),
                "reading": (
                    "The output values are still normalized model-space values. The "
                    "Retrieve tab converts them back to physical units and mixing ratios."
                ),
                "kind": "output",
                "values": z_output,
                "telescope": spec.telescope,
            },
        ]

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
