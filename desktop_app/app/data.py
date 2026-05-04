import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from desktop_app.app.constants import (
    BANDS,
    PHYS_PARAMS,
    MAIN_CHEM,
    OTHER_CHEM,
    TELESCOPE_CONFIG,
)

LABEL_KEYS = list(PHYS_PARAMS + MAIN_CHEM + OTHER_CHEM)

_META_FEATURES = {
    "OBJECT-STAR-TYPE": tf.io.FixedLenFeature([], tf.string),
    "GEOMETRY-OBS-ALTITUDE": tf.io.FixedLenFeature([], tf.float32),
    "Earth_type": tf.io.FixedLenFeature([], tf.string),
    "SNR_FEATURE_PCTL_B-Vis": tf.io.FixedLenFeature([], tf.float32),
    "SNR_FEATURE_PCTL_SS-Vis": tf.io.FixedLenFeature([], tf.float32),
    **{k: tf.io.FixedLenFeature([], tf.float32) for k in LABEL_KEYS},
}


@dataclass
class SampleMeta:
    index: int
    star_type: str
    distance_pc: float
    era: str
    snr_luvoir_vis: float
    snr_habex_vis: float
    truth: dict[str, float]

    def short_label(self) -> str:
        return (
            f"#{self.index:05d}  "
            f"{self.star_type}-star  "
            f"{self.distance_pc:.1f} pc  "
            f"{self.era}"
        )


@dataclass
class Spectrum:
    telescope: str
    wavelengths: dict[str, np.ndarray]
    noisy_albedo: dict[str, np.ndarray]
    noise: dict[str, np.ndarray]
    clean_albedo: Optional[dict[str, np.ndarray]] = None

    wave_full: Optional[np.ndarray] = None
    noisy_full: Optional[np.ndarray] = None
    noise_full: Optional[np.ndarray] = None
    clean_full: Optional[np.ndarray] = None


class DataStore:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tfrecord_path = project_root / "data" / "test.tfrecord"
        self.norm_stats_path = project_root / "data" / "normalization_stats.json"
        self.uncertainty_path = (
            project_root / "data" / "bootstrap" / "combined_uncertainty.npy"
        )

        with open(self.norm_stats_path) as f:
            self.norm_stats = json.load(f)

        self._raw_records: list[bytes] = []
        self._metas: list[SampleMeta] = []
        self._uncertainty: Optional[dict] = None
        self._ig_data: Optional[dict] = None
        self._wave_full: dict[str, np.ndarray] = {}

    def load_metadata(self, progress=None) -> list[SampleMeta]:
        if self._metas:
            return self._metas

        ds = tf.data.TFRecordDataset(str(self.tfrecord_path))
        for i, raw in enumerate(ds):
            raw_b = raw.numpy()
            self._raw_records.append(raw_b)
            parsed = tf.io.parse_single_example(raw_b, _META_FEATURES)
            truth = {k: float(parsed[k].numpy()) for k in LABEL_KEYS}
            meta = SampleMeta(
                index=i,
                star_type=parsed["OBJECT-STAR-TYPE"].numpy().decode(),
                distance_pc=float(parsed["GEOMETRY-OBS-ALTITUDE"].numpy()),
                era=parsed["Earth_type"].numpy().decode(),
                snr_luvoir_vis=float(parsed["SNR_FEATURE_PCTL_B-Vis"].numpy()),
                snr_habex_vis=float(parsed["SNR_FEATURE_PCTL_SS-Vis"].numpy()),
                truth=truth,
            )
            self._metas.append(meta)
            if progress is not None and (i % 200 == 0):
                progress(i)
        if progress is not None:
            progress(len(self._metas))
        return self._metas

    def wave_full(self, telescope: str) -> np.ndarray:
        if telescope in self._wave_full:
            return self._wave_full[telescope]
        ig_key = "LUVOUIR-B" if telescope == "LUVOIR" else "HABEX-SS"
        ig = self._load_ig()
        wave = np.asarray(ig[ig_key]["wave"], dtype=float)
        self._wave_full[telescope] = wave
        return wave

    def _load_ig(self) -> dict:
        if self._ig_data is None:
            path = self.project_root / "data" / "bootstrap" / "ig_heatmaps.npy"
            self._ig_data = np.load(path, allow_pickle=True).item()
        return self._ig_data

    def ig_heatmap(self, telescope: str, era: str) -> tuple[np.ndarray, np.ndarray]:
        ig_key = "LUVOUIR-B" if telescope == "LUVOIR" else "HABEX-SS"
        ig = self._load_ig()
        wave = np.asarray(ig[ig_key]["wave"], dtype=float)
        hmap = np.asarray(ig[ig_key]["ig_heatmap"][era.lower()], dtype=float)
        return wave, hmap

    def load_spectrum(self, index: int, telescope: str) -> Spectrum:
        cfg = TELESCOPE_CONFIG[telescope]
        pfx = cfg["prefix"]

        feat = {
            **{
                f"NOISY_ALBEDO_{pfx}-{b}": tf.io.VarLenFeature(tf.float32)
                for b in BANDS
            },
            **{f"NOISE_{pfx}-{b}": tf.io.VarLenFeature(tf.float32) for b in BANDS},
            **{f"ALBEDO_{pfx}-{b}": tf.io.VarLenFeature(tf.float32) for b in BANDS},
        }
        parsed = tf.io.parse_single_example(self._raw_records[index], feat)

        def dense(key: str) -> np.ndarray:
            t = parsed[key]
            if isinstance(t, tf.SparseTensor):
                t = tf.sparse.to_dense(t, default_value=0.0)
            return t.numpy().astype(np.float32)

        noisy = {b: dense(f"NOISY_ALBEDO_{pfx}-{b}") for b in BANDS}
        noise = {b: dense(f"NOISE_{pfx}-{b}") for b in BANDS}
        clean = {b: dense(f"ALBEDO_{pfx}-{b}") for b in BANDS}

        wave_full = self.wave_full(telescope)
        n_uv = TELESCOPE_CONFIG[telescope]["bins"]["UV"]
        n_vis = TELESCOPE_CONFIG[telescope]["bins"]["Vis"]
        wls = {
            "UV": wave_full[:n_uv],
            "Vis": wave_full[n_uv : n_uv + n_vis],
            "NIR": wave_full[n_uv + n_vis :],
        }
        noisy_full = np.concatenate([noisy["UV"], noisy["Vis"], noisy["NIR"]])
        noise_full = np.concatenate([noise["UV"], noise["Vis"], noise["NIR"]])
        clean_full = np.concatenate([clean["UV"], clean["Vis"], clean["NIR"]])

        return Spectrum(
            telescope=telescope,
            wavelengths=wls,
            noisy_albedo=noisy,
            noise=noise,
            clean_albedo=clean,
            wave_full=wave_full,
            noisy_full=noisy_full,
            noise_full=noise_full,
            clean_full=clean_full,
        )

    def spectrum_from_text(self, text: str, telescope: str) -> Spectrum:
        rows = self._numeric_rows(text)
        if not rows:
            raise ValueError("Paste at least one numeric flux value.")

        total_bins = sum(TELESCOPE_CONFIG[telescope]["bins"][b] for b in BANDS)
        wave_full = self.wave_full(telescope)
        matrix = [r for r in rows if r]
        widths = {len(r) for r in matrix}

        flux: np.ndarray
        noise: np.ndarray
        clean: Optional[np.ndarray] = None

        if len(matrix) == total_bins and min(widths) >= 2:
            first = np.asarray([row[0] for row in matrix], dtype=float)
            has_wave = self._looks_like_wavelength(first, wave_full)
            if has_wave:
                flux = np.asarray([row[1] for row in matrix], dtype=float)
                if min(widths) >= 3:
                    noise = np.maximum(
                        np.asarray([row[2] for row in matrix], dtype=float), 0.0
                    )
                else:
                    noise = self._default_noise(flux)
            elif min(widths) >= 2:
                flux = np.asarray([row[0] for row in matrix], dtype=float)
                noise = np.maximum(
                    np.asarray([row[1] for row in matrix], dtype=float), 0.0
                )
            else:
                flux = first
                noise = self._default_noise(flux)
        else:
            flat = np.asarray([x for row in matrix for x in row], dtype=float)
            if flat.size == total_bins:
                flux = flat
                noise = self._default_noise(flux)
            elif flat.size == total_bins * 2:
                flux = flat[:total_bins]
                noise = np.maximum(flat[total_bins:], 0.0)
            elif flat.size == total_bins * 3:
                trip = flat.reshape(total_bins, 3)
                first = trip[:, 0]
                if self._looks_like_wavelength(first, wave_full):
                    flux = trip[:, 1]
                    noise = np.maximum(trip[:, 2], 0.0)
                else:
                    flux = trip[:, 0]
                    noise = np.maximum(trip[:, 1], 0.0)
                    clean = trip[:, 2]
            else:
                raise ValueError(
                    f"{TELESCOPE_CONFIG[telescope]['label']} expects {total_bins} "
                    f"flux bins. Received {flat.size} numeric values."
                )

        if flux.size != total_bins or noise.size != total_bins:
            raise ValueError(
                f"{TELESCOPE_CONFIG[telescope]['label']} expects {total_bins} bins."
            )
        if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(noise)):
            raise ValueError("Flux and noise values must be finite numbers.")

        flux = np.clip(flux.astype(np.float32), 0.0, None)
        noise = np.clip(noise.astype(np.float32), 0.0, None)
        if clean is None or clean.size != total_bins:
            clean = flux.copy()
        clean = np.clip(clean.astype(np.float32), 0.0, None)

        return self._spectrum_from_full_arrays(telescope, flux, noise, clean)

    def spectrum_to_text(self, spec: Spectrum) -> str:
        lines = [
            "# wavelength_um, noisy_albedo, noise_1sigma",
            f"# {TELESCOPE_CONFIG[spec.telescope]['label']} example "
            "copied from the selected test spectrum",
        ]
        for wave, flux, noise in zip(spec.wave_full, spec.noisy_full, spec.noise_full):
            lines.append(f"{wave:.8g}, {flux:.8g}, {noise:.8g}")
        return "\n".join(lines)

    def _spectrum_from_full_arrays(
        self,
        telescope: str,
        noisy_full: np.ndarray,
        noise_full: np.ndarray,
        clean_full: np.ndarray,
    ) -> Spectrum:
        cfg = TELESCOPE_CONFIG[telescope]
        n_uv = cfg["bins"]["UV"]
        n_vis = cfg["bins"]["Vis"]
        wave_full = self.wave_full(telescope)
        slices = {
            "UV": slice(0, n_uv),
            "Vis": slice(n_uv, n_uv + n_vis),
            "NIR": slice(n_uv + n_vis, None),
        }
        wls = {b: wave_full[slices[b]] for b in BANDS}
        noisy = {b: noisy_full[slices[b]].astype(np.float32) for b in BANDS}
        noise = {b: noise_full[slices[b]].astype(np.float32) for b in BANDS}
        clean = {b: clean_full[slices[b]].astype(np.float32) for b in BANDS}
        return Spectrum(
            telescope=telescope,
            wavelengths=wls,
            noisy_albedo=noisy,
            noise=noise,
            clean_albedo=clean,
            wave_full=wave_full,
            noisy_full=noisy_full.astype(np.float32),
            noise_full=noise_full.astype(np.float32),
            clean_full=clean_full.astype(np.float32),
        )

    @staticmethod
    def _numeric_rows(text: str) -> list[list[float]]:
        rows: list[list[float]] = []
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[\s,;]+", line) if p]
            try:
                rows.append([float(p) for p in parts])
            except ValueError as exc:
                raise ValueError(f"Could not parse this row: {raw_line!r}") from exc
        return rows

    @staticmethod
    def _looks_like_wavelength(values: np.ndarray, expected: np.ndarray) -> bool:
        if values.size != expected.size:
            return False
        if not np.all(np.diff(values) >= -1e-8):
            return False
        lo = expected.min() - 0.05
        hi = expected.max() + 0.05
        return bool(np.nanmin(values) >= lo and np.nanmax(values) <= hi)

    @staticmethod
    def _default_noise(flux: np.ndarray) -> np.ndarray:
        scale = np.nanmedian(np.abs(flux))
        floor = 1e-4 if not np.isfinite(scale) or scale <= 0 else 0.03 * scale
        return np.full_like(flux, floor, dtype=np.float32)

    def uncertainty(self, telescope: str, index: int) -> np.ndarray:
        if self._uncertainty is None:
            self._uncertainty = np.load(self.uncertainty_path, allow_pickle=True).item()
        key = TELESCOPE_CONFIG[telescope]["uncertainty_key"]
        return np.asarray(self._uncertainty[key]["sigma_total"][index], dtype=float)

    def reference_uncertainty(self, telescope: str) -> np.ndarray:
        if self._uncertainty is None:
            self._uncertainty = np.load(self.uncertainty_path, allow_pickle=True).item()
        key = TELESCOPE_CONFIG[telescope]["uncertainty_key"]
        sigmas = np.asarray(self._uncertainty[key]["sigma_total"], dtype=float)
        return np.nanmedian(sigmas, axis=0)
