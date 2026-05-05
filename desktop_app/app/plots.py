from typing import Optional

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from geexhp import datavis

from desktop_app.app.constants import (
    ALL_PARAMS,
    BANDS,
    MAIN_CHEM,
    OTHER_CHEM,
    PARAM_LABELS,
    PHYS_PARAMS,
    TELESCOPE_CONFIG,
)
from desktop_app.app.data import Spectrum
from desktop_app.app import theme

datavis.configure_matplotlib()

TEL_COLORS = {"LUVOIR": theme.TEL_LUVOIR, "HABEX": theme.TEL_HABEX}

CORNER_LABELS_FULL = [
    r"$R_\oplus$",
    r"$g$ [m s$^{-2}$]",
    r"$T_{\rm surf}$ [K]",
    r"$P_{\rm surf}$ [mbar]",
    r"$\log_{10}$(O$_2$)",
    r"$\log_{10}$(O$_3$)",
    r"$\log_{10}$(CH$_4$)",
    r"$\log_{10}$(CO$_2$)",
    r"$\log_{10}$(H$_2$O)",
    r"$\log_{10}$(N$_2$)",
]

plt.rcParams.update(
    {
        "axes.edgecolor": "#cbd2dd",
        "axes.labelcolor": theme.INK,
        "axes.titlecolor": theme.NAVY_700,
        "axes.titlepad": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.color": theme.INK_DIM,
        "ytick.color": theme.INK_DIM,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.framealpha": 0.0,
        "savefig.facecolor": theme.SURFACE,
        "figure.facecolor": theme.SURFACE,
        "axes.facecolor": theme.SURFACE,
    }
)


class _CanvasBase(QWidget):
    def __init__(self, figsize=(9, 4), parent=None, show_toolbar: bool = True):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, layout="tight", facecolor=theme.SURFACE)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet(f"background:{theme.SURFACE};")
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setMouseTracking(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        if show_toolbar:
            self.toolbar = NavToolbar(self.canvas, self)
            self.toolbar.setIconSize(QSize(16, 16))
            self.toolbar.setMovable(False)
            self.toolbar.setFloatable(False)
            self.toolbar.setStyleSheet(
                f"""
                QToolBar {{
                    background: {theme.SURFACE_2};
                    border: 1px solid {theme.BORDER};
                    border-radius: 6px;
                    spacing: 3px;
                    padding: 3px 6px;
                }}
                QToolButton {{
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 4px;
                    padding: 3px;
                }}
                QToolButton:hover {{
                    background: #edf4fb;
                    border-color: {theme.BORDER_2};
                }}
                QToolButton:checked {{
                    background: #e5f5f3;
                    border-color: {theme.CYAN};
                }}
                QLabel {{
                    color: {theme.INK_FAINT};
                    font-size: 11px;
                }}
                """
            )
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self) -> None:
        self.fig.clear()
        self.canvas.draw_idle()


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_color(theme.BORDER)
    ax.spines["right"].set_color(theme.BORDER)
    ax.tick_params(top=False, right=False)


BAND_TINTS = {
    "UV": ("#7e6dbf", 0.045),
    "Vis": ("#3fbfb0", 0.045),
    "NIR": ("#e8b657", 0.055),
}


class SpectrumCanvas(_CanvasBase):
    def __init__(self, parent=None):
        super().__init__(figsize=(11, 4.2), parent=parent)

    def show_spectrum(self, spec: Spectrum, title_suffix: str = "") -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        color = TEL_COLORS.get(spec.telescope, theme.TEL_LUVOIR)
        tel_label = TELESCOPE_CONFIG[spec.telescope]["label"]

        for band in BANDS:
            wl = spec.wavelengths[band]
            tint, alpha = BAND_TINTS[band]
            ax.axvspan(wl.min(), wl.max(), color=tint, alpha=alpha, zorder=0)

        wave = spec.wave_full
        noisy = spec.noisy_full
        err = spec.noise_full
        _, caps, bars = ax.errorbar(
            wave,
            noisy,
            yerr=err,
            fmt=".",
            capsize=0,
            lw=0.8,
            color="#9aa4b6",
            alpha=0.45,
            markersize=2.6,
            zorder=2,
            label="noisy",
        )
        for b in bars:
            b.set_alpha(0.25)
        for c in caps:
            c.set_alpha(0.25)

        if spec.clean_full is not None:
            ax.plot(
                wave,
                spec.clean_full,
                color=color,
                lw=1.6,
                zorder=3,
                label="true albedo",
            )

        self.fig.canvas.draw_idle()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.08)
        for band in BANDS:
            wl = spec.wavelengths[band]
            ax.text(
                0.5 * (wl.min() + wl.max()),
                ymax * 1.05,
                band,
                ha="center",
                va="top",
                fontsize=9.5,
                color=BAND_TINTS[band][0],
                weight="bold",
                alpha=0.95,
            )

        ax.set_xlabel(r"Wavelength  [$\mu$m]")
        ax.set_ylabel("Apparent albedo")
        ax.set_xlim(wave.min(), wave.max())
        leg = ax.legend(loc="upper right", fontsize=10)
        for t in leg.get_texts():
            t.set_color(theme.INK_DIM)
        title = f"{tel_label}  reflected-light spectrum"
        if title_suffix:
            title += f"   ·   {title_suffix}"
        ax.set_title(title, loc="left")
        self.canvas.draw_idle()


class RetrievalCanvas(_CanvasBase):
    def __init__(self, parent=None):
        super().__init__(figsize=(11, 6), parent=parent)

    def show_retrieval(
        self,
        telescope: str,
        predicted: dict[str, float],
        sigma_phys: dict[str, float],
        truth: Optional[dict[str, float]] = None,
        hide: tuple[str, ...] = (),
    ) -> None:
        self.fig.clear()
        chem_names_full = list(MAIN_CHEM) + list(OTHER_CHEM)
        chem_names = [n for n in chem_names_full if n not in hide]
        n_chem = len(chem_names)
        n_cols = max(n_chem, 6)

        gs = self.fig.add_gridspec(
            2,
            n_cols,
            hspace=0.65,
            wspace=0.55,
            left=0.07,
            right=0.98,
            top=0.88,
            bottom=0.08,
        )
        color = TEL_COLORS.get(telescope, theme.TEL_LUVOIR)
        truth_color = theme.NAVY_700
        tel_label = TELESCOPE_CONFIG[telescope]["label"]

        phys_offset = max((n_cols - len(PHYS_PARAMS)) // 2, 0)
        for j, name in enumerate(PHYS_PARAMS):
            ax = self.fig.add_subplot(gs[0, phys_offset + j])
            mu = predicted[name]
            sd = sigma_phys.get(name, np.nan)
            ax.errorbar(
                [0.5],
                [mu],
                yerr=[sd],
                fmt="o",
                color=color,
                capsize=4,
                lw=1.6,
                markersize=8,
                label="predicted",
                markeredgecolor="white",
                markeredgewidth=1.0,
            )
            if truth is not None and name in truth:
                ax.scatter(
                    [0.5],
                    [truth[name]],
                    marker="D",
                    s=46,
                    color=truth_color,
                    zorder=4,
                    label="truth",
                    edgecolor="white",
                    linewidths=1.0,
                )
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_title(PARAM_LABELS[j])
            ax.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            if j == 0:
                leg = ax.legend(
                    loc="lower center",
                    bbox_to_anchor=(2.2, 1.16),
                    fontsize=10,
                    ncol=2,
                    frameon=False,
                )
                for t in leg.get_texts():
                    t.set_color(theme.INK_DIM)

        chem_offset = max((n_cols - n_chem) // 2, 0)
        for j, name in enumerate(chem_names):
            ax = self.fig.add_subplot(gs[1, chem_offset + j])
            mu = max(predicted[name], 1e-30)
            sd = sigma_phys.get(name, 0.0)
            mu_log = np.log10(mu)
            lo = max(mu - sd, 1e-30)
            hi = mu + sd
            err_low = mu_log - np.log10(lo)
            err_high = np.log10(hi) - mu_log
            ax.errorbar(
                [0.5],
                [mu_log],
                yerr=[[err_low], [err_high]],
                fmt="o",
                color=color,
                capsize=4,
                lw=1.6,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )
            if truth is not None and name in truth and truth[name] > 0:
                ax.scatter(
                    [0.5],
                    [np.log10(truth[name])],
                    marker="D",
                    s=46,
                    color=truth_color,
                    zorder=4,
                    edgecolor="white",
                    linewidths=1.0,
                )

            full_idx = chem_names_full.index(name)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_title(PARAM_LABELS[4 + full_idx])
            ax.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.7, which="major")
            ax.set_axisbelow(True)
            ax.set_ylim(-12, 0.5)
            ax.set_yticks([-12, -9, -6, -3, 0])

        self.fig.text(
            0.015,
            0.70,
            "PHYSICAL",
            rotation=90,
            va="center",
            fontsize=9,
            color=theme.NAVY_700,
            weight="bold",
        )
        self.fig.text(
            0.015,
            0.27,
            "log₁₀  vmr",
            rotation=90,
            va="center",
            fontsize=9,
            color=theme.NAVY_700,
            weight="bold",
        )
        self.fig.suptitle(
            f"Retrieval  ·  {tel_label}",
            x=0.07,
            y=0.96,
            ha="left",
            fontsize=12.5,
            color=theme.NAVY_700,
            weight="bold",
        )
        self.canvas.draw_idle()


class IGCanvas(_CanvasBase):
    CHEMS = ("O2", "O3", "CH4", "CO2", "H2O", "N2")

    def __init__(self, parent=None):
        super().__init__(figsize=(11, 6.5), parent=parent)

    def show_heatmap(
        self,
        wave: np.ndarray,
        heatmap: np.ndarray,
        telescope: str,
        era: str,
        spectrum: Optional[Spectrum] = None,
    ) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(
            2,
            2,
            height_ratios=[3, 1],
            width_ratios=[40, 1],
            hspace=0.10,
            wspace=0.04,
            left=0.09,
            right=0.95,
            top=0.90,
            bottom=0.12,
        )
        ax = self.fig.add_subplot(gs[0, 0])
        cax = self.fig.add_subplot(gs[0, 1])
        ax_s = self.fig.add_subplot(gs[1, 0], sharex=ax)

        vmax = float(np.nanpercentile(np.abs(heatmap), 99))
        vmax = vmax if vmax > 0 else 1e-3
        im = ax.pcolormesh(
            wave,
            np.arange(len(self.CHEMS)),
            heatmap,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="nearest",
        )
        chem_labels = [r"O$_2$", r"O$_3$", r"CH$_4$", r"CO$_2$", r"H$_2$O", r"N$_2$"]
        ax.set_yticks(np.arange(len(self.CHEMS)))
        ax.set_yticklabels(chem_labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_ylabel("Chemical species")
        ax.tick_params(axis="x", labelbottom=False)
        ax.spines[:].set_visible(False)
        cbar = self.fig.colorbar(im, cax=cax)
        cbar.set_label("Integrated gradient", fontsize=10, color=theme.INK_DIM)
        cbar.ax.tick_params(labelsize=9, colors=theme.INK_DIM)
        cbar.outline.set_visible(False)

        tel_label = TELESCOPE_CONFIG[telescope]["label"]
        ax.set_title(
            f"Wavelength sensitivity  ·  {tel_label}   ·   {era.capitalize()} Earth",
            loc="left",
            fontsize=12,
            color=theme.NAVY_700,
            weight="bold",
        )

        if spectrum is not None and spectrum.clean_full is not None:
            ax_s.fill_between(
                spectrum.wave_full,
                0,
                spectrum.clean_full,
                color=TEL_COLORS[telescope],
                alpha=0.18,
            )
            ax_s.plot(
                spectrum.wave_full,
                spectrum.clean_full,
                color=TEL_COLORS[telescope],
                lw=1.4,
            )
        ax_s.set_xlabel(r"Wavelength  [$\mu$m]")
        ax_s.set_ylabel("Albedo", fontsize=10)
        ax_s.set_xlim(wave.min(), wave.max())
        ax_s.tick_params(labelsize=9)
        ax_s.grid(axis="y", color=theme.BORDER, lw=0.5, alpha=0.6)
        ax_s.set_axisbelow(True)

        self.canvas.draw_idle()


class NetworkCanvas(_CanvasBase):
    FLOW_LABELS = (
        "Spectrum",
        "Norm",
        "Conv",
        "Downsample",
        "Attention",
        "Pool",
        "Dense",
        "Output",
    )

    OUTPUT_LABELS = (
        r"$R_\oplus$",
        r"$g$",
        r"$T$",
        r"$P$",
        r"O$_2$",
        r"O$_3$",
        r"CH$_4$",
        r"CO$_2$",
        r"H$_2$O",
        r"N$_2$",
    )

    def __init__(self, parent=None):
        super().__init__(figsize=(11.5, 6.6), parent=parent)
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(
            0.5,
            0.58,
            "Run a network walkthrough to see the selected spectrum move through the CNN.",
            ha="center",
            va="center",
            color=theme.INK_DIM,
            fontsize=13,
        )
        ax.text(
            0.5,
            0.48,
            "The walkthrough uses real intermediate tensors from the saved Keras model.",
            ha="center",
            va="center",
            color=theme.INK_FAINT,
            fontsize=10.5,
        )
        self.canvas.draw_idle()

    def show_stage(
        self,
        stages: list[dict],
        active_index: int,
        hide: tuple[str, ...] = (),
        truth_z: Optional[np.ndarray] = None,
    ) -> None:
        if not stages:
            self.show_placeholder()
            return

        stage = stages[active_index]
        tel = stage.get("telescope", "LUVOIR")
        color = TEL_COLORS.get(tel, theme.TEL_LUVOIR)
        tel_label = TELESCOPE_CONFIG[tel]["label"]

        self.fig.clear()
        ax_detail = self.fig.add_subplot(1, 1, 1)
        self._draw_detail(ax_detail, stage, color, hide, truth_z=truth_z)

        self.fig.suptitle(
            f"Network walkthrough  ·  {tel_label}",
            x=0.07,
            y=0.985,
            ha="left",
            fontsize=12.5,
            color=theme.NAVY_700,
            weight="bold",
        )
        self.fig.text(
            0.07,
            0.905,
            stage["subtitle"],
            ha="left",
            va="center",
            fontsize=10,
            color=theme.INK_DIM,
        )
        self.canvas.draw_idle()

    def _draw_flow(self, ax, active_index: int, color: str) -> None:
        ax.axis("off")
        xs = np.linspace(0.05, 0.95, len(self.FLOW_LABELS))
        y = 0.52
        for i, label in enumerate(self.FLOW_LABELS):
            active = i == active_index
            done = i < active_index
            face = color if active or done else theme.SURFACE_2
            edge = color if active or done else theme.BORDER_2
            txt = "white" if active else theme.NAVY_700
            ax.scatter(
                xs[i],
                y,
                s=760 if active else 570,
                color=face,
                alpha=0.24 if done and not active else 1.0,
                edgecolor=edge,
                linewidth=1.7,
                zorder=3,
                transform=ax.transAxes,
            )
            ax.text(
                xs[i],
                y,
                str(i + 1),
                ha="center",
                va="center",
                fontsize=10,
                color=txt,
                weight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                xs[i],
                0.13,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=theme.INK_DIM,
                transform=ax.transAxes,
            )
            if i < len(xs) - 1:
                ax.annotate(
                    "",
                    xy=(xs[i + 1] - 0.035, y),
                    xytext=(xs[i] + 0.035, y),
                    xycoords=ax.transAxes,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color if i < active_index else theme.BORDER_2,
                        lw=1.25,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )

    def _draw_detail(
        self,
        ax,
        stage: dict,
        color: str,
        hide: tuple[str, ...],
        truth_z: Optional[np.ndarray] = None,
    ) -> None:
        kind = stage["kind"]
        values = np.asarray(stage["values"], dtype=float)
        ax.set_title(
            stage["title"],
            loc="left",
            fontsize=12,
            color=theme.NAVY_700,
            weight="bold",
        )
        if kind in {"spectrum", "line"}:
            wave = np.asarray(stage.get("wave"), dtype=float)
            ax.plot(wave, values, color=color, lw=1.7)
            if kind == "spectrum":
                ax.fill_between(wave, 0, values, color=color, alpha=0.16)
                ax.set_ylabel("Apparent albedo")
            else:
                ax.axhline(0.0, color=theme.INK_FAINT, lw=0.9, ls="--", alpha=0.75)
                ax.set_ylabel("Standardized albedo")
            ax.set_xlabel(r"Wavelength  [$\mu$m]")
            ax.set_xlim(wave.min(), wave.max())
            ax.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            _clean_axes(ax)
            return

        if kind == "activation":
            data = values.T
            vmax = float(np.nanpercentile(np.abs(data), 99))
            vmax = vmax if vmax > 0 else 1.0
            im = ax.imshow(
                data,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
                interpolation="nearest",
            )
            tel = stage.get("telescope", "LUVOIR")
            ranges = TELESCOPE_CONFIG.get(tel, {}).get("ranges_um", {})
            if ranges:
                wmin = ranges["UV"][0]
                wmax = ranges["NIR"][1]
                nx = data.shape[1]
                tick_pos = np.linspace(0, nx - 1, 5)
                tick_lbl = [f"{w:.2f}" for w in np.linspace(wmin, wmax, 5)]
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(tick_lbl)
                ax.set_xlabel(r"Wavelength position [$\mu$m, approx.]")
            else:
                ax.set_xlabel("Compressed spectral position")
            ax.set_ylabel("Feature channel")
            ax.grid(False)
            cbar = self.fig.colorbar(im, ax=ax, pad=0.012, fraction=0.032)
            cbar.set_label("Activation", fontsize=9, color=theme.INK_DIM)
            cbar.ax.tick_params(labelsize=8, colors=theme.INK_DIM)
            cbar.outline.set_visible(False)
            mean_abs = np.mean(np.abs(values), axis=1)
            if np.nanmax(mean_abs) > 0:
                y_line = (mean_abs / np.nanmax(mean_abs)) * (data.shape[0] - 1)
                ax.plot(
                    np.arange(len(y_line)),
                    y_line,
                    color=theme.GOLD_DARK,
                    lw=1.45,
                    alpha=0.95,
                )
                ax.text(
                    0.985,
                    1.045,
                    "mean |activation|",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8.5,
                    color=theme.GOLD_DARK,
                    weight="bold",
                    clip_on=False,
                )
            _clean_axes(ax)
            return

        if kind == "vector":
            x = np.arange(values.size)
            colors = [color if v >= 0 else theme.ROSE for v in values]
            ax.bar(x, values, color=colors, alpha=0.86, width=0.82)
            ax.axhline(0.0, color=theme.INK_FAINT, lw=0.9)
            ax.set_xlabel("Latent unit")
            ax.set_ylabel("Activation")
            ax.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            if values.size > 24:
                ax.set_xticks(np.arange(0, values.size, 4))
            _clean_axes(ax)
            return

        keep = [i for i, name in enumerate(ALL_PARAMS) if name not in hide]
        labels = [self.OUTPUT_LABELS[i] for i in keep]
        shown = values[keep]
        x = np.arange(len(keep))
        out_colors = []
        for idx in keep:
            if idx < len(PHYS_PARAMS):
                out_colors.append(color)
            elif idx < len(PHYS_PARAMS) + len(MAIN_CHEM):
                out_colors.append(theme.GOLD_DARK)
            else:
                out_colors.append(theme.SKY)
        ax.bar(x, shown, color=out_colors, alpha=0.88, width=0.72, label="_nolegend_")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Normalized model output")

        ymax = max(1.05, float(np.nanmax(shown)) * 1.18)
        if truth_z is not None and truth_z.size == len(ALL_PARAMS):
            truth_show = truth_z[keep]
            for xi, t in zip(x, truth_show):
                if not np.isfinite(t):
                    continue
                ax.hlines(
                    t,
                    xi - 0.36,
                    xi + 0.36,
                    colors=theme.NAVY_700,
                    lw=1.8,
                    zorder=5,
                )
                ymax = max(ymax, float(t) * 1.08)
            ax.plot(
                [], [], color=theme.NAVY_700, lw=1.8, label="Truth (normalized)"
            )
            ax.legend(loc="upper right", fontsize=8.5, frameon=False)

        phys_end = sum(1 for i in keep if i < len(PHYS_PARAMS))
        main_end = sum(
            1 for i in keep if i < len(PHYS_PARAMS) + len(MAIN_CHEM)
        )
        for boundary in (phys_end, main_end):
            if 0 < boundary < len(keep):
                ax.axvline(
                    boundary - 0.5,
                    color=theme.BORDER_2,
                    lw=0.9,
                    ls=":",
                    alpha=0.85,
                    zorder=1,
                )

        ax.set_ylim(0.0, ymax)
        ax.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        _clean_axes(ax)


class CompareCanvas(_CanvasBase):
    def __init__(self, parent=None):
        super().__init__(figsize=(11, 4.5), parent=parent)

    def show_compare(
        self,
        results: dict[str, tuple[dict[str, float], dict[str, float]]],
        truth: Optional[dict[str, float]] = None,
        hide: tuple[str, ...] = (),
    ) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(
            2,
            4,
            height_ratios=[1.0, 1.25],
            hspace=0.58,
            wspace=0.36,
            left=0.06,
            right=0.98,
            top=0.86,
            bottom=0.12,
        )
        phys_axes = [self.fig.add_subplot(gs[0, i]) for i in range(4)]
        ax_chem = self.fig.add_subplot(gs[1, :])
        offsets = {"LUVOIR": -0.12, "HABEX": 0.12}

        phys = list(PHYS_PARAMS)
        chem_all = list(MAIN_CHEM) + list(OTHER_CHEM)
        chem = [name for name in chem_all if name not in hide]
        chem_labels = [
            PARAM_LABELS[len(PHYS_PARAMS) + chem_all.index(name)] for name in chem
        ]

        x_chem = np.arange(len(chem))

        for j, name in enumerate(phys):
            ax = phys_axes[j]
            vals_for_range = []
            for tel, (pred, sigma) in results.items():
                mu = float(pred[name])
                sd = float(sigma.get(name, 0.0))
                vals_for_range.extend([mu - sd, mu + sd])
                ax.errorbar(
                    [0.5 + offsets[tel]],
                    [mu],
                    yerr=[sd],
                    fmt="o",
                    capsize=3,
                    lw=1.5,
                    markersize=7,
                    color=TEL_COLORS[tel],
                    label=TELESCOPE_CONFIG[tel]["label"],
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                )
            if truth is not None and name in truth:
                vals_for_range.append(float(truth[name]))
                ax.scatter(
                    [0.5],
                    [truth[name]],
                    marker="D",
                    s=44,
                    color=theme.NAVY_700,
                    zorder=5,
                    label="truth",
                    edgecolor="white",
                    linewidths=1.0,
                )
            if vals_for_range:
                lo, hi = np.nanmin(vals_for_range), np.nanmax(vals_for_range)
                pad = 0.14 * (hi - lo if hi > lo else max(abs(hi), 1.0))
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(0.14, 0.86)
            ax.set_xticks([])
            ax.set_title(PARAM_LABELS[j], fontsize=11, color=theme.NAVY_700, weight="bold")
            ax.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            _clean_axes(ax)

        for tel, (pred, sigma) in results.items():
            c_mu = np.array([pred[k] for k in chem], dtype=float)
            c_sd = np.array([sigma.get(k, 0.0) for k in chem], dtype=float)
            c_log = np.full_like(c_mu, np.nan, dtype=float)
            c_err_low = np.zeros_like(c_mu, dtype=float)
            c_err_high = np.zeros_like(c_mu, dtype=float)
            ok = c_mu > 0
            c_log[ok] = np.log10(c_mu[ok])
            lo = np.maximum(c_mu - c_sd, 1e-30)
            hi = np.maximum(c_mu + c_sd, 1e-30)
            c_err_low[ok] = c_log[ok] - np.log10(lo[ok])
            c_err_high[ok] = np.log10(hi[ok]) - c_log[ok]
            ax_chem.errorbar(
                x_chem + offsets[tel],
                c_log,
                yerr=[c_err_low, c_err_high],
                fmt="o",
                capsize=3,
                lw=1.6,
                markersize=7,
                color=TEL_COLORS[tel],
                label=TELESCOPE_CONFIG[tel]["label"],
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

        if truth is not None:
            t_chem = np.array([truth.get(k, np.nan) for k in chem], dtype=float)
            t_chem_log = np.full_like(t_chem, np.nan, dtype=float)
            ok_truth = t_chem > 0
            t_chem_log[ok_truth] = np.log10(t_chem[ok_truth])
            ax_chem.scatter(
                x_chem,
                t_chem_log,
                marker="D",
                s=46,
                color=theme.NAVY_700,
                zorder=5,
                label="truth",
                edgecolor="white",
                linewidths=1.0,
            )

        ax_chem.set_xticks(x_chem)
        ax_chem.set_xticklabels(chem_labels, fontsize=11)
        ax_chem.set_ylabel(r"$\log_{10}$ volume mixing ratio")
        ax_chem.set_title("Chemical abundances", loc="left", fontsize=12, color=theme.NAVY_700, weight="bold")
        ax_chem.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.6)
        ax_chem.set_ylim(-12.5, 0.5)
        ax_chem.set_yticks([-12, -9, -6, -3, 0])

        ax_chem.set_axisbelow(True)
        _clean_axes(ax_chem)

        self.fig.suptitle(
            "LUVOIR  vs  HabEx  ·  physical scales separated, chemistry in log-space",
            x=0.055,
            y=0.98,
            ha="left",
            fontsize=12.5,
            color=theme.NAVY_700,
            weight="bold",
        )
        leg = ax_chem.legend(loc="best", fontsize=10)
        for t in leg.get_texts():
            t.set_color(theme.INK_DIM)
        self.canvas.draw_idle()


class CornerCanvas(_CanvasBase):
    def __init__(self, parent=None):
        super().__init__(figsize=(11, 8.2), parent=parent)

    def show_placeholder(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "Paste a custom spectrum and run retrieval to generate a corner plot.",
            ha="center",
            va="center",
            color=theme.INK_DIM,
            fontsize=13,
        )
        self.canvas.draw_idle()

    def show_corner(
        self,
        bootstrap: np.ndarray,
        mc_dropout: np.ndarray,
        spec: Spectrum,
        hide: tuple[str, ...] = (),
    ) -> None:
        self.fig.clear()
        keep = [i for i, name in enumerate(ALL_PARAMS) if name not in hide]
        labels = [CORNER_LABELS_FULL[i] for i in keep]
        bs = self._finite_rows(bootstrap[:, keep])
        mc = self._finite_rows(mc_dropout[:, keep])

        if bs.size == 0 or mc.size == 0:
            self.show_placeholder()
            return

        try:
            self._show_corner_notebook_style(bs, mc, spec, labels)
            return
        except Exception:
            self.fig.clear()

        k = len(keep)
        gs = self.fig.add_gridspec(
            k,
            k,
            left=0.08,
            right=0.97,
            bottom=0.08,
            top=0.92,
            wspace=0.06,
            hspace=0.06,
        )
        axes = np.empty((k, k), dtype=object)
        c_bs = TEL_COLORS.get(spec.telescope, theme.TEL_LUVOIR)
        c_mc = "#5f6672"

        ranges = []
        for j in range(k):
            col = np.concatenate([bs[:, j], mc[:, j]])
            lo, hi = np.nanpercentile(col, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = np.nanmin(col), np.nanmax(col)
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            ranges.append((lo - pad, hi + pad))

        for row in range(k):
            for col in range(k):
                ax = self.fig.add_subplot(gs[row, col])
                axes[row, col] = ax
                if row < col:
                    ax.axis("off")
                    continue

                if row == col:
                    ax.hist(
                        bs[:, col],
                        bins=28,
                        density=True,
                        color=c_bs,
                        alpha=0.34,
                        histtype="stepfilled",
                        linewidth=1.0,
                    )
                    ax.hist(
                        mc[:, col],
                        bins=28,
                        density=True,
                        color=c_mc,
                        histtype="step",
                        linestyle="--",
                        linewidth=1.2,
                    )
                    q16, q50, q84 = np.nanpercentile(bs[:, col], [16, 50, 84])
                    ax.set_title(
                        f"{q50:.3g} (+{q84 - q50:.2g}/-{q50 - q16:.2g})",
                        fontsize=7.5,
                        color=theme.NAVY_700,
                        loc="left",
                        pad=2,
                    )
                else:
                    ax.scatter(
                        bs[:, col],
                        bs[:, row],
                        s=4,
                        color=c_bs,
                        alpha=0.18,
                        linewidths=0,
                    )
                    ax.scatter(
                        mc[:, col],
                        mc[:, row],
                        s=4,
                        color=c_mc,
                        alpha=0.10,
                        linewidths=0,
                    )

                ax.set_xlim(ranges[col])
                if row != col:
                    ax.set_ylim(ranges[row])
                ax.grid(color=theme.BORDER, lw=0.4, alpha=0.5)
                ax.tick_params(axis="both", labelsize=6, pad=1)

                if row == k - 1:
                    ax.set_xlabel(labels[col], fontsize=7.5)
                else:
                    ax.set_xticklabels([])
                if col == 0 and row > 0:
                    ax.set_ylabel(labels[row], fontsize=7.5)
                else:
                    ax.set_yticklabels([])
                _clean_axes(ax)

        tel_label = TELESCOPE_CONFIG[spec.telescope]["label"]
        self.fig.suptitle(
            f"Corner plot  ·  {tel_label} custom spectrum",
            x=0.08,
            y=0.985,
            ha="left",
            fontsize=12.5,
            color=theme.NAVY_700,
            weight="bold",
        )

        ax_in = self.fig.add_axes([0.68, 0.70, 0.27, 0.20])
        rng = np.random.default_rng(222)
        for _ in range(18):
            eps = rng.standard_normal(len(spec.wave_full)).astype(np.float32)
            ax_in.plot(
                spec.wave_full,
                spec.noisy_full + eps * spec.noise_full,
                color=c_bs,
                alpha=0.15,
                lw=0.7,
            )
        ax_in.plot(spec.wave_full, spec.noisy_full, color=c_bs, lw=1.7)
        ax_in.set_xlabel(r"Wavelength [$\mu$m]", fontsize=7)
        ax_in.set_ylabel("Apparent albedo", fontsize=7)
        ax_in.tick_params(axis="both", labelsize=6, pad=1)
        ax_in.grid(color=theme.BORDER, lw=0.4, alpha=0.5)
        _clean_axes(ax_in)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=c_bs, edgecolor=c_bs, alpha=0.45, label="Bootstrap"),
            Line2D([0], [0], color=c_mc, lw=1.4, ls="--", label="MC Dropout"),
        ]
        self.fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.985),
            frameon=False,
            fontsize=9,
        )
        self.canvas.draw_idle()

    def _show_corner_notebook_style(
        self,
        bootstrap: np.ndarray,
        mc_dropout: np.ndarray,
        spec: Spectrum,
        labels: list[str],
    ) -> None:
        import corner
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        self.fig.clear()
        c_bs = "#B5D31F" if spec.telescope == "LUVOIR" else TEL_COLORS[spec.telescope]
        c_mc = "#6B6B6B"
        levels = (0.393, 0.865)
        k = bootstrap.shape[1]

        corner.corner(
            bootstrap,
            labels=labels,
            color=c_bs,
            fill_contours=True,
            plot_density=False,
            plot_datapoints=False,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 8.5},
            label_kwargs={"fontsize": 8.5},
            labelpad=0.18,
            max_n_ticks=2,
            hist_kwargs={"linewidth": 1.3},
            contourf_kwargs={"alpha": 0.30},
            contour_kwargs={"linewidths": 1.1},
            levels=levels,
            fig=self.fig,
        )
        corner.corner(
            mc_dropout,
            fig=self.fig,
            color=c_mc,
            fill_contours=False,
            plot_density=False,
            plot_datapoints=False,
            max_n_ticks=2,
            hist_kwargs={"linewidth": 1.5, "linestyle": "--"},
            contour_kwargs={"linewidths": 1.4, "linestyles": "--"},
            levels=levels,
        )

        axes = np.array(self.fig.axes[: k * k]).reshape((k, k))
        for row in range(k):
            for col in range(k):
                ax = axes[row, col]
                if row < col:
                    ax.axis("off")
                    continue
                ax.tick_params(axis="both", which="major", labelsize=6.5, pad=1)
                _clean_axes(ax)
                if row == col:
                    ax.title.set_ha("left")
                    ax.title.set_x(0.02)
                    ax.title.set_color(theme.NAVY_700)

        self.fig.subplots_adjust(
            left=0.08,
            right=0.97,
            bottom=0.08,
            top=0.92,
            wspace=0.05,
            hspace=0.05,
        )

        tel_label = TELESCOPE_CONFIG[spec.telescope]["label"]
        self.fig.suptitle(
            f"Corner plot  ·  {tel_label} custom spectrum",
            x=0.08,
            y=0.985,
            ha="left",
            fontsize=12.5,
            color=theme.NAVY_700,
            weight="bold",
        )

        ax_in = self.fig.add_axes([0.68, 0.73, 0.27, 0.18])
        rng = np.random.default_rng(222)
        for _ in range(24):
            eps = rng.standard_normal(len(spec.wave_full)).astype(np.float32)
            ax_in.plot(
                spec.wave_full,
                spec.noisy_full + eps * spec.noise_full,
                color=c_bs,
                alpha=0.18,
                lw=0.75,
            )
        ax_in.plot(
            spec.wave_full,
            spec.noisy_full,
            color=c_bs,
            lw=1.9,
            label=tel_label,
        )
        ax_in.set_xlabel(r"Wavelength [$\mu$m]", fontsize=7)
        ax_in.set_ylabel("Apparent albedo", fontsize=7)
        ax_in.tick_params(axis="both", which="major", labelsize=6, pad=1)
        ax_in.grid(color=theme.BORDER, lw=0.4, alpha=0.5)
        _clean_axes(ax_in)

        handles = [
            Patch(
                facecolor=c_bs,
                alpha=0.55,
                edgecolor=c_bs,
                label=r"Bootstrap  ($\sigma_{\rm data}$)",
            ),
            Line2D(
                [0],
                [0],
                color=c_mc,
                lw=1.5,
                ls="--",
                label=r"MC Dropout  ($\sigma_{\rm model}$)",
            ),
        ]
        self.fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.985),
            frameon=False,
            fontsize=8.5,
        )
        self.canvas.draw_idle()

    @staticmethod
    def _finite_rows(samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)
        if samples.size == 0:
            return samples
        out = samples.copy()
        for col in range(out.shape[1]):
            vals = out[:, col]
            finite = np.isfinite(vals)
            if not np.any(finite):
                out[:, col] = 0.0
            elif not np.all(finite):
                out[~finite, col] = np.nanmedian(vals[finite])
        return out
