import numpy as np
import pandas as pd
from math import atan2, degrees

import matplotlib.pyplot as plt
#from matplotlib_inline import backend_inline
from matplotlib import lines as mlines

from typing import List, Optional, Union

def configure_matplotlib(oldschool: bool = False) -> None:
    """
    Configures matplotlib parameters.
    """
    #backend_inline.set_matplotlib_formats("svg") 

    if oldschool:
        import smplotlib
    else:
        plt.rcParams.update({
            "xtick.top": True,          
            "ytick.right": True,        
            "xtick.direction": "in",    
            "ytick.direction": "in",    
            "font.size": 12,            
            "font.family": "Lato",      
            "axes.labelsize": 12,  
            "axes.titlesize": 12,  
            "legend.fontsize": 10,  
            "xtick.labelsize": 10,  
            "ytick.labelsize": 10,  
            "xtick.minor.visible": True,  
            "ytick.minor.visible": True  
        })

def plot_spectrum(df: pd.DataFrame, label: Optional[str] = None, index: Optional[int] = None,
                instruments: Optional[Union[str, List[str]]] = None, ax: Optional[plt.Axes] = None,
                noise: bool = False, **kwargs) -> List[plt.Axes]:
    """
    Plots the albedo spectrum of an exoplanet for specified instruments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spectrum data.
    label : str, optional
        Base label for the plot legend. If None, no label will be added.
    index : int, optional
        Index of the planet in the DataFrame. If None, assumes the DataFrame directly contains
        the spectrum data without needing to specify an index.
    instruments : str or list of str, optional
        Instrument(s) to plot. If None, plots HWC data on one plot and combines SS instruments
        on a separate plot.
    ax : matplotlib.axes.Axes or list of Axes, optional
        Axes object(s) to plot on. If None, new figures and axes will be created.
    noise : bool, optional
        If True, will also plot the noisy data with error bars. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to the plot function.

    Returns
    -------
    ax : list of matplotlib.axes.Axes
        List of Axes where the spectra are plotted.
    """
    if instruments is not None and isinstance(instruments, str):
        instruments = [instruments]

    if instruments is None:
        # Default behavior: plot HWC data on one plot and combine SS instruments on another plot
        instruments_hwc = ["HWC"]
        instruments_ss = ["SS-UV", "SS-Vis", "SS-NIR"]

        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        _plot_instruments(df, label, index, instruments_hwc, axes[0], noise, **kwargs)
        axes[0].set_title("The HabEx Workforce Camera (HWC)")

        _plot_instruments(df, label, index, instruments_ss, axes[1], noise, **kwargs)
        axes[1].set_title("Combined The HabEx StarShade (SS)")

        plt.tight_layout()
        return axes
    else:
        if ax is None:
            _, ax = plt.subplots()

        _plot_instruments(df, label, index, instruments, ax, noise, **kwargs)
        return [ax]

def _plot_instruments(df, label, index, instruments, ax, noise, **kwargs):
    """
    Helper function to plot spectra for specified instruments on a given Axes.
    """
    if index is not None:
        df_row = df.iloc[index]
    else:
        df_row = df

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {instr: color_cycle[i % len(color_cycle)] for i, instr in enumerate(instruments)}

    for instrument in instruments:
        wavelength_col = f"WAVELENGTH_{instrument}"
        albedo_col = f"ALBEDO_{instrument}"
        noisy_albedo_col = f"NOISY_ALBEDO_{instrument}"
        noise_col = f"NOISE_{instrument}"

        if wavelength_col in df_row and albedo_col in df_row:
            wavelength = np.array(df_row[wavelength_col])
            albedo = np.array(df_row[albedo_col])

            instr_label = f"{label} ({instrument})" if label else f"{instrument}"

            if noise and noisy_albedo_col in df_row and noise_col in df_row:
                noisy_albedo = np.array(df_row[noisy_albedo_col])
                noise_err = np.array(df_row[noise_col])

                (_, caps, bars) = ax.errorbar(
                    wavelength, noisy_albedo, yerr=noise_err, fmt='.', capsize=2, capthick=2,
                    color=color_map[instrument], label=f"{instr_label} - Noisy", zorder=1, **kwargs)
                [bar.set_alpha(0.2) for bar in bars]
                [cap.set_alpha(0.2) for cap in caps]

                ax.plot(wavelength, albedo, color=color_map[instrument], label=instr_label, **kwargs)
            else:
                ax.plot(wavelength, albedo, color=color_map[instrument], label=instr_label, **kwargs)
        else:
            print(f"Data for instrument '{instrument}' not found in DataFrame.")

    ax.set_xlabel("Wavelength [$\\mu$m]")
    ax.set_ylabel("Apparent Albedo")
    ax.legend()

def label_line(line: mlines.Line2D, x: float, label: str = None, align: bool = True, **kwargs) -> None:
    """
    Adds a label to a line at a specified x-coordinate.

    Parameters
    ----------
    line : mlines.Line2D
        The line object to label.
    x : float
        The x-coordinate where the label will be placed.
    label : str, optional
        The label text to display. If not provided, uses the line's current label.
    align : bool, optional
        If True, aligns the label with the line direction. Default is True.
    **kwargs
        Additional keyword arguments passed to `ax.text`.

    Returns
    -------
    None
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # Find corresponding y-coordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    if xdata[ip] == xdata[ip - 1]:  # Avoid division by zero
        y = ydata[ip]
    else:
        y = ydata[ip - 1] + (ydata[ip] - ydata[ip - 1]) * (x - xdata[ip - 1]) / (xdata[ip] - xdata[ip - 1])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip - 1]
        dy = ydata[ip] - ydata[ip - 1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen coordinates
        pt = np.array([10**x, y]).reshape((1, 2)) if ax.get_xscale() == 'log' else np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array([ang]), pt)[0]
    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()
    if "horizontalalignment" not in kwargs and "ha" not in kwargs:
        kwargs["ha"] = "center"
    if "verticalalignment" not in kwargs and "va" not in kwargs:
        kwargs["va"] = "center"
    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()
    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True
    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5
    ax.text(10**x if ax.get_xscale() == 'log' else x, y, label, rotation=trans_angle, **kwargs)

