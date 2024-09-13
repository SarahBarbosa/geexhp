import numpy as np
import pandas as pd
from math import atan2, degrees

import matplotlib.pyplot as plt
#from matplotlib_inline import backend_inline
from matplotlib import lines as mlines

import corner


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


def plot_spectrum(df: pd.DataFrame, label: str, index: int = None, ax: plt.Axes = None, noise: bool = False, **kwargs) -> plt.Axes:
    """
    Plots the albedo spectrum of a exoplanet.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spectrum data.
    label : str
        Label for the plot legend.
    index : int, optional
        Index of the planet in the DataFrame. If None, assumes the DataFrame directly 
        contains the 'WAVELENGTH' and 'ALBEDO' columns without needing to specify an index. 
        Default is None.
    ax : plt.Axes, optional
        Existing matplotlib Axes object to plot on. If None, a new figure and axes will be created. 
        Default is None.
    noise : bool, optional
        If True, will also plot the noisy data with error bars. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to the plot function.
    
    Returns
    -------
    ax : plt.Axes
        Axes where the spectrum is plotted.
    """
    if ax is None:
        _, ax = plt.subplots()
    
    if index is not None:
        wavelength = df.iloc[index]["WAVELENGTH"]
        albedo = df.iloc[index]["ALBEDO"]
        if noise:
            noisy_albedo = df.iloc[index]["NOISY_ALBEDO"]
            noise_err = df.iloc[index]["NOISE"]
    else:
        wavelength = df["WAVELENGTH"]
        albedo = df["ALBEDO"]
        if noise:
            noisy_albedo = df["NOISY_ALBEDO"]
            noise_err = df["NOISE"]

    if noise:
        _, caps, bars = ax.errorbar(wavelength, noisy_albedo, yerr=noise_err, fmt=".", capsize=2, capthick=2, zorder=1)
        ax.plot(wavelength, albedo, color="tab:orange", label=f"{label} (Noisy)", **kwargs)
        [bar.set_alpha(0.1) for bar in bars]
        [cap.set_alpha(0.1) for cap in caps]
    else:
        ax.plot(wavelength, albedo, label=label, **kwargs)

    ax.set(xlabel="Wavelength [$\mu$m]", ylabel="Apparent Albedo")
    ax.set_ylim(-0.001, albedo.max())
    ax.legend()
    
    return ax



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

def _get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

def overlaid_corner(samples_list, sample_labels, feature_names, true_values=None, filename='corner_plot.png'):
    """
    Plots multiple corner plots overlaid on top of each other.
    """
    CORNER_KWARGS = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=12),
        title_kwargs=dict(fontsize=12),
        quantiles=[0.16, 0.5, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=True,
        max_n_ticks=3,
        title_fmt=".2E"
    )

    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    cmap = plt.cm.get_cmap('gist_rainbow', n)
    colors = [cmap(i) for i in range(n)]

    # Determine the range for each dimension
    plot_range = []
    for dim in range(ndim):
        dim_min = min([np.min(samples_list[i][:, dim]) for i in range(n)])
        dim_max = max([np.max(samples_list[i][:, dim]) for i in range(n)])
        plot_range.append((dim_min, dim_max))

    CORNER_KWARGS.update(range=plot_range)
    CORNER_KWARGS.update(labels=feature_names)

    # First corner plot
    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        **CORNER_KWARGS
    )

    # Overlay additional corner plots
    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=_get_normalisation_weight(len(samples_list[idx]), max_len),
            color=colors[idx],
            **CORNER_KWARGS
        )

    # Plot true values if provided
    if true_values is not None:
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(true_values[i], color='black', linestyle='--')

    # Add legend
    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=12, frameon=False,
        bbox_to_anchor=(1, ndim), loc="upper right"
    )

    plt.savefig(filename)
    plt.show()

