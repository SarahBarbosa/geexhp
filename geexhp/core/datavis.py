import pandas as pd
from matplotlib import pyplot as plt

class DataVis:
    def __init__(self) -> None:
        self._configurar_matplotlib()

    def _configurar_matplotlib(self) -> None:
        """
        Configura os parâmetros do matplotlib.
        """
        plt.rcParams.update({
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize" : (10, 4)
            })