import pandas as pd
from matplotlib import pyplot as plt

class DataVis:
    
    @staticmethod
    def _configurar_matplotlib() -> None:
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

    @staticmethod
    def plot_espectro(df: pd.DataFrame, indice: int) -> plt.Axes:
        """
        Plota o espectro de albedo de um planeta.

        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame contendo os dados do espectro.
        indice : int
            Índice do planeta no DataFrame.

        Retorna:
        --------
        ax : plt.Axes
            Eixo onde o espectro é plotado.
        """
        DataVis._configurar_matplotlib()

        wavelength_str = df.iloc[indice]["WAVELENGHT"]
        wavelength = [float(value.strip()) for value in wavelength_str.split(',')]

        albedo_str = df.iloc[indice]["ALBEDO"]
        albedo = [float(value.strip()) for value in albedo_str.split(',')]

        _, ax = plt.subplots()
        ax.plot(wavelength, albedo, label=f"planeta = {indice}")
        ax.set(xlabel="Comprimento de onda [$\mu$m]", ylabel="Albedo Aparente")
        plt.legend()

        return ax
