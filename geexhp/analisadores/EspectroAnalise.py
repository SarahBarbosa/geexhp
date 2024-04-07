from typing import Dict, Optional
from matplotlib import pyplot as plt
import pandas as pd

class EspectroAnalise:
    """
    Esta classe fornece métodos para analisar espectros, incluindo a conversão de espectros para DataFrames pandas,
    a plotagem de espectros e a exibição de cabeçalhos de resultados de espectros.
    """
    
    def __init__(self, resultado: Dict) -> None:
        """
        Inicializa a classe com um resultado de espectro.
        
        Args:
            resultado (Dict): Um dicionário contendo o resultado do espectro.
        """
        self._resultado = resultado
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

    def mostrar_cabecalho(self) -> Optional[str]:
        """
        Mostra o cabeçalho do resultado do espectro.

        Este método retorna o cabeçalho do resultado do espectro. Se nenhum cabeçalho estiver disponível,
        ele retornará "Nenhum cabeçalho disponível".

        Returns:
            str: O cabeçalho do resultado do espectro ou uma mensagem indicando que nenhum cabeçalho está disponível.
        """
        cabecalho = self._resultado.get("header", "Nenhum cabeçalho disponível")
        return cabecalho
    
    def converter_para_dataframe(self) -> pd.DataFrame:
        """
        Converte o espectro em um DataFrame pandas.
        
        Este método converte o espectro, que é uma lista de listas, em um DataFrame pandas. As colunas do DataFrame
        são "Wave/freq [um]", "Total [I/F apparent albedo]", "Noise", "Stellar", "Planet".
        
        Returns:
            pd.DataFrame: Um DataFrame contendo o espectro.
        """
        try:
            espectro = self._resultado.get("spectrum", [])
            if len(espectro[0]) == 5:
                colunas = ["Wave/freq [um]", "Total [I/F apparent albedo]", "Noise", "Stellar", "Planet"]
            else:
                colunas = ["Wave/freq [um]", "Total [I/F apparent albedo]", "Noise", "Planet"]
            return pd.DataFrame(espectro, columns=colunas)
        except:
            try:
                cabecalho = self.mostrar_cabecalho()
                texto_apos_pumas = cabecalho.split('PUMAS |', 1)[1].strip()
                texto = texto_apos_pumas.split('#', 1)[0].strip()
                print("Opa! Aqui retornou esse erro:", texto)
            except IndexError:
                print("O número de colunas excedeu o necessário!")
            except Exception as e:
                print("Ocorreu um erro ao processar o cabeçalho:", e)
            
    def plotar_espectro(self, cor_linha: str = "tab:blue", ax = None, label: str = None, 
                        cor_erro: str = "gray", mostrar_erro: bool = False) -> plt.Axes:
        """
        Plota o espectro.

        Este método plota o espectro usando matplotlib. Ele plota "Wave/freq [um]" no eixo x 
        e "Total [I/F apparent albedo]" no eixo y.

        Parâmetros:
        cor_linha (str, optional): Cor da linha do gráfico. Padrão é "tab:blue".
        ax (matplotlib.axes._axes.Axes, optional): Eixo em que o gráfico será plotado. Se None, cria um novo eixo. Padrão é None.
        label (str, optional): Rótulo do gráfico para a legenda. Padrão é None.
        cor_erro (str, optional): Cor das barras de erro. Padrão é "gray".
        mostrar_erro (bool, optional): Se True, mostra as barras de erro no gráfico. Padrão é False.

        Retorna:
        matplotlib.axes._axes.Axes: O eixo onde o gráfico foi plotado.
        """
        espectro_df = self.converter_para_dataframe()

        if isinstance(espectro_df, pd.core.frame.DataFrame):
            try:
                wave_freq = espectro_df["Wave/freq [um]"]
                total_albedo = espectro_df["Total [I/F apparent albedo]"]
                erro = espectro_df["Noise"] if mostrar_erro else None

                if ax is None:
                    _, ax = plt.subplots()

                ax.plot(wave_freq, total_albedo, color=cor_linha, lw=2, label=label)

                if mostrar_erro:
                    ax.errorbar(wave_freq, total_albedo, yerr=erro, fmt="o", capsize=3, 
                                color=cor_erro, alpha=0.35, markersize=5)

                ax.set(xlabel="Comprimento de onda [$\mu$m]", ylabel="Albedo aparente")

                return ax
            except IndexError:
                print("O número de colunas excedeu o necessário!")
            except Exception as e:
                print("Ocorreu um erro ao plotar o espectro:", e)