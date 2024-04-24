import os
from collections import OrderedDict
from typing import Union, Dict
import msgpack
import pandas as pd
from tqdm import tqdm
from pypsg import PSG
import geexhp.util._mod as mod


class DataGen:
    def __init__(self, url: str, config: str = "../geexhp/config/default_habex.config") -> None:
        """
        Inicializa a classe DataGen.

        Parâmetros:
        -----------
        url : str
            URL do servidor PSG.
        config : str, opcional
            Caminho para o arquivo de configuração PSG. O padrão é "../geexhp/config/default_habex.config".
        """
        self.url = url
        self.psg = self._conecta_psg()
        self.config = self._set_config(config)
    
    def _conecta_psg(self) -> PSG:
        """
        Conecta-se ao servidor PSG.

        Retorna:
        --------
        PSG
            Objeto PSG conectado.
        """
        try:
            psg = PSG(server_url=self.url, timeout_seconds=200)
            return psg
        except:
            raise ConnectionError("Erro de conexão. Tente novamente.")
        
    def _set_config(self, config: str) -> Dict[str, Union[str, int, float]]:
        """
        Define a configuração do PSG.

        Parâmetros:
        -----------
        config : str
            Caminho para o arquivo de configuração PSG.

        Retorna:
        --------
        Dict[str, Union[str, int, float]]
            Configuração do PSG.
        """
        with open(config, "rb") as f:
            config = OrderedDict(msgpack.unpack(f, raw=False))
            return config
    
    def _para_dataframe(self, resultado) -> pd.DataFrame:
        """
        Converte o resultado em um DataFrame pandas.

        Parâmetros:
        -----------
        resultado : any
            Resultado do cálculo do PSG.

        Retorna:
        --------
        pd.DataFrame
            DataFrame contendo os dados do resultado.
        """
        try:
            espectro = resultado.get("spectrum", [])
            colunas = ["Wave/freq [um]", "Total [I/F apparent albedo]", "Noise", "Stellar", "Planet"]
            return pd.DataFrame(espectro, columns=colunas)
        except Exception:
            print("O número de colunas excedeu o necessário!")

    def _converte_config(self, configuracao: Union[str, Dict[str, Union[str, int, float]]]) -> pd.DataFrame:
        """
        Converte a configuração PSG para DataFrame.

        Parâmetros:
        -----------
        configuracao : Union[str, Dict[str, Union[str, int, float]]]
            Configuração PSG como string ou dicionário.

        Retorna:
        --------
        pd.DataFrame
            DataFrame contendo a configuração PSG.
        """
        if isinstance(configuracao, str):
            config_dict = {chave.strip("<"): [valor.strip()] 
                        for linha in configuracao.split("\n") if linha.strip() 
                        for chave, valor in [linha.split(">", 1)]}
            return pd.DataFrame(config_dict)
        elif isinstance(configuracao, dict):
            return pd.DataFrame.from_dict(configuracao, orient="index").T
        else:
            raise ValueError("Tipo de dado não suportado para conversão")
    
    def gerador(self, nplanetas: int, verbose: bool, instrumento: str = "HWC", arq: str = "dados") -> None:
        """
        Gera um conjunto de dados usando o PSG para um número especificado de planetas.

        Parâmetros:
        -----------
        nplanetas : int
            Número de planetas a serem gerados.
        verbose : bool
            Indica se mensagens de saída devem ser impressas ou não.
        instrumento : str, opcional
            O instrumento para o qual as configurações do telescópio devem ser modificadas. 
            As opções são 'HWC', 'SS-NIR', 'SS-UV' e 'SS-Vis'. O padrão é 'HWC'.
        arq : str, opcional
            Nome do arquivo que será salvo. O padrão é "dados".

        Retorna:
        --------
        None
            Este método não retorna nenhum valor. Os dados são salvos em um arquivo Parquet.
        """
        # Verifica se o instrumento está dentro das opções permitidas
        if instrumento not in ["HWC", "SS-NIR", "SS-UV", "SS-Vis"]:
            raise ValueError("O instrumento deve ser 'HWC', 'SS-NIR', 'SS-UV' ou 'SS-Vis'.")
        
        planetas = 0
        df_planetas = pd.DataFrame()

        with tqdm(total=nplanetas, desc="Gerando planetas", unit=" planeta", disable=not verbose) as barra:
            for i in range(nplanetas):
                try:
                    mod.rnd(self.config)

                    if instrumento != "SS-Vis":
                        mod.instrumento(self.config, instrumento)
                    
                    resultado = self.psg.run(self.config)
                    espectro_df = self._para_dataframe(resultado)                
                    config_df = self._converte_config(self.config)

                    if not os.path.exists("../data/"):
                        os.makedirs("../data/")

                    df_dados = pd.concat([config_df, espectro_df.apply(lambda col: [list(col)], axis=0)], axis=1)     
                    df_planetas = pd.concat([df_planetas, df_dados], axis=0)

                    planetas += 1

                except Exception:
                    print("> Erro ao processar esse planeta. Pulando...")
                    continue

                finally:
                    barra.update(1)
        
        df_planetas.to_parquet(f"../data/{arq}.parquet", index=False)
