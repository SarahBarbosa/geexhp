import os
from multiprocessing import Pool
from functools import partial
from collections import OrderedDict
from typing import Union, Dict, List
import msgpack
import pandas as pd
import tqdm
from pypsg import PSG
import geexhp.util._mod as mod
from tqdm import tqdm


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
        """
        try:
            psg = PSG(server_url=self.url, timeout_seconds=200)
            return psg
        except:
            raise ConnectionError("Erro de conexão. Tente novamente.")
        
    def _set_config(self, config: str) -> Dict[str, Union[str, int, float]]:
        """
        Define a configuração do PSG.
        """
        with open(config, "rb") as f:
            config = OrderedDict(msgpack.unpack(f, raw=False))
            return config
    
    def _para_dataframe(self, resultado) -> pd.DataFrame:
        """
        Converte o resultado em um DataFrame pandas.
        """
        try:
            espectro = resultado["spectrum"][:, :2]
            colunas = ["Wave/freq", "Total"]
            return pd.DataFrame(espectro, columns=colunas)
        except Exception as e:
            print(f"Houve um erro ao gerar o DataFrame: {e}")

    def _converte_config(self, configuracao: Union[str, Dict[str, Union[str, int, float]]]) -> pd.DataFrame:
        """
        Converte a configuração PSG para DataFrame.
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
        
        dfs_planetas = [None] * nplanetas
        DATA_DIR = "../data/"

        # Verifica se o diretório de dados existe
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with tqdm(total=nplanetas, desc="Gerando planetas", unit=" planeta", disable=not verbose, colour="green") as barra:
            for i in range(nplanetas):
                try:
                    configuracao = self.config
                    mod.rnd(configuracao)

                    if instrumento != "SS-Vis":
                        mod.instrumento(configuracao, instrumento)
                    
                    resultado = self.psg.run(configuracao)
                    espectro_df = self._para_dataframe(resultado)                
                    config_df = self._converte_config(configuracao)

                    df_planeta = pd.concat([config_df, espectro_df.apply(lambda col: [list(col)], axis=0)], axis=1)
                    dfs_planetas[i] = df_planeta

                except Exception:
                    print("> Erro ao processar esse planeta. Pulando...")
                    continue

                finally:
                    barra.update(1)

        df_final = pd.concat(dfs_planetas)
        df_final.to_parquet(f"../data/{arq}.parquet", index=False)      