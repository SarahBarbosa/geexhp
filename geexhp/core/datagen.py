import os
from collections import OrderedDict, defaultdict
from typing import Union, Dict
import msgpack
import pandas as pd
import tqdm
from pypsg import PSG
from geexhp.core import geostages
from geexhp.utils import mod
from tqdm import tqdm

class DataGen:
    def __init__(self, url: str, config: str = "../geexhp/config/default_habex.config", 
                 estagio: str = "moderna", instrumento: str = "HWC") -> None:
        """
        Inicializa a classe DataGen.

        Parâmetros:
        -----------
        url : str
            URL do servidor PSG.
        config : str, opcional
            Caminho para o arquivo de configuração PSG. O padrão é "../geexhp/config/default_habex.config".
        estagio : str, opcional
            Estágio geológico da Terra a ser considerado. Opções: "moderna", "hadeano"
        instrumento : str, opcional
            O instrumento para o qual as configurações do telescópio devem ser modificadas. 
            As opções são 'HWC', 'SS-NIR', 'SS-UV' e 'SS-Vis'. O padrão é 'HWC'.
        """
        self.url = url
        self.psg = self._conecta_psg()
        self.config = self._set_config(config, estagio, instrumento)

    def _conecta_psg(self) -> PSG:
        """
        Conecta-se ao servidor PSG.
        """
        try:
            psg = PSG(server_url=self.url, timeout_seconds=200)
            return psg
        except:
            raise ConnectionError("Erro de conexão. Tente novamente.")
        
    def _set_config(self, config: str, estagio: str, instrumento: str) -> Dict[str, Union[str, int, float]]:
        """
        Define a configuração do PSG.
        """
        with open(config, "rb") as f:
            config = OrderedDict(msgpack.unpack(f, raw=False))

            if estagio == "moderna":
                geostages.terra_moderna(config)

            if instrumento not in ["HWC", "SS-NIR", "SS-UV", "SS-Vis"]:
                raise ValueError("O instrumento deve ser 'HWC', 'SS-NIR', 'SS-UV' ou 'SS-Vis'.")
            
            if instrumento != "SS-Vis":
                mod.instrumento(config, instrumento)
            
            return config
    
    def gerador(self, nplanetas: int, verbose: bool,  arq: str = "dados") -> None:
        """
        Gera um conjunto de dados usando o PSG para um número especificado de planetas.

        Parâmetros:
        -----------
        nplanetas : int
            Número de planetas a serem gerados.
        verbose : bool
            Indica se mensagens de saída devem ser impressas ou não.
        arq : str, opcional
            Nome do arquivo que será salvo. O padrão é "dados".

        Retorna:
        --------
        None
            Este método não retorna nenhum valor. Os dados são salvos em um arquivo Parquet.
        """      
        d = defaultdict(list)
        DATA_DIR = "../data/"
        os.makedirs(DATA_DIR, exist_ok=True)

        with tqdm(total=nplanetas, desc="Gerando planetas", disable=not verbose, colour="green",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [tempo restante: {remaining}, tempo gasto: {elapsed}]") as barra:

            for _ in range(nplanetas):
                try:
                    configuracao = self.config.copy()
                    mod.rnd(configuracao)

                    espectro = self.psg.run(configuracao)

                    wavelenght = espectro["spectrum"][:, 0].tolist()
                    albedo = espectro["spectrum"][:, 1].tolist()

                    config_dict = {**configuracao, "WAVELENGHT": wavelenght, "ALBEDO": albedo}

                    for k, v in config_dict.items():
                        d[k].append(v)

                except Exception as e:
                    if verbose:
                        print(f"> Erro ao processar esse planeta: {e}. Pulando...")
                    continue
                finally:
                    barra.update(1)

        df_final = pd.DataFrame(d)
        df_final.to_parquet(os.path.join(DATA_DIR, f"{arq}.parquet"), index=False)

        if verbose:
            print("Concluído.") 