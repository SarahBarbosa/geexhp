import os
from collections import OrderedDict, defaultdict
from typing import Union, Dict
import msgpack
import pandas as pd
import tqdm
from pypsg import PSG
import geexhp.util.mod as mod
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
        
        d = defaultdict(list)
        DATA_DIR = "../data/"

        # Verifica se o diretório de dados existe
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with tqdm(total=nplanetas, desc="Gerando planetas", disable=not verbose, colour="green", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [ tempo restante: {remaining}, tempo gasto: {elapsed}]") as barra:

            for _ in range(nplanetas):              
                try:
                    configuracao = self.config.copy()

                    mod.rnd(configuracao)

                    if instrumento != "SS-Vis":
                        mod.instrumento(configuracao, instrumento)

                    config_dict = dict(configuracao)
                    espectro = self.psg.run(configuracao)

                    wavelenght = ", ".join(str(num) for num in espectro["spectrum"][:, 0])
                    albedo = ", ".join(str(num) for num in espectro["spectrum"][:, 1])

                    espectro_dict = {"WAVELENGHT": wavelenght, "ALBEDO": albedo}
                    config_dict.update(espectro_dict)

                    for k, v in config_dict.items():
                        d[k].append(v)

                except Exception:
                    print("> Erro ao processar esse planeta. Pulando...")
                    continue

                finally:
                    barra.update(1)

        df_final = pd.DataFrame(d)
        df_final.to_parquet(f"../data/{arq}.parquet", index=False)      