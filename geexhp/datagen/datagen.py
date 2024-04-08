import os
import msgpack
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from geexhp.pypsg import PSG
from geexhp.conversores import converter_configuracao
from geexhp.modificadores import modificador_aleatorio
from geexhp.modificadores import mod_telescopio
from geexhp.analisadores import EspectroAnalise

def datagen(nplanetas: int, dir: str, nome_arq: str, api: str = "https://psg.gsfc.nasa.gov/api.php",
            instrumento: str = "HWC", verbose: bool = True) -> None:
    """
    Esta função gera um conjunto de dados usando o PSG para um número especificado de planetas.

    Parâmetros:
    ----------
    nplanetas (int): Número de planetas a serem gerados.
    dir (str): Diretório onde os dados serão salvos.
    api (str, opcional): API do PSG. O padrão é "https://psg.gsfc.nasa.gov/api.php".
    nome_arq (str): Nome do arquivo que será salvo.
    instrumento (str, opcional): O instrumento para o qual as configurações do telescópio devem ser modificadas. 
                                As opções são 'HWC', 'SS-Vis', 'SS-NIR' e 'SS-UV'. O padrão é 'HWC'.
    verbose (bool, opcional): Indica se mensagens de saída devem ser impressas ou não. O padrão é True.
    
    Retorna:
    -------
    None: Esta função não retorna nenhum valor, mas gera e salva os dados dos planetas no formato Parquet.
    """
    if verbose:
        print("\n" + "*" * 30 + " MODO DE GERAÇÃO DE DADOS " + "*" * 30 + "\n")
    
    planetas = 0
    dados_planetas = pd.DataFrame()
    psg = PSG(server_url=api, timeout_seconds=200)
    print("\n")

    with open("../geexhp/config/default_habex.config", "rb") as f:
        config = OrderedDict(msgpack.unpack(f, raw=False))

    with tqdm(total=nplanetas, desc='Gerando planetas', unit='planeta', disable=not verbose) as progress_bar:
        for i in range(nplanetas):
            try:
                modificador_aleatorio(config)

                if instrumento != 'SS-Vis':
                    mod_telescopio(config, instrumento)
                else:
                    pass
                
                resultado = psg.run(config)
                spectrum = EspectroAnalise(resultado)
                spectrum_df = spectrum.converter_para_dataframe()

                config_df = converter_configuracao(config)

                if not os.path.exists(os.path.join(dir, "data/")):
                    os.makedirs(os.path.join(dir, "data/"))

                dados_planetas = pd.concat([dados_planetas, 
                                            pd.concat([config_df, spectrum_df.apply(lambda col: [list(col)], axis=0)], axis=1)])
                planetas += 1

            except Exception as e:
                if verbose:
                    print("\n> Erro no PSG:", e) 
                    print("> Pulando esse planeta...\n")
                continue

            finally:
                progress_bar.update(1)

    if verbose:
        print("\n" + "*" * 30 + " Arquivo salvo com sucesso! " + "*" * 30)
    
    dados_planetas.to_parquet(os.path.join(dir, "data/", f"{nome_arq}.parquet"), index=False)