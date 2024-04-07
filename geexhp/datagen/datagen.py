import os
import msgpack
import pandas as pd
from collections import OrderedDict
from geexhp.pypsg import PSG
from geexhp.conversores import converter_configuracao
from geexhp.modificadores import modificador_aleatorio
from geexhp.modificadores import mod_telescopio
from geexhp.analisadores import EspectroAnalise

def datagen(nplanetas, dir, nome_arq, api = "https://psg.gsfc.nasa.gov/api.php", instrumento = "HWC", verbose = True):
    """
    Esta função gera um conjunto de dados usando o PSG para um número especificado de planetas.

    Parâmetros:
    ----------
    nplanetas : int
        Número de planetas a serem gerados.
    dir : str
        Diretório onde os dados serão salvos.
    api : str
        API do PSG. 
    nome_arq : str
        Nome do arquivo que será salvo
    instrumento : srt
         O instrumento para o qual as configurações do telescópio devem ser modificadas. 
         As opções são 'HWC', 'SS-Vis', 'SS-NIR' e 'SS-UV'. O padrão é 'HWC'.
    verbose : bool, opcional
        Indica se mensagens de saída devem ser impressas ou não. O padrão é True.
 
    Retorna:
    -------
    Esta função não retorna nenhum valor, mas gera e salva os dados dos planetas no formato Parquet.
    """
    if verbose:
        print("\n***** MODO DE GERAÇÃO DE DADOS *****")
    
    planetas = 0
    dados_planetas = pd.DataFrame()
    psg = PSG(server_url=api, timeout_seconds = 200)

    with open("../geexhp/config/default_habex.config", "rb") as f:
        config = OrderedDict(msgpack.unpack(f, raw=False))

    for i in range(nplanetas):

        if verbose:
            print(f"> Gerando exoplaneta {i+1}/{nplanetas}...")

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

            dados_planetas = pd.concat([dados_planetas, pd.concat([config_df, spectrum_df.apply(lambda col: [list(col)], axis=0)], axis=1)])
            planetas += 1

        except Exception as e:
            print("> Erro no PSG:", e)
            print("> Pulando esse planeta...\n")
            continue
    
    if verbose:
      print("> Salvando arquivo...")

    dados_planetas.to_parquet(os.path.join(dir, "data/", f"{nome_arq}.parquet"), index=False)

    if verbose:
      print("***** Arquivo salvo com sucesso! *****")