import pandas as pd
from typing import Union, Dict

def converter_configuracao(configuracao: Union[Dict, str], para_dataframe: bool = True) -> Union[pd.DataFrame, str]:
    """
    Converte uma configuração de string para DataFrame ou vice-versa.

    Args:
        configuracao (Union[Dict, str]): Configuração a ser convertida.
        para_dataframe (bool, optional): Se True, converte para DataFrame. 
        Se False, converte para string.

    Raises:
        ValueError: Se o tipo de dado da configuração não for suportado para conversão.

    Returns:
        Union[pd.DataFrame, str]: Configuração convertida.
    """
    if isinstance(configuracao, str):
        if para_dataframe:
            config_dict = {chave.strip("<"): [valor.strip()] 
                           for linha in configuracao.split("\n") if linha.strip() 
                           for chave, valor in [linha.split(">", 1)]}
            return pd.DataFrame(config_dict)
        else:
            return configuracao
    elif isinstance(configuracao, dict):
        if para_dataframe:
            return pd.DataFrame.from_dict(configuracao, orient="index").T
        else:
            return "\n".join([f"<{chave}>{valor}" for chave, valor in configuracao.items()])
    else:
        raise ValueError("Tipo de dado não suportado para conversão")