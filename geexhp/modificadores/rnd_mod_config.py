import numpy as np
from geexhp.modificadores import mod_valor_gas
from astropy import constants as const

def modificador_aleatorio(dicionario: dict) -> None:
    """
    Modifica aleatoriamente as configurações de um dicionário de entrada com base em vários parâmetros.

    Parâmetros:
    dicionario (dict): Dicionário de configurações a ser modificado.

    Retorna:
    None
    """
    # Modifica o tipo espectral da estrela
    tipo_espectral = ['U', 'G', 'K', 'M']
    classe_estrela =  tipo_espectral[np.random.randint(0, len(tipo_espectral))]

    dicionario['OBJECT-STAR-TYPE'] = classe_estrela         # Classe da estrela
    dicionario['GEOMETRY-STELLAR-TYPE'] = classe_estrela    # Classe da estrela de ocultação

    # Modifica o raio e a temperatura da estrela
    # Fonte: A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence (2019)
    # https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.dat
    if classe_estrela == 'U':                               # Isso é tipo F -- usa modelo de corpo negro
        temp_estrela = round(np.random.uniform(6000, 7220), 3)
        raio_estrela = round(np.random.uniform(1.18, 1.79), 3)
    elif classe_estrela == 'G':
        temp_estrela = round(np.random.uniform(5340, 5920), 3)
        raio_estrela = round(np.random.uniform(0.876, 1.12), 3)
    elif classe_estrela == 'K':
        temp_estrela = round(np.random.uniform(3940, 5280), 3)
        raio_estrela = round(np.random.uniform(0.552, 0.817), 3)
    elif classe_estrela == 'M':
        temp_estrela = round(np.random.uniform(2320, 3870), 3)
        raio_estrela = round(np.random.uniform(0.104, 0.559), 3)

    dicionario['OBJECT-STAR-RADIUS'] = raio_estrela             
    dicionario['OBJECT-STAR-TEMPERATURE'] = temp_estrela        
    dicionario['GEOMETRY-STELLAR-TEMPERATURE'] = temp_estrela  

    # Gera a longitude sub solar em graus
    longitude = np.random.uniform(-360, 360)
    dicionario['OBJECT-SOLAR-LONGITUDE']  = longitude   

    # Gera a latitude sub solar em graus
    latitude = np.random.uniform(-90, 90)
    dicionario['OBJECT-SOLAR-LATITUDE']   = latitude    

    # Pressão atmosférica em mbar
    pressao = round(np.random.uniform(500, 1500), 3)
    dicionario["ATMOSPHERE-PRESSURE"] = str(pressao)

    # Modificando os mixings ratio
    gases_simulados = ['H2O','CO2','O3','N2O','CO','CH4','O2','N2']
    elementos = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3', 'OH', 
                 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 
                 'C2H2', 'C2H6', 'PH3']
    
    for gas in elementos:
        # Estou eliminando todos os outros elementos que não fazem parte dos gases simulados
        if gas in gases_simulados:
            if gas in ["H2O", "CO2", "O2", "N2"]:
                valor_max = 1e-2
            else:
                valor_max = 0.5
        else:
            valor_max = 0

        multiplicador = np.random.uniform(0, valor_max)               # Intervalo aleatório de multiplicadores
        
        if np.random.random() < 0.3:                                  # Probabilidade de 30% de definir o multiplicador como zero
            multiplicador = 0.0
            
        mod_valor_gas(dicionario, gas, multiplicador)
    
    # Fazendo com que os dados fiquem normalizados
    for i in range(60):
        valores = dicionario[f"ATMOSPHERE-LAYER-{i+1}"].split(',')
        valores_to_normalize = valores[2:]  # Valores a serem normalizados

        # Verificando se há valores a serem normalizados
        if valores_to_normalize:
            soma = sum(float(valor) for valor in valores_to_normalize)
            
            # Verificando se a soma é diferente de zero
            if soma != 0:
                # Substituindo os valores pelos valores divididos pela soma
                valores[2:] = [str(float(valor) / soma) for valor in valores_to_normalize]
            else:
                # Se a soma for zero, mantenha os valores originais
                valores[2:] = valores_to_normalize
            
            # Valores normalizados
            dicionario[f"ATMOSPHERE-LAYER-{i+1}"] = ','.join(valores)
        else:
            # Se não houver valores para normalizar, mantenha os valores originais
            dicionario[f"ATMOSPHERE-LAYER-{i+1}"] = ','.join(valores)
    
    pesos_molares = np.array([
        18.01528,   # H2O
        44.0095,    # CO2
        47.9982,    # O3
        44.0128,    # N2O
        28.0101,    # CO
        16.0425,    # CH4
        31.9988,    # O2
        30.0061,    # NO
        64.066,     # SO2
        46.0055,    # NO2
        17.0305,    # NH3
        63.0129,    # HNO3
        17.0073,    # OH
        20.0063,    # HF
        36.4609,    # HCl
        80.9119,    # HBr
        127.912,    # HI
        51.45,      # ClO
        60.075,     # OCS
        30.026,     # H2CO
        52.46,      # HOCl
        28.0134,    # N2
        27.0253,    # HCN
        50.4875,    # CH3Cl
        34.0147,    # H2O2
        26.0373,    # C2H2
        30.069,     # C2H6
        33.9976     # PH3
        ])

    somas = []

    for i in range(50):
        valores = dicionario[f"ATMOSPHERE-LAYER-{i+1}"].split(',')[2:]
        soma_multiplicacao = sum(float(valor) * peso_molecular for valor, peso_molecular in zip(valores, pesos_molares))
        somas.append(soma_multiplicacao)

    peso_molecular_medio = round(np.mean(somas), 2)
    dicionario['ATMOSPHERE-WEIGHT'] = peso_molecular_medio    # Peso molecular da atmosfera [g/mol]

    ## Modificando o Semi-eixo maior (AU) para estar dentro da zona de habitabilidade
    # Fonte: Habitable zones around main-sequence stars... (Kopparapu et al. 2013) 
    # https://iopscience.iop.org/article/10.1088/0004-637X/765/2/131/pdf
    # Equações 2, 3 e Tabela 3 de Kopparapu et al. 2013
    temp = temp_estrela - 5780

    # Lei de Stefan-Boltzmann para a luminosidade! 
    # https://arxiv.org/abs/2402.07947
    luminosidade_estrela = 4 * np.pi * ((raio_estrela) ** 2) * const.sigma_sb.value * temp_estrela ** 4
    
    # Recent Venus (limite inferior) - Tabela 3 (primeira coluna)
    S_eff_odot = 1.7753
    a = 1.4316e-4
    b = 2.9875e-9
    c = -7.5702e-12
    d = -1.1635e-15
    S_eff_inferior = S_eff_odot + a * temp + b * temp ** 2 + c * temp ** 3 + d * temp ** 4

    # Early Mars (limite superior) - Tabela 3 (segunda coluna)
    S_eff_odot = 0.3179
    a = 5.4513e-5
    b = 1.5313e-9
    c = -2.7786e-12
    d = -4.8997e-16
    S_eff_superior = S_eff_odot + a * temp + b * temp ** 2 + c * temp ** 3 + d * temp ** 4

    # Distância da zona de habitabilidade (Equação 3)
    a_inferior = ((luminosidade_estrela / const.L_sun.value) / S_eff_inferior) ** 0.5
    a_superior = ((luminosidade_estrela / const.L_sun.value) / S_eff_superior) ** 0.5

    # Colocando  o planeta em algum lugar desse range...
    a = np.random.uniform(a_inferior, a_superior)
    dicionario['OBJECT-STAR-DISTANCE'] = a

    ## Modificando a metalicidade da estrela de forma aleatória
    # Motivação e fonte: High metallicity and non-equilibrium chemistry... (Madhusudhan1 e Seager 2011)
    # https://iopscience.iop.org/article/10.1088/0004-637X/729/1/41/meta
    # 10x maior e menor a metalicidade do sol (em dex)
    dicionario['OBJECT-STAR-METALLICITY'] = round(np.random.uniform(-1, 1), 3)