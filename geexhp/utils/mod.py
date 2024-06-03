import numpy as np
import numpy.random as npr
import astropy.constants as const
import astropy.units as u

def instrumento(dicionario: dict, instrumento: str) -> None:
    """
    Modifica as configurações do telescópio em um dicionário de entrada. Se essa função não for utilizada, o padrão será SS-VIS.

    Parâmetros
    ----------
    dicionario (dict): Dicionário de configurações a ser modificado.
    instrumento (str): O instrumento para o qual as configurações do telescópio devem ser modificadas. As opções são 'HWC', 'SS-NIR' e 'SS-UV'.

    Retorna
    -------
    None
    """

    # Verifica se o instrumento está dentro das opções permitidas
    if instrumento == "SS-Vis":
        pass
    elif instrumento not in ['HWC', 'SS-NIR', 'SS-UV', 'SS-Vis']:
        raise ValueError("O instrumento deve ser 'HWC', 'SS-NIR', 'SS-UV' ou 'SS-Vis.")

    if instrumento == "HWC":
        dicionario['GENERATOR-INSTRUMENT'] = """
        HabEx_HWC-Spec: The HabEx Workforce Camera (HWC) has two channels that can simultaneously observe the same field of view: 
        an optical channel using delta-doped CCD detectors providing access from 370 nm to 950 nm (QE:0.9), and a near-IR channel 
        using Hawaii-4RG HgCdTe (QE:0.9) arrays providing good throughput from 950 µm to 1.8 µm. The imaging mode can provide 
        spectroscopy (RP<10) via filters at high-throughput, while a grating delivering RP=1000 is assumed to reduce the throughput 
        by 50%.
        """
        dicionario['GENERATOR-RANGE1'] = 0.37
        dicionario['GENERATOR-RANGE2'] = 1.80
        dicionario['GENERATOR-RESOLUTION'] = 1000
        dicionario['GENERATOR-TELESCOPE'] = "SINGLE"
        dicionario['GENERATOR-TELESCOPE1'] = 1
        dicionario['GENERATOR-TELESCOPE2'] = 2.0
        dicionario['GENERATOR-TELESCOPE3'] = 1.0
        dicionario['GENERATOR-NOISEOTEMP'] = 250
        dicionario['GENERATOR-NOISEOEFF'] = '0.000@0.325,0.003@0.337,0.016@0.348,0.067@0.353,0.183@0.365,0.222@0.370,0.240@0.381,\
            0.251@0.401,0.273@0.421,0.302@0.454,0.312@0.508,0.302@0.620,0.283@0.714,0.258@0.793,0.248@0.836,0.261@0.905,0.280@0.955, \
                0.287@1.004,0.295@1.131,0.302@1.291,0.314@1.426,0.321@1.561,0.330@1.693,0.335@1.800'
        dicionario['GENERATOR-NOISEFRAMES'] = 10
        dicionario['GENERATOR-NOISETIME'] = 3600   # Aqui usando o tempo de exposição = 0 (modifica o original para evitar erros)
        dicionario['GENERATOR-NOISEPIXELS'] = 8
        dicionario['GENERATOR-CONT-STELLAR'] = 'Y'
    
    elif instrumento == "SS-NIR":
        dicionario['GENERATOR-INSTRUMENT'] = """
        HabEx_SS-NIR: The HabEx StarShade (SS) will provide extraordinary high-contrast capabilities from the UV (0.2 to 0.45 um), to the 
        visible (0.45 to 1um), and to the infrared (0.975 to 1.8 um). By limiting the number of optical surfaces, this configuration provides 
        high optical throughput (0.2 to 0.4) across this broad of wavelengths, while the quantum efficiency (QE) is expected to be 0.9 for the 
        VU and visible detectors and 0.6 for the infrared detector. The UV channel provides a resolution (RP) of 7, visible channel a maximum 
        of 140 and the infrared 40.
        """
        dicionario['GENERATOR-RANGE1'] = 0.975
        dicionario['GENERATOR-RANGE2'] = 1.80
        dicionario['GENERATOR-RESOLUTION'] = 40
        dicionario['GENERATOR-TELESCOPE3'] = '7e-11@-0.000e+00,7e-11@-1.995e-02,7e-11@-3.830e-02,3.544e-03@-5.439e-02,1.949e-02@-6.791e-02,\
            3.367e-02@-7.434e-02,6.734e-02@-7.982e-02,1.241e-01@-8.561e-02,2.091e-01@-9.108e-02,2.818e-01@-9.526e-02,3.332e-01@-9.752e-02,\
                3.987e-01@-1.014e-01,4.661e-01@-1.052e-01,5.352e-01@-1.075e-01,6.008e-01@-1.110e-01,6.344e-01@-1.130e-01,6.699e-01@-1.155e-01,\
                    6.911e-01@-1.184e-01,7.000e-01@-1.278e-01,7.000e-01@-1.561e-01,7.000e-01@-1.950e-01,7.000e-01@-2.224e-01,7.000e-01@-2.349e-01'
        dicionario['GENERATOR-NOISEFRAMES'] = 1
        dicionario['GENERATOR-NOISETIME'] = 1000

    elif instrumento == "SS-UV":
        dicionario['GENERATOR-INSTRUMENT'] = """
        HabEx_SS-UV: The HabEx StarShade (SS) will provide extraordinary high-contrast capabilities from the UV (0.2 to 0.45 um), to the visible 
        (0.45 to 1um), and to the infrared (0.975 to 1.8 um). By limiting the number of optical surfaces, this configuration provides high optical 
        throughput (0.2 to 0.4) across this broad of wavelengths, while the quantum efficiency (QE) is expected to be 0.9 for the VU and visible 
        detectors and 0.6 for the infrared detector. The UV channel provides a resolution (RP) of 7, visible channel a maximum of 140 and 
        the infrared 40.
        """
        dicionario['GENERATOR-RANGE1'] = 0.2
        dicionario['GENERATOR-RANGE2'] = 0.45
        dicionario['GENERATOR-RESOLUTION'] = 7
        dicionario['GENERATOR-TELESCOPE3'] = '7e-11@-0.000e+00,7e-11@-7.483e-03,7e-11@-1.436e-02,3.544e-03@-2.040e-02,1.949e-02@-2.547e-02,\
            3.367e-02@-2.788e-02,6.734e-02@-2.993e-02,1.241e-01@-3.210e-02,2.091e-01@-3.416e-02,2.818e-01@-3.572e-02,3.332e-01@-3.657e-02,\
                3.987e-01@-3.802e-02,4.661e-01@-3.947e-02,5.352e-01@-4.031e-02,6.008e-01@-4.164e-02,6.344e-01@-4.236e-02,6.699e-01@-4.333e-02,\
                    6.911e-01@-4.441e-02,7.000e-01@-4.791e-02,7.000e-01@-5.853e-02,7.000e-01@-7.314e-02,7.000e-01@-8.340e-02,7.000e-01@-8.810e-02'
        dicionario['GENERATOR-NOISEFRAMES'] = 1
        dicionario['GENERATOR-NOISETIME'] = 1000


def _gas(dicionario: dict, gas: str, multiplicador: float) -> None:
    """
    Modifica os valores de gás em um dicionário ordenado (isso não normaliza as abundâncias!)

    Parâmetros
    ----------
    dicionario (dict): O dicionário ordenado contendo os valores de gás.
    gas (str): O gás a ser modificado: "CO2" , "N2" , "O2" , "H2O", "CO", "H2" , "C2H6" , "HCN", "H2S", "SO2" , "O3" , "CH4" , "N2O", "NH3" , "CH3Cl"
    multiplicador (float): O fator pelo qual modificar o valor do gás.

    Retorna
    -------
    None: O dicionário ordenado é modificado in-place.
    """
    indice_gas = None
    elementos = ["CO2" , "N2" , "O2" , "H2O", "CO", "H2" , "C2H6" , "HCN", "H2S", "SO2" , "O3" , "CH4" , "N2O", "NH3" , "CH3Cl"]

    try:
        indice_gas = elementos.index(gas)
    except ValueError:
        print("Gás não encontrado na lista de elementos.")
        return

    for key, value in dicionario.items():
        if key.startswith("ATMOSPHERE-LAYER-"):
            values = value.split(',')
            valor_gas = float(values[indice_gas + 2])
            novo_valor = valor_gas * multiplicador
            
            values[indice_gas + 2] = str(novo_valor)
            dicionario[key] = ','.join(values)
            
def rnd(dicionario: dict) -> None:
    """
    Modifica aleatoriamente as configurações de um dicionário de entrada com base em vários parâmetros.

    Parâmetros
    ----------
    dicionario (dict): Dicionário de configurações a ser modificado.

    Retorna
    -------
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
    if classe_estrela == 'U':                                   # Isso é tipo F -- usa modelo de corpo negro
        temp_estrela = round(np.random.uniform(6000, 7220), 3)  # Em K
        raio_estrela = round(np.random.uniform(1.18, 1.79), 3)  # Em raio solar 
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
    elementos = ["CO2" , "N2" , "O2" , "H2O", "CO", "H2" , "C2H6" , "HCN", "H2S", "SO2" , "O3" , "CH4" , "N2O", "NH3" , "CH3Cl"]    

    for gas in elementos:
        multiplicador = np.random.uniform(0,2)                        # Intervalo aleatório de multiplicadores
        if np.random.random() < 0.2:                                  # Probabilidade de 20% de definir o multiplicador como zero
            multiplicador = 0.0
            
        _gas(dicionario, gas, multiplicador)
    
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

    
    # Aqui precisamos modificar futuramente: precisamos pegar a média da molécula em todas as camas e ai sim multipĺicar pelo peso
    # molecular médio
    peso_molecular = np.array([44.0095, 28.01340, 31.99880, 18.01528, 28.0101, 2.01588, 30.0690, 27.0253, 34.0809, 64.0638, 47.99820, 
                      16.0425, 44.01280, 17.03052, 50.4875])
    valores = np.array(dicionario["ATMOSPHERE-LAYER-1"].split(',')[2:], dtype=float)
    peso_molecular_medio = np.sum(peso_molecular * valores)
    dicionario['ATMOSPHERE-WEIGHT'] = peso_molecular_medio    # Peso molecular da atmosfera [g/mol]

    ## Modificando o Semi-eixo maior (AU) para estar dentro da zona de habitabilidade
    # Fonte: Habitable zones around main-sequence stars... (Kopparapu et al. 2013) 
    # https://iopscience.iop.org/article/10.1088/0004-637X/765/2/131/pdf
    # Equações 2, 3 e Tabela 3 de Kopparapu et al. 2013
    temp = temp_estrela - 5780

    # Lei de Stefan-Boltzmann para a luminosidade! 
    # https://arxiv.org/abs/2402.07947
    luminosidade_estrela = 4 * np.pi * ((raio_estrela * const.R_sun.value) ** 2) * const.sigma_sb.value * temp_estrela ** 4
    
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
    semi_eixo_maior = np.random.uniform(a_inferior, a_superior)
    dicionario['OBJECT-STAR-DISTANCE'] = semi_eixo_maior    # em AU

    ## Modificando a metalicidade da estrela de forma aleatória
    # Motivação e fonte: High metallicity and non-equilibrium chemistry... (Madhusudhan1 e Seager 2011)
    # https://iopscience.iop.org/article/10.1088/0004-637X/729/1/41/meta
    # 10x maior e menor a metalicidade do sol (em dex)
    dicionario['OBJECT-STAR-METALLICITY'] = round(np.random.uniform(-1, 1), 3)

    ## Modificando o raio e massa do planeta de tal forma que o planeta possa manter a atmosfera
    # Baseado em INARA: Intelligent exoplaNet Atmospheric RetrievAl
    # https://ui.adsabs.harvard.edu/abs/2019absc.conf10105O/abstract
    mantem_atmosfera = False

    while not mantem_atmosfera:
        # Intervalo do raio para planetas terrestes ~ 1.7 raios terrestres
        # Fonte: The Super-Earth Opportunity – Search for Habitable Exoplanets in the 2020s
        # https://arxiv.org/pdf/1903.05258
        raio_planeta = np.random.uniform(0.5, 1.6)

        # Considerando as equações (11) e (12) de Sotin, Grasset e Mocquet (2007):
        # Fonte: Mass–radius curve for extrasolar Earth-like planets and ocean planets
        # https://ui.adsabs.harvard.edu/abs/2007Icar..191..337S/abstract
        if raio_planeta <= 1:
            massa_planeta = raio_planeta ** (1 / 0.306)
        else:           
            massa_planeta = raio_planeta ** (1 / 0.274)
        
        # Calculando a gravidade do planeta (g = GM/r²) em m/s²
        gravidade = const.G.value * (massa_planeta * const.M_earth.value) / (raio_planeta * const.R_earth.value) ** 2

        # Calculando a velociade de escape
        velocidade_escape = np.sqrt(2 * gravidade * raio_planeta * const.R_earth.value)
        velocidade_escape_km = velocidade_escape / 1000

        # Vamos calcular a insolação do planeta em termos da Terra
        # Equação (4) de Zahnle e Catling (2017)
        # https://arxiv.org/pdf/1702.03386
        insolacao = (luminosidade_estrela / const.L_sun.value)  * (1 / semi_eixo_maior ** 2)

        # Vamos seguir a aproximação "arriscada" do INARA aqui, onde usaremos a Figura 2 de Zahnle e Catling (2017)
        # para "estimar" de forma grotesca a linha no plot loglog (inclusive o paper fala The line is drawn by eye.)
        slope_shoreline = np.log10(1e4 / 1e-6) / np.log10(70 / 0.2)
        insolacao_planeta = 1e4 * (velocidade_escape_km / 70) ** slope_shoreline

        # A insolação atual precisa ser menor que a insolação do planeta
        if insolacao < insolacao_planeta:
            mantem_atmosfera = True
    
    # Qaudno tudo der certo, podemos modificar o dicionário
    dicionario['OBJECT-DIAMETER'] = 2 * raio_planeta * const.R_earth.to(u.km).value   # Diâmetro do planeta (em km)
    dicionario['OBJECT-GRAVITY']  = gravidade                                         # Gravidade do planeta em m/s²










        




