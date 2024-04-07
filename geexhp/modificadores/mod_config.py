def mod_valor_gas(dicionario: dict, gas: str, multiplicador: float) -> None:
    """
    Modifica os valores de gás em um dicionário ordenado (isso não normaliza as abundâncias!)

    Parâmetros:
    dicionario (dict): O dicionário ordenado contendo os valores de gás.
    gas (str): O gás a ser modificado: 'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 
    'NH3', 'HNO3', 'OH','HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2',
    'C2H2', 'C2H6', 'PH3'
    multiplicador (float): O fator pelo qual modificar o valor do gás.

    Retorna:
    None: O dicionário ordenado é modificado in-place.
    """
    indice_gas = None
    elementos = ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3', 'OH', 
                 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 
                 'C2H2', 'C2H6', 'PH3']
    try:
        indice_gas = elementos.index(gas)
    except ValueError:
        print("Gás não encontrado na lista de elementos.")
        return

    for key, value in dicionario.items():
        if key.startswith("ATMOSPHERE-LAYER-"):
            values = value.split(',')
            valor_gas = float(values[indice_gas + 2])
            if multiplicador == 0:
                novo_valor = 0
            else:
                novo_valor = valor_gas * multiplicador
            
            values[indice_gas + 2] = str(novo_valor)
            dicionario[key] = ','.join(values)

def mod_telescopio(dicionario: dict, instrumento: str) -> None:
    """
    Modifica as configurações do telescópio em um dicionário de entrada. Se essa função não for utilizada, o padrão será SS-VIS.

    Parâmetros:
    dicionario (dict): Dicionário de configurações a ser modificado.
    instrumento (str): O instrumento para o qual as configurações do telescópio devem ser modificadas. As opções são 'HWC', 'SS-NIR' e 'SS-UV'.

    Retorna:
    None
    """

    # Verifica se o instrumento está dentro das opções permitidas
    if instrumento not in ['HWC', 'SS-NIR', 'SS-UV']:
        raise ValueError("O instrumento deve ser 'HWC', 'SS-NIR' ou 'SS-UV'.")

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
