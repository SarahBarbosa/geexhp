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

def moleculas_simuladas(dicionario):
    camadas = 60

    # VALORES FIXOS IGUAIS DA TERRA
    pressao_fixa = np.array([1.013e+00, 8.988e-01, 7.950e-01, 7.012e-01, 6.166e-01, 5.405e-01,
                        4.722e-01, 4.111e-01, 3.565e-01, 3.080e-01, 2.650e-01, 2.270e-01,
                        1.940e-01, 1.658e-01, 1.417e-01, 1.211e-01, 1.035e-01, 8.850e-02,
                        7.565e-02, 6.467e-02, 5.529e-02, 4.729e-02, 4.047e-02, 3.467e-02,
                        2.972e-02, 2.549e-02, 1.743e-02, 1.197e-02, 8.258e-03, 5.746e-03,
                        4.041e-03, 2.871e-03, 2.060e-03, 1.491e-03, 1.090e-03, 7.978e-04,
                        4.250e-04, 2.190e-04, 1.090e-04, 5.220e-05, 2.400e-05, 1.050e-05,
                        4.460e-06, 1.840e-06, 7.600e-07, 3.200e-07, 1.450e-07, 7.100e-08,
                        4.010e-08, 2.540e-08, 6.023e-09, 2.614e-09, 1.337e-09, 7.427e-10,
                        4.386e-10, 2.681e-10, 1.693e-10, 1.097e-10, 7.234e-11, 4.900e-11])
    temperatura_fixa = np.array([ 288.2,  281.7,  275.2,  268.7,  262.2,  255.7,  249.2,  242.7,
                                236.2,  229.7,  223.3,  216.8,  216.7,  216.7,  216.7,  216.7,
                                216.7,  216.7,  216.7,  216.7,  216.7,  217.6,  218.6,  219.6,
                                220.6,  221.6,  224. ,  226.5,  229.6,  236.5,  243.4,  250.4,
                                257.3,  264.2,  270.6,  270.7,  260.8,  247. ,  233.3,  219.6,
                                208.4,  198.6,  188.9,  186.9,  188.4,  195.1,  208.8,  240. ,
                                300. ,  360. ,  610. ,  759. ,  853. ,  911. ,  949. ,  973. ,
                                988. ,  998. , 1000. , 1010. ])
    
    CO2 = np.full(camadas, 3.795e-04)
    N2 = np.full(camadas, 0.781)
    O2 = np.full(camadas, 0.209)
    H2O = np.array([7.745e-03, 6.071e-03, 4.631e-03, 3.182e-03, 2.158e-03, 1.397e-03,
                    9.254e-04, 5.720e-04, 3.667e-04, 1.583e-04, 6.996e-05, 3.613e-05,
                    1.906e-05, 1.085e-05, 5.927e-06, 5.000e-06, 3.950e-06, 3.850e-06,
                    3.825e-06, 3.850e-06, 3.900e-06, 3.975e-06, 4.065e-06, 4.200e-06,
                    4.300e-06, 4.425e-06, 4.575e-06, 4.725e-06, 4.825e-06, 4.900e-06,
                    4.950e-06, 5.025e-06, 5.150e-06, 5.225e-06, 5.250e-06, 5.225e-06,
                    5.100e-06, 4.750e-06, 4.200e-06, 3.500e-06, 2.825e-06, 2.050e-06,
                    1.330e-06, 8.500e-07, 5.400e-07, 4.000e-07, 3.400e-07, 2.800e-07,
                    2.400e-07, 2.000e-07, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
                    0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00])
    CO = np.array([1.500e-07, 1.450e-07, 1.399e-07, 1.349e-07, 1.312e-07, 1.303e-07,
                    1.288e-07, 1.247e-07, 1.185e-07, 1.094e-07, 9.962e-08, 8.964e-08,
                    7.814e-08, 6.374e-08, 5.025e-08, 3.941e-08, 3.069e-08, 2.489e-08,
                    1.966e-08, 1.549e-08, 1.331e-08, 1.232e-08, 1.232e-08, 1.307e-08,
                    1.400e-08, 1.498e-08, 1.598e-08, 1.710e-08, 1.850e-08, 2.009e-08,
                    2.220e-08, 2.497e-08, 2.824e-08, 3.241e-08, 3.717e-08, 4.597e-08,
                    6.639e-08, 1.073e-07, 1.862e-07, 3.059e-07, 6.375e-07, 1.497e-06,
                    3.239e-06, 5.843e-06, 1.013e-05, 1.692e-05, 2.467e-05, 3.356e-05,
                    4.148e-05, 5.000e-05, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
                    0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00])
    C2H6 = np.array([2.00e-09, 2.00e-09, 2.00e-09, 2.00e-09, 1.98e-09, 1.95e-09,
                    1.90e-09, 1.85e-09, 1.79e-09, 1.72e-09, 1.58e-09, 1.30e-09,
                    9.86e-10, 7.22e-10, 4.96e-10, 3.35e-10, 2.14e-10, 1.49e-10,
                    1.05e-10, 7.96e-11, 6.01e-11, 4.57e-11, 3.40e-11, 2.60e-11,
                    1.89e-11, 1.22e-11, 5.74e-12, 2.14e-12, 8.49e-13, 3.42e-13,
                    1.34e-13, 5.39e-14, 2.25e-14, 1.04e-14, 6.57e-15, 4.74e-15,
                    3.79e-15, 3.28e-15, 2.98e-15, 2.79e-15, 2.66e-15, 2.56e-15,
                    2.49e-15, 2.43e-15, 2.37e-15, 2.33e-15, 2.29e-15, 2.25e-15,
                    2.22e-15, 2.19e-15, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])
    HCN = np.array([1.70e-10, 1.65e-10, 1.63e-10, 1.61e-10, 1.60e-10, 1.60e-10,
                    1.60e-10, 1.60e-10, 1.60e-10, 1.60e-10, 1.60e-10, 1.60e-10,
                    1.60e-10, 1.59e-10, 1.57e-10, 1.55e-10, 1.52e-10, 1.49e-10,
                    1.45e-10, 1.41e-10, 1.37e-10, 1.34e-10, 1.30e-10, 1.25e-10,
                    1.19e-10, 1.13e-10, 1.05e-10, 9.73e-11, 9.04e-11, 8.46e-11,
                    8.02e-11, 7.63e-11, 7.30e-11, 7.00e-11, 6.70e-11, 6.43e-11,
                    6.21e-11, 6.02e-11, 5.88e-11, 5.75e-11, 5.62e-11, 5.50e-11,
                    5.37e-11, 5.25e-11, 5.12e-11, 5.00e-11, 4.87e-11, 4.75e-11,
                    4.62e-11, 4.50e-11, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])
    SO2 = np.array([3.00e-10, 2.74e-10, 2.36e-10, 1.90e-10, 1.46e-10, 1.18e-10,
                    9.71e-11, 8.30e-11, 7.21e-11, 6.56e-11, 6.08e-11, 5.79e-11,
                    5.60e-11, 5.59e-11, 5.64e-11, 5.75e-11, 5.75e-11, 5.37e-11,
                    4.78e-11, 3.97e-11, 3.19e-11, 2.67e-11, 2.28e-11, 2.07e-11,
                    1.90e-11, 1.75e-11, 1.54e-11, 1.34e-11, 1.21e-11, 1.16e-11,
                    1.21e-11, 1.36e-11, 1.65e-11, 2.10e-11, 2.77e-11, 3.56e-11,
                    4.59e-11, 5.15e-11, 5.11e-11, 4.32e-11, 2.83e-11, 1.33e-11,
                    5.56e-12, 2.24e-12, 8.96e-13, 3.58e-13, 1.43e-13, 5.73e-14,
                    2.29e-14, 9.17e-15, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]) 
    O3 = np.array([2.660e-08, 2.931e-08, 3.237e-08, 3.318e-08, 3.387e-08, 3.768e-08,
                    4.112e-08, 5.009e-08, 5.966e-08, 9.168e-08, 1.313e-07, 2.149e-07,
                    3.095e-07, 3.846e-07, 5.030e-07, 6.505e-07, 8.701e-07, 1.187e-06,
                    1.587e-06, 2.030e-06, 2.579e-06, 3.028e-06, 3.647e-06, 4.168e-06,
                    4.627e-06, 5.118e-06, 5.803e-06, 6.553e-06, 7.373e-06, 7.837e-06,
                    7.800e-06, 7.300e-06, 6.200e-06, 5.250e-06, 4.100e-06, 3.100e-06,
                    1.800e-06, 1.100e-06, 7.000e-07, 3.000e-07, 2.500e-07, 3.000e-07,
                    5.000e-07, 7.000e-07, 7.000e-07, 4.000e-07, 2.000e-07, 5.000e-08,
                    5.000e-09, 5.000e-10, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
                    0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00])
    CH4 = np.full(camadas, 1.700e-06)
    N2O = np.full(camadas, 3.200e-07)
    NH3 = np.array([5.00e-10, 5.00e-10, 4.63e-10, 3.80e-10, 2.88e-10, 2.04e-10,
                    1.46e-10, 9.88e-11, 6.48e-11, 3.77e-11, 2.03e-11, 1.09e-11,
                    6.30e-12, 3.12e-12, 1.11e-12, 4.47e-13, 2.11e-13, 1.10e-13,
                    6.70e-14, 3.97e-14, 2.41e-14, 1.92e-14, 1.72e-14, 1.59e-14,
                    1.44e-14, 1.23e-14, 9.37e-15, 6.35e-15, 3.68e-15, 1.82e-15,
                    9.26e-16, 2.94e-16, 8.72e-17, 2.98e-17, 1.30e-17, 7.13e-18,
                    4.80e-18, 3.66e-18, 3.00e-18, 2.57e-18, 2.27e-18, 2.04e-18,
                    1.85e-18, 1.71e-18, 1.59e-18, 1.48e-18, 1.40e-18, 1.32e-18,
                    1.25e-18, 1.19e-18, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]) 
    CH3Cl = np.array([7.00e-10, 6.70e-10, 6.43e-10, 6.22e-10, 6.07e-10, 6.02e-10,
                    6.00e-10, 6.00e-10, 5.98e-10, 5.94e-10, 5.88e-10, 5.79e-10,
                    5.66e-10, 5.48e-10, 5.28e-10, 5.03e-10, 4.77e-10, 4.49e-10,
                    4.21e-10, 3.95e-10, 3.69e-10, 3.43e-10, 3.17e-10, 2.86e-10,
                    2.48e-10, 1.91e-10, 1.10e-10, 4.72e-11, 1.79e-11, 7.35e-12,
                    3.03e-12, 1.32e-12, 8.69e-13, 6.68e-13, 5.60e-13, 4.94e-13,
                    4.56e-13, 4.32e-13, 4.17e-13, 4.05e-13, 3.96e-13, 3.89e-13,
                    3.83e-13, 3.78e-13, 3.73e-13, 3.69e-13, 3.66e-13, 3.62e-13,
                    3.59e-13, 3.56e-13, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
                    0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])                 

    # ADICIONANDO NOVOS ELEMENTOS (2%, isso é muito!):
    H2 = np.full(camadas, 0.02)
    H2S = np.full(camadas, 0.02)

    # Moléculas (https://pt.webqc.org/mmcalc.php)
    moleculas = ["CO2" , "N2" , "O2" , "H2O", "CO", "H2" , "C2H6" , "HCN", "H2S", "SO2" , "O3" , "CH4" , "N2O", "NH3" , "CH3Cl"]
    peso_molecular = [44.0095, 28.01340, 31.99880, 18.01528, 28.0101, 2.01588, 30.0690, 27.0253, 34.0809, 64.0638, 47.99820, 
                      16.0425, 44.01280, 17.03052, 50.4875]
    
    for i in range(camadas):
        total_abundancia_camada = CO2[i] + N2[i] + O2[i] + H2O[i] + CO[i] + H2[i] + C2H6[i] + HCN[i] + H2S[i] + SO2[i] + \
             O3[i] + CH4[i] + N2O[i] + NH3[i] + CH3Cl[i]
        
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] = ','.join([str(pressao_fixa[i]), str(temperatura_fixa[i])])
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(CO2[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(N2[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(O2[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(H2O[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(CO[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(H2[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(C2H6[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(HCN[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(H2S[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(SO2[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(O3[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(CH4[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(N2O[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(NH3[i] / total_abundancia_camada)
        dicionario['ATMOSPHERE-LAYER-' + str(i + 1)] += "," + str(CH3Cl[i] / total_abundancia_camada)

    molweights = np.array(peso_molecular)
    abuns = np.array([CO2[0], N2[0], O2[0], H2O[0], CO[0], H2[0], C2H6[0], HCN[0], H2S[0], SO2[0], O3[0], CH4[0], 
                        N2O[0], NH3[0], CH3Cl[0]])
    
    # https://hitran.org/lbl/
    HITDICT = {"CO2": "HIT[2]", "N2" : "HIT[22]", "O2" : "HIT[7]", 
                "H2O": "HIT[1]", "CO": "HIT[5]", "H2": "HIT[45]", "C2H6": "HIT[27]",
                "HCN": "HIT[23]", "H2S": "HIT[31]", "SO2": "HIT[9]", "O3":"HIT[3]", 
                "CH4" :"HIT[6]", "N2O": "HIT[4]", "NH3" : "HIT[11]", "CH3Cl": "HIT[24]"}
    
    peso_molecular_medio = np.sum(molweights * abuns)
    dicionario['ATMOSPHERE-WEIGHT'] = peso_molecular_medio
    dicionario['ATMOSPHERE-NGAS'] = len(moleculas)
    dicionario['ATMOSPHERE-GAS'] = ','.join(moleculas)
    dicionario['ATMOSPHERE-TYPE'] = ','.join([HITDICT[i] for i in moleculas]) 
    dicionario['ATMOSPHERE-ABUN'] = '1,'*(len(moleculas)-1)+ '1'
    dicionario['ATMOSPHERE-UNIT'] = 'scl,'*(len(moleculas)-1)+ 'scl' 
    dicionario['ATMOSPHERE-LAYERS-MOLECULES'] = ','.join(moleculas)

            
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










        




