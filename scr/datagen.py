import sys
import os
import PT
import time
import msgpack
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.interpolate as si
import astropy.constants as const
from collections import OrderedDict
from dotenv import load_dotenv
load_dotenv()

# Os códigos a seguir são versões atualizadas e modificadas do INARA proposto 
# por Soboczenski, Frank, et al. (2018) https://arxiv.org/abs/1811.03390

def gerar_PT(log10kappa, log10gamma1, log10gamma2, alpha, beta, pmin, pmax, Rstar,
             Tstar, sma, surfgrav, Tint=0, nlayers=60):
    """
    Gera um perfil pressão-temperatura (PT) para os parâmetros de entrada fornecidos 
    usando o modelo de Line et al. (2013).

    Observação: Isso utiliza PT.py e reader.py do BART, o código Bayesian 
    Atmospheric Radiative Transfer, que possui uma licença de pesquisa 
    reproduzível de código aberto.

    Parâmetros
    ----------
    log10kappa : float
        Opacidade térmica do IR de Planck (em unidades cm^2/gr).
    log10gamma1 : float
        Razão da opacidade média de Planck de corrente visível para térmica.
    log10gamma2 : float
        Razão da opacidade média de Planck de corrente visível para térmica.
    alpha : float
        Partição de corrente visível (0.0-1.0).
    beta : float
        Um 'coringa' para albedo, emissividade e redistribuição dia-noite 
        (na ordem de unidade).
    pmin : float
        Pressão mínima no topo da atmosfera (em bares).
    pmax : float
        Pressão na superfície do planeta (em bares).
    Rstar : float
        Raio da estrela hospedeira (em raios solares).
    Tstar : float
        Temperatura da estrela hospedeira (em Kelvin).
    sma : float
        Semieixo maior do planeta (em UA).
    surfgrav : float
        Gravidade planetária a 1 bar de pressão (em cm/s^2).
    Tint : float, opcional
        Fluxo de calor interno do planeta (em Kelvin). O padrão é 0.
    nlayers : int, opcional
        Número de camadas na atmosfera. O padrão é 60.

    Retorna
    -------
    PTprof : ndarray
        Array 2D contendo a pressão (espaçada igualmente em log) e temperatura em 
        cada pressão.
    """   
    # Faixa de pressões igualmente espaçadas em log
    pressao = np.logspace(np.log10(pmax), np.log10(pmin), nlayers)

    # Gerar perfil de temperatura
    temp = PT.PT_line(pressao, log10kappa, log10gamma1, log10gamma2, alpha, beta, 
                      Rstar * const.R_sun.value, Tstar, Tint, sma * const.au.value, 
                      surfgrav, 'const')
    
    # Temperatura e pressão em um array
    PTprof = np.column_stack((pressao, temp))
    
    return PTprof


def calcular_nuvens(abun, PTprof, nT=400):
    """
    Calcula a quantidade de condensação que ocorrerá para uma espécie.

    Note que as nuvens são calculadas apenas dentro da faixa dos nossos dados.

    Espécies permitidas: H2O (255.9 K <= T <= 573 K), NH3 (164 K <= T <= 371.5 K)

    Fontes:
    H2O: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=4#Thermo-Phase
    NH3: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7664417&Units=SI&Mask=4#Thermo-Phase

    Parâmetros
    --------
    abun : float
        Abundância de H2O e NH3.
    PTprof : ndarray
        Perfil PT criado por gerar_PT().
    nT : int
        Número de pontos para calcular SVP.

    Retorna
    -------
    nuvens : ndarray
        Array 2D contendo a abundância de nuvens de H2O e NH3 em cada camada da 
        atmosfera.
    """
    # Arrays de parâmetros da equação de Antoine para Pressão de Vapor de Saturação (SVP)
    #                Tmim,  Tmax ,  A,       B,          C
    h2o = np.array([[255.9, 373. , 4.6543 , 1435.264,  -64.848], 
                    [273. , 303. , 5.40221, 1838.675,  -31.737], 
                    [304. , 333. , 5.20389, 1733.926,  -39.485], 
                    [334. , 363. , 5.0768 , 1659.793,  -45.854], 
                    [344. , 373. , 5.08354, 1663.125,  -45.622], 
                    [379. , 573. , 3.55959,  643.748, -198.043]])
    
    nh3 = np.array([[164. , 239.6, 3.18757,  506.713,  -80.78 ], 
                    [239.6, 371.5, 4.86886, 1113.928,  -10.409]])

    Th2o = np.linspace(np.amin(h2o[:,0]), np.amax(h2o[:,1]), num=nT)
    Tnh3 = np.linspace(np.amin(nh3[:,0]), np.amax(h2o[:,1]), num=nT)

    # Usar a equação de Antoine para calcular SVP
    # p[bar] = 10^(A - (B / (T[K] + C)))
    svph2o = np.zeros(nT)
    svpnh3 = np.zeros(nT)

    # Diferentes regimes de temperatura para H2O
    ir1 =  Th2o < h2o[1,0]
    ir2 = (Th2o >= h2o[1,0])              * (Th2o <= (h2o[2,0]+h2o[1,1])/2.)
    ir3 = (Th2o > (h2o[2,0]+h2o[1,1])/2.) * (Th2o <= (h2o[3,0]+h2o[2,1])/2.)
    ir4 = (Th2o > (h2o[3,0]+h2o[2,1])/2.) * (Th2o <   h2o[4,0])
    ir5 = (Th2o >= h2o[4,0]) * (Th2o<=h2o[3,1])
    ir6 = (Th2o >  h2o[3,1]) * (Th2o<=h2o[4,1])
    ir7 =  Th2o >= h2o[5,0]

    # Preencher SVP para H2O
    svph2o[ir1] =  10**(h2o[0,2] - (h2o[0,3] / (Th2o[ir1] + h2o[0,4])))
    svph2o[ir2] =  10**(h2o[1,2] - (h2o[1,3] / (Th2o[ir2] + h2o[1,4])))
    svph2o[ir3] =  10**(h2o[2,2] - (h2o[2,3] / (Th2o[ir3] + h2o[2,4])))
    svph2o[ir4] =  10**(h2o[3,2] - (h2o[3,3] / (Th2o[ir4] + h2o[3,4])))
    svph2o[ir5] = (10**(h2o[3,2] - (h2o[3,3] / (Th2o[ir5] + h2o[3,4]))) + \
                   10**(h2o[4,2] - (h2o[4,3] / (Th2o[ir5] + h2o[4,4])))) / 2.
    svph2o[ir6] =  10**(h2o[4,2] - (h2o[4,3] / (Th2o[ir6] + h2o[4,4])))
    svph2o[ir7] =  10**(h2o[5,2] - (h2o[5,3] / (Th2o[ir7] + h2o[5,4])))

    # Diferentes regimes de temperatura para NH3
    ir1 = Tnh3 <= nh3[0,1]
    ir2 = Tnh3 >  nh3[1,0]

    # Preencher SVP para NH3
    svpnh3[ir1] = 10**(nh3[0,2] - (nh3[0,3] / (Tnh3[ir1] + nh3[0,4])))
    svpnh3[ir2] = 10**(nh3[1,2] - (nh3[1,3] / (Tnh3[ir2] + nh3[1,4])))

    # Remover os valores nulos para evitar erro na interpolação
    Th2o   =   Th2o[svph2o!=0]
    Tnh3   =   Tnh3[svpnh3!=0]

    svph2o = svph2o[svph2o!=0]
    svpnh3 = svpnh3[svpnh3!=0]

    # Funções e índice para interpolação
    h2osvpinterp = si.interp1d(Th2o, svph2o)
    ih2o         = (PTprof[:,1] >= Th2o[0]) * (PTprof[:,1] <= Th2o[-1])

    nh3svpinterp = si.interp1d(Tnh3, svpnh3)
    inh3         = (PTprof[:,1] >= Tnh3[0]) * (PTprof[:,1] <= Tnh3[-1])

    # Interpolando os valores desejados
    laysvp          = np.zeros((len(PTprof[:,1]), 2))
    laysvp[ih2o, 0] = h2osvpinterp(PTprof[:,1][ih2o])
    laysvp[inh3, 1] = nh3svpinterp(PTprof[:,1][inh3])

    # Arrays que carregam as informações da nuvens
    nuvens    = np.zeros((len(PTprof), 2) )
    nuvensh2o = np.zeros( len(PTprof[:,0]))
    nuvensnh3 = np.zeros( len(PTprof[:,0]))

    # Indíces para determinar onde SVP é definido 
    ih2o = laysvp[:,0]!=0
    inh3 = laysvp[:,1]!=0

    # Calculando a abundância das nuvens
    nuvensh2o[ih2o] = abun[0] * ((abun[0] * PTprof[:,0])[ih2o] - laysvp[ih2o,0]) / \
                                 (abun[0] * PTprof[:,0])[ih2o]
    nuvensnh3[inh3] = abun[1] * ((abun[1] * PTprof[:,0])[inh3] - laysvp[inh3,1]) / \
                                 (abun[1] * PTprof[:,0])[inh3]
    
    # Nuvens em um array e remove negativos
    nuvens = np.column_stack((nuvensh2o, nuvensnh3))
    nuvens[nuvens < 0] = 0

    return nuvens


def gera_parametros(pmin=1e-6):
    """
    Essa função amostra parâmetros dentro de uma faixa dada. Alguns parâmetros
    influenciam outros.

    Parâmetros:
    ----------
    pmin : float, opcional
        Pressão mínima a ser considerada no modelo atmosférico. [bars]
        O padrão é aproximadamente a base da termosfera para a Terra.

    Retorna:
    -------
    tuple
        Uma tupla contendo os seguintes parâmetros:

        - starclass : str
            Classe estelar. (F, G, K ou M)
        - Tstar : float
            Temperatura estelar. [K]
        - Rstar : float
            Raio estelar. [R_sun]
        - dist : float
            Distância do sistema [AU]
        - sma : float
            Semieixo maior do planeta. [AU]
        - rplanet : float
            Raio planetário. [R_earth]
        - densidade : float
            Densidade planetária. [g/cm^3]
        - kappa : float
            Opacidade térmica IR de Planck [cm^2/g]
        - gamma1 : float
            Razão de opacidade média de Planck entre o fluxo visível e térmico.
        - gamma2 : float
            Razão de opacidade média de Planck entre o fluxo visível e térmico.
        - alpha : float
            Partição de fluxo visível (0.0--1.0).
        - beta : float
            Um 'apanhador geral' para albedo, emissividade e redistribuição dia-noite
            (da ordem da unidade).
        - surfpres : float
            Pressão 'superficial' planetária. [bars]
        - PTprof : array
            Perfil pressão-temperatura.
        - surftemp : float
            Temperatura 'superficial' planetária. [K]
        - mols : list
            Moléculas na atmosfera.
        - layabuns : list
            Abundância em cada camada.
        - abuns : list
            Abundâncias de cada molécula na atmosfera dadas em `mols'.
        - avgweight : list
            Peso molecular médio de cada molécula.
        - albedo : float
            Albedo planetário.
        - clouds : list
            Gerado do calcular_nuvens.
        - naero : int
            Número de aerosols.
        - aeros : str
            Nome do aerosol.
        - atype : int
            Sub tipo do aerosol.
        - aabun : list
            Abundância de aerossóis.
        - aunit : list
            Unidade das abundâncias.
        - asize : list
            Raio efetivo as partículas de aerossol.
        - nmax : int
            Ao realizar cálculos de aerossóis de dispersão, este parâmetro indica o número de pares 
            de n-fluxo.
        - lmax : int
            Ao realizar cálculos de aerossóis de dispersão, este parâmetro indica o número de 
            polinômios de Legendre de dispersão usados para descrever a função de fase.
        - longitude : float
            longitude sub solar em graus
        - latitude : float
            latitude sub solar em graus 
    """

    # Classe de estrelas consideradas
    sclass = ['U', 'G', 'K', 'M']
    starclass = sclass[np.random.randint(0, len(sclass))]

    # Gerando a temperatura e o raio para a estrela 

    # Fonte para temepratura: https://www.cfa.harvard.edu/~pberlind/atlas/htmls/note.html

    # Fonte para raio e temperatura para F, G, K: https://arxiv.org/pdf/1306.2974.pdf
    # Boyajian et al. 2013, ApJ, "Stellar Diameters and Temperatures III. ..."

    # Fonte para raio e temperatura para K, M: https://arxiv.org/pdf/1208.2431.pdf
    # Boyajian et al, 2012, ApJ, "Stellar Diameters and Temperatures II. ..."

    if starclass == 'U': # Isso é tipo F -- usa modelo de corpo negro
        Tstar = np.random.uniform(5900., 7200.)

        if Tstar > 6400:
            Rstar = np.random.uniform(1.40, 2.00)
        else:
            Rstar = np.random.uniform(1.10, 2.00)

    elif starclass=='G':
        Tstar = np.random.uniform(5300., 5900.)

        if Tstar > 5500:
            Rstar = np.random.uniform(0.90, 1.30)
        else:
            Rstar = np.random.uniform(0.80, 1.10)

    elif starclass=='K':
        Tstar = np.random.uniform(3800., 5300.)

        if Tstar > 5000:
            Rstar = np.random.uniform(0.70, 0.95)
        else:
            Rstar = np.random.uniform(0.60, 0.85)

    elif starclass=='M':
        Tstar = np.random.uniform(3000., 3800.)

        if Tstar < 3250:
            Rstar = np.random.uniform(0.14, 0.40)
        else:
            Rstar = np.random.uniform(0.30, 0.55)
    
    else:
        print("Deu algo muito errado aqui na geração de raio e massa.")
        sys.exit()
    
    # Colocando a distância
    if starclass=='M':
        dist = np.random.uniform(5. , 25.)
    else:
        dist = np.random.uniform(1.3, 15.)

    
    # Calculando a luminosidade estelar
    Lstar = const.sigma_sb.value * \
        4. * np.pi * (Rstar * const.R_sun.value)**2 * Tstar**4
    
    # Usando Tstar para encontrar os limites do semi-eixo maior
    # Fonte: Kopparapu et al 2013, ApJ, "Habitable Zones Around Main-sequence..."
    # S_eff = S_eff0 + aT_star + bT_star² + cT_star³ + dT_star⁴
    Slo = 1.7763 + 1.4335e-04 * (Tstar - 5780.)    + 3.3954e-09 * (Tstar - 5780.)**2 - \
                   7.6364e-12 * (Tstar - 5780.)**3 - 1.1950e-15 * (Tstar - 5780.)**4
    Shi = 0.3207 + 5.4471e-05 * (Tstar - 5780.)    + 1.5275e-09 * (Tstar - 5780.)**2 - \
                   2.1709e-12 * (Tstar - 5780.)**3 - 3.8282e-16 * (Tstar - 5780.)**4
    
    # Converter valores de S para semieixo maior [AU]
    smalo = (Lstar / const.L_sun.value / Slo)**0.5
    smahi = (Lstar / const.L_sun.value / Shi)**0.5

    # Gera o semieixo maior [AU]
    sma = np.random.uniform(smalo, smahi)

    # Gera um planeta que consegue segurar uma atmosfera
    podeseguraratm = False
    while podeseguraratm == False:
        # Gera um raio planetário [R_earth]
        # Marte próximo ao limite superior em planetas rochosos
        rplanet = np.random.uniform(0.5, 1.6)

        # Calcula a massa do raio usando a relação de Sotin et al 2007
        # "Mass-radius curve for extrasolar Earth-like planets and ocean planets"
        # Incluindo um fator de +- 2%
        if rplanet <= 1.:
            mplanet = (rplanet / 1.)**(1./0.306) * np.random.uniform(0.98, 1.02)
        elif rplanet > 1.:
            mplanet = (rplanet / 1.)**(1./0.274) * np.random.uniform(0.98, 1.02)
        else:
            pass

        # Calcula a densidade [g/cm^3], gravidade superficial [cm/s^2]
        density  = mplanet * const.M_earth.cgs.value /                      \
                   (4./3. * np.pi * (rplanet * const.R_earth.cgs.value)**3)
        surfgrav = const.G.cgs.value * mplanet * const.M_earth.cgs.value /  \
                   (rplanet * const.R_earth.cgs.value)**2

         # Calcule a velocidade de escape da superfície [km/s^2]
        vesc = (2. * const.G.value * mplanet * const.M_earth.value / \
                (rplanet * const.R_earth.value))**0.5                 / 1000.

        # Calculte a insolação do planeta Insol, em relação à Terra
        Insol = Lstar / const.L_sun.value / sma**2

        # Aproximação da equação da 'cosmic shoreline' em Zahnle & Catling 2016
        shoreline = np.log10(1e4 / 1e-6) / np.log10(70. / 0.2)

        # Calcula a insolação da costa, pinsol, valor para vesc do planeta
        # para ver se ele consegue manter uma atmosfera
        # Requer que a insolação real, Insol, seja menor que pinsol
        pinsol = 1e4 * (vesc / 70.)**shoreline
        if Insol < pinsol:
            podeseguraratm = True
        
    # Gera pressão superficial - o range é de Marte a Vênus,
    # favorecendo pressões na faixa de 0,1 a 10 bar
    # Norma truncada no intervalo de 0,1 a 10, média de 1,0, desvio padrão de 2,5
    # O limite superior verdadeiro é de aproximadamente 15 barras usando este método.
    mean = 1.0
    stdv = 2.5
    lo   = 0.1
    hi   = 90.
    a, b = (lo - mean) / stdv, (hi - mean) / stdv
    rv   = ss.truncnorm(a, b, loc=mean, scale=stdv)
    surfpres = rv.rvs()

    # Gera um array da pressão 
    press = np.logspace(np.log10(pmin), np.log10(surfpres), num=60)

    ## Gera o perfil atmosférico T(p)
    # Amostra log10(kappa), log10(gamma1), log10(gamma2), alpha, beta
    kappa  = np.random.uniform(-3.5, -2.0)
    gamma1 = np.random.uniform(-1.5,  1.1)
    gamma2 = np.random.uniform(-1.5,  0.)
    alpha  = np.random.uniform( 0.,   1.)

    # Cria beta de uma distribuição normal truncada limitada por [0,7, 1,1] com média de 0,95
    mean   = 0.95
    stdv   = 0.1
    lo     = 0.7
    hi     = 1.1
    a, b   = (lo - mean) / stdv, (hi - mean) / stdv
    rv     = ss.truncnorm(a, b, loc=mean, scale=stdv)
    beta   = rv.rvs()

    PTprof = gerar_PT(kappa, gamma1, gamma2, alpha, beta,
                   pmin, surfpres, Rstar, Tstar, sma, surfgrav, Tint=0, nlayers=60)

    # Extrair a temperatura de superfície
    surftemp = PTprof[0,1]

    # Moléculas que consideramos
    #         0      1      2     3     4      5      6     7     8      9
    mols = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O', 'CO', 'O3', 'SO2', 'NH3', 
            'C2H6', 'NO2']
    molweights = np.array([18.02, 44.01, 31.9988, 28.0134, 16.043, 44.013, 
                           28.011, 47.998, 64.06, 17.02, 28.05, 46.0055]) #g/mol

    # Gera a abundância molecular - ranges generosos
    abuns      = np.zeros(len(mols), dtype=np.float64)
    abuns[0]   = np.random.uniform(0., 0.1)                 # O gás H2O será <10%
    abuns[1:4] = np.random.uniform(0., 1., 3)               # CO2, O2, N2 pode ser qualquer quantidade
    abuns[4]   = np.random.uniform(0., 0.1)                 # 10% CH4 é MUITO
    abuns[5]   = np.random.uniform(0., 0.02)                # 2% N2O é MUITO
    abuns[6]   = np.random.uniform(0., 0.02)                # 2% CO é MUITO - e provavelmente um planeta morto
    abuns[7]   = np.random.uniform(0., 0.01 * abuns[2])     # O3 é no máximo 1% de O2, geralmente menos
    abuns[8]   = np.random.uniform(0., 0.02)                # 2% SO2 é MUITO
    abuns[9]   = np.random.uniform(0., 0.01)                # 1% NH3 é MUITO, reage rapidamente, etc.
    abuns[10]  = np.random.uniform(0., 0.01 * abuns[4])     # produto de CH4, então no máximo 1% disso
    abuns[11]  = np.random.uniform(0., 20e-6)               # civilização poluente 1000 vezes pior que os humanos

    # Normaliza
    abuns /= np.sum(abuns)

    # Calcula o peso molecular médio em in g/mol
    avgweight = np.sum(molweights * abuns)

    # Gera o albedo
    albedo = np.random.uniform(0.1, 0.8)

    # Gera a longitude sub solar em graus
    longitude = np.random.uniform(-360, 360)

    # Gera a latitude sub soolar em graus
    latitude = np.random.uniform(-90, 90)

    # Material condensado
    icloud = [0, 8] # Indica a espécie da nuvem, H2O e NH3
    clouds = calcular_nuvens(abuns[icloud], PTprof)

    # Repita para cada camada, para facilitar a transmissão da abundância de nuvens
    layabuns = np.broadcast_to(abuns, (PTprof.shape[0], len(abuns))).copy()

    # Nuvens e neblinas
    if np.any(clouds[:,0]!=0) and np.any(clouds[:,1]!=0): # Ambas nuvens de amônia e água
        layabuns[:,icloud] -= clouds                      # Remove o gás H2O e NH3
        naero = 2
        aeros = 'Water,Ammonia'
        atype = 'AFCRL_WATER_HRI[0.20um-0.03m],Ice_Martonchik_GSFC[0.19-200.00um]'
        aabun = '1,1'
        aunit = 'scl,scl'
        asize = '10,10' # um
        nmax = 8        # Parâmetros de espalhamento para nuvens
        lmax = 8
    
    elif np.any(clouds[:,0]!=0): # Apenas nuvem de água
        clouds = clouds[:,0]
        layabuns[:,0] -= clouds  # Remove o gás H2O
        naero = 1
        aeros = 'Water'
        atype = 'AFCRL_WATER_HRI[0.20um-0.03m]'
        aabun = '1'
        aunit = 'scl'
        asize = '10'    
        nmax = 8        
        lmax = 8

    elif np.any(clouds[:,1]!=0): # Apenas nuvem de amônia
        clouds = clouds[:,1]
        layabuns[:,1] -= clouds  # Remove o gás NH3
        naero = 1
        aeros = 'Ammonia'
        atype = 'Ice_Martonchik_GSFC[0.19-200.00um]'
        aabun = '1'
        aunit = 'scl'
        asize = '10'    
        nmax = 8        
        lmax = 8

    else: # Sem nuvens
        clouds = None
        naero  = 0
        aeros  = ''
        atype  = ''
        aabun  = ''
        aunit  = ''
        asize  = ''
        nmax   = 0
        lmax   = 0

    ## Fontes do tamanho das particulas:
        
    # H2O: https://www.atmos-meas-tech.net/10/2105/2017/amt-10-2105-2017.pdf
    #      https://www.goes-r.gov/products/ATBDs/baseline/Cloud_DCOMP_v2.0_no_color.pdf
    #      http://www.curry.eas.gatech.edu/currydoc/Liu_JGR108.pdf

    # NH3: https://www.sciencedirect.com/science/article/pii/0019103582901713
    #      http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.651.9301&rep=rep1&type=pdf

    return starclass, Tstar, Rstar, dist, sma, rplanet*const.R_earth.value/1000., density,  \
           kappa, gamma1, gamma2, alpha, beta,                                              \
           surfpres, PTprof, surftemp, mols, layabuns, abuns, avgweight, albedo,            \
           clouds, naero, aeros, atype, aabun, aunit, asize, nmax, lmax, longitude, latitude


def gera_config(nome_planeta, quer_nuvem):
    """
    Essa função gera o arquivo config com as informações geradas aleatoriamente.

    Parâmetros:
    ----------
    nome_planeta : str
        Nome do planeta simulado.
    quer_nuvem : bool
        Determina se a dispersão de nuvens será simulada ou não.

    Retorna:
    -------
    - config : dict
        O arquivo de configuração.
    """

    # HabEx default config
    habex = os.getenv('habex_config')

    # Abre o arquivo e carrega o objeto OrderedDict
    with open(habex, 'rb') as f:
        config = OrderedDict(msgpack.unpack(f, raw=False))

    # Gera os parâmetros (função anterior)
    starclass, Tstar, Rstar, dist, sma, rplanet, density,                         \
        kappa, gamma1, gamma2, alpha, beta,                                        \
            surfpres, PTprof, surftemp, mols, layabuns, abuns, avgweight, albedo,   \
                clouds, naero, aeros, atype, aabun, aunit, asize, nmax, lmax,        \
                    longitude, latitude = gera_parametros()
    
    # Dicionário com a lista das linhas HITRAN associadas
    HITDICT = {'H2O':'HIT[1]', 'CO2':'HIT[2]', 'O3':'HIT[3]', 'N2O':'HIT[4]',
               'CO':'HIT[5]', 'CH4':'HIT[6]', 'O2':'HIT[7]', 'NO':'HIT[8]',
               'SO2':'HIT[9]', 'NO2':'HIT[10]', 'NH3':'HIT[11]', 'N2':'HIT[22]',
               'HCN':'HIT[23]', 'C2H6':'HIT[27]', 'H2S':'HIT[31]'}
    
    ## Aqui vamos modificar o arquivo config

    # Informação do sistema
    config['OBJECT-NAME']     = nome_planeta        # Nome do planeta
    config['OBJECT-DIAMETER'] = 2. * rplanet        # Diâmetro do planeta [km]
    config['OBJECT-GRAVITY']  = density             # Densidade do planeta [g/cm3]
    config['OBJECT-STAR-DISTANCE']    = sma         # Semieixo maior [AU]
    config['OBJECT-STAR-TYPE']        = starclass   # Classe da estrela
    config['GEOMETRY-STELLAR-TYPE']   = starclass   # Classe da estrela de ocultação
    config['OBJECT-STAR-TEMPERATURE'] = Tstar       # Temperatura da estreka [K]
    config['GEOMETRY-STELLAR-TEMPERATURE'] = Tstar  # Temperatura da estreka de ocultação [K] 
    config['OBJECT-STAR-RADIUS']      = Rstar       # Stellar radius [Rsun]
    config['GEOMETRY-OBS-ALTITUDE']   = dist        # Distância do sistema
    config['OBJECT-SOLAR-LATITUDE']   = latitude    # Latitude sub solar em graus
    config['OBJECT-SOLAR-LONGITUDE']  = longitude   # Longitude sub solar em graus

    # Parâmetros atmosféricos
    config['ATMOSPHERE-PRESSURE'] = surfpres        # Pressão superficial do planeta [bars]
    config['ATMOSPHERE-WEIGHT']   = avgweight       # Peso molecular da atmosfera [g/mol]
    config['ATMOSPHERE-NGAS']     = len(mols)       # Número de gases na atmosfera
    config['ATMOSPHERE-GAS']  = ','.join(mols)      # Gases na atmosfera

    config['ATMOSPHERE-TYPE'] = ','.join([HITDICT[i] for i in mols])    # HITRAN lista
    config['ATMOSPHERE-ABUN'] = '1,'*(len(layabuns[0])-1)+ '1'          # Abundância dos gases
    config['ATMOSPHERE-UNIT'] = 'scl,'*(len(layabuns[0])-1)+ 'scl'      # Unidade para as abundâncias
    config['ATMOSPHERE-LAYERS-MOLECULES'] = ','.join(mols)              # Moléculas
    config['ATMOSPHERE-TAU'] = '0.07,'*(len(layabuns[0])-1)+ '0.07'     # Tempo de vida de fotodissociação das moléculas

    for i in range(len(PTprof)):                    # Informação da camada do modelo de atmosfera
        config['ATMOSPHERE-LAYER-'+str(i+1)] = ','.join(map(str, PTprof[i])) + \
                                         ',' + ','.join(map(str, list(layabuns[i].astype(str))))
        if quer_nuvem==True:
            if np.all(clouds != None):                                      # Adiciona informação de nuvens
                if i==0:
                    config['ATMOSPHERE-LAYERS-MOLECULES'] += ',' + aeros    # Nome das nuvens
                if len(clouds.shape) > 1:                                   # Abundância das nuvens
                    config['ATMOSPHERE-LAYER-'+str(i+1)] += ',' + \
                                               ','.join(map(str, list(clouds[i].astype(str))))
                else:
                    config['ATMOSPHERE-LAYER-'+str(i+1)] += ',' + str(clouds[i])
    
    # Informação da superfície
    config['SURFACE-TEMPERATURE'] = surftemp
    config['SURFACE-ALBEDO']      = albedo
    config['SURFACE-EMISSIVITY']  = 1. - albedo

    # Informação sobre as nuvens/aerosol
    if quer_nuvem==True:
        config['ATMOSPHERE-NAERO'] = naero
        config['ATMOSPHERE-AEROS'] = aeros
        config['ATMOSPHERE-ATYPE'] = atype
        config['ATMOSPHERE-AABUN'] = aabun
        config['ATMOSPHERE-AUNIT'] = aunit
        config['ATMOSPHERE-ASIZE'] = asize
        config['ATMOSPHERE-NMAX']  = nmax
        config['ATMOSPHERE-LMAX']  = lmax
    else:
        pass

    return config

## Aqui para baixo são funções que vão utilizar os resultados do config no PSG

def obter_dados_psg(psg, config, nome_planeta):
    """
    Esta função retorna um dicionário contendo as chaves 'header', 'spectrum' e 
    'duration_seconds' diretamente do PSG para um único planeta com parâmetros 
    gerados aleatoriamente.

    Parâmetros:
    ----------
    psg : PSG
        Instância do PSG do pacote pypsg.
    config : OrderedDict
        Dados de configuração.
    nome_planeta : str
        Nome do planeta.

    Retorna:
    -------
    dict
        Um dicionário contendo as informações do PSG para o planeta especificado.
    """
    ## Aqui vamos inserir esses dados no PSG
    sucesso = False # Bandeira para calcular os erros
    try:
        resultado = psg.run(config)
    except:
        print("Alguma coisa estranha aconteceu ao rodar no PSG!")
    
    if not sucesso:
        pass
    
    return resultado

def config_dict_to_str(config_dict):
    """
    Converte um dicionário em uma string formatada (Função do pypsg).

    Parâmetros:
    -----------
    config_dict : dict

    Retorna:
    --------
    str
        Uma string formatada contendo as configurações.
    """
    ret = []
    for key, value in config_dict.items():
        ret.append('<{}>{}'.format(key, str(value)))
    return '\n'.join(ret)


def gerar_conjunto_dados(psg, n_planetas, datagen_dir, nome_arquivo, quer_nuvem = False, verbose = True):
    """
    Esta função gera um conjunto de dados usando o PSG para um número especificado de planetas.

    Parâmetros:
    ----------
    psg : PSG
        Instância do PSG do pacote pypsg.
    n_planetas : int
        Número de planetas a serem gerados.
    datagen_dir : str
        Diretório onde os dados serão salvos.
    nome_arquivo : str
        Nome do arquivo que será salvo
    quer_nuvem : bool
        Indica se a nuvem deve ser incluída na simulação. O padrão é False.
    verbose : bool, opcional
        Indica se mensagens de saída devem ser impressas ou não. O padrão é True.
 
    Retorna:
    -------
    Esta função não retorna nenhum valor, mas gera e salva os dados dos planetas no formato Parquet.
    """
    if verbose:
        print("\n***** MODO DE GERAÇÃO DE DADOS *****")
    
    # Inicializa o contador de planetas gerados
    planetas = 0

    # DataFrame vazio para armazenar os dados
    dados_planetas = pd.DataFrame()

    # Loop para gerar os dados para cada planeta
    for i in range(n_planetas):
        planeta_id = f'exo{i}'  # Gera um identificador único para cada planeta
        
        if verbose:
            print(f'> Gerando exoplaneta {i+1}/{n_planetas}...')

        try:
            # Gera as configurações para o planeta atual
            config = gera_config(planeta_id, quer_nuvem=quer_nuvem)
                    
            # Chama a função para obter os dados do PSG para o planeta atual
            dicionario = obter_dados_psg(psg, config, planeta_id)
            espectro = dicionario['spectrum']   
            header = ['Wave/freq [um]', 'Total [I/F apparent albedo]', 'Noise', 'Stellar', 'Planet']
            espectro_df =  pd.DataFrame(espectro, columns=header)

            config = config_dict_to_str(config)

            # Convertendo os dados de configuração em um dicionário
            config_dict = {}
            for line in config.split('\n'):
                if line.strip():
                    key, value = line.split('>', 1)
                    key = key.strip('<')
                    config_dict[key] = [value.strip()] 

            # Convertendo o dicionário em um DataFrame
            config_df = pd.DataFrame(config_dict) 

            # Cria o diretório se não existir
            if not os.path.exists(os.path.join(datagen_dir, 'data/')):
                os.makedirs(os.path.join(datagen_dir, 'data/'))

            # Juntando os dois dataframes
            dados_planetas = pd.concat([dados_planetas, pd.concat([config_df, espectro_df.apply(lambda col: [list(col)], axis=0)], axis=1)])

            # Atualiza o contador de planetas gerados
            planetas += 1

        except Exception as e:
            print('> Erro no PSG:', e)
            print('> Pulando esse planeta...\n')
            continue
    
    if verbose:
      print('> Salvando arquivo...')

    # Salvando o arquivo!   
    dados_planetas.to_parquet(os.path.join(datagen_dir, 'data/', f'{nome_arquivo}.parquet'))

    if verbose:
      print('***** Arquivo salvo com sucesso! *****')

            

    











    









