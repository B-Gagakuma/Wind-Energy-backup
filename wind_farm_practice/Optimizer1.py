from pyoptsparse import Optimization, SNOPT, pyOpt_solution
import numpy as np
import matplotlib.pyplot as plt
import _floris

from scipy.stats.distributions import norm
from pyDOE import *
import os
from sys import argv

# from joblib import Parallel, delayed


# PARAMETERS USED FOR TUNING THE FLORIS MODEL
def floris_parameters():
    pP = 1.88
    ke = 0.065
    ke_in = ke
    keCorrArray = 0.0
    keCorrCT = 0.0
    Region2CT = 4.0*(1.0/3.0)*(1.0-(1.0/3.0))
    kd = 0.15
    # me = np.array([-0.5, 0.22, 1.0]) #np.conp.sine off
    me = np.array([-0.5, 0.3, 1.0]) #np.conp.sine on

    initialWakeDisplacement = -4.5
    initialWakeAngle = 1.5
    shearzh = 90.

    baselineCT = 4./3.*(1.-1./3.)

    keCorrTI = 0.0
    baselineTI = 0.045

    keCorrHR = 0.0 # neutral, with heating rate 0, is baseline
    keCorrHRTI = 0.0
    keSaturation = 0.0

    kdCorrYawDirection = 0.0

    MU = np.array([0.5, 1.0, 5.5])
    shearCoefficientAlpha = 0.16
    nSamples = 1
    wspositionxyzw = np.zeros([3, nSamples])

    CTcorrected = True # CT factor already corrected by CCBlade calculation (approximately factor np.cos(yaw)^2)
    CPcorrected = True # CP factor already corrected by CCBlade calculation (assumed with approximately factor np.cos(yaw)^3)

    axialIndProvided = True # CT factor already corrected by CCBlade calculation (approximately factor np.cos(yaw)^2)
    a_in = axialIndProvided
    a = a_in
    # useWakeAngle = False #np.conp.sine off
    useWakeAngle = True #np.conp.sine on

    ad = -4.5
    bd = -0.01

    useaUbU = True
    aU = 5.0 # degrees
    bU = 1.66

    adjustInitialWakeDiamToYaw = False

    FLORISoriginal = False # override all parameters and use FLORIS as original in first Wind Energy paper

    # np.cos_spread = 1E12 # spread of np.conp.sine smoothing factor (percent of sum of wake and rotor radii) np.conp.sine off
    np.cos_spread = 2.0 # spread of np.conp.sine smoothing factor (percent of sum of wake and rotor radii) np.conp.sine on

    return pP, ke, ke_in, a_in, keCorrArray, keCorrCT, Region2CT, kd, me, initialWakeDisplacement, initialWakeAngle, baselineCT, keCorrTI, baselineTI, keCorrHR, keCorrHRTI, keSaturation, kdCorrYawDirection, MU, CTcorrected, CPcorrected, axialIndProvided, useWakeAngle, bd, useaUbU, aU, bU, adjustInitialWakeDiamToYaw, FLORISoriginal, np.cos_spread, wspositionxyzw, shearCoefficientAlpha, shearzh



# POWER CALCULATION FROM FLORIS MODEL
def floris_power(turbineXw, turbineYw, rotorDiameter, Vinf, rho, generator_efficiency, yawDeg, hubheight):

    pP, ke, ke_in, a_in, keCorrArray, keCorrCT, Region2CT, kd, me, initialWakeDisplacement, initialWakeAngle, baselineCT, keCorrTI, baselineTI, keCorrHR, keCorrHRTI, keSaturation, kdCorrYawDirection, MU, CTcorrected, CPcorrected, axialIndProvided, useWakeAngle, bd, useaUbU, aU, bU, adjustInitialWakeDiamToYaw, FLORISoriginal, np.cos_spread, wspositionxyzw, shearCoefficientAlpha, shearzh = floris_parameters()

    p_near0 = 1.0

    # FOR REFERENCE:
    axialInduction = np.ones_like(turbineXw)*1.0/3.0
    Ct = 4.0*axialInduction*(1.0-axialInduction)
    Cp = np.ones_like(turbineXw)*(0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2))

    wtvelocity,wsarray,wakeCentersYT_vec,wakeDiametersT_vec,wakeOverlapTRel_vec = _floris.floris(turbineXw,turbineYw,yawDeg,rotorDiameter,hubheight,Vinf,Ct,axialInduction,ke,kd,me,initialWakeDisplacement,bd,MU,aU,bU,initialWakeAngle,np.cos_spread,keCorrCT,Region2CT,keCorrArray,useWakeAngle,adjustInitialWakeDiamToYaw,axialIndProvided,useaUbU,wspositionxyzw,shearCoefficientAlpha,shearzh)

    wt_power = generator_efficiency*(0.5*rho*((np.pi/4)*rotorDiameter**2)*Cp*wtvelocity**3)/1000.
    power = np.sum(wt_power)

    return wtvelocity, wt_power, power



# TURBINE SEPARATION FUNCTION
# def sep_func(loc):
#     global rotorDiameter
#
#     space = 2. # rotor diameters apart
#
#     n = np.size(loc)/2
#     x = loc[0:n]
#     y = loc[n:]
#     sep = np.zeros((n-1)*n/2)
#
#     k = 0
#     for i in range(0, n):
#         for j in range(i+1, n):
#             sep[k] = sqrt((x[j]-x[i])**2+(y[j]-y[i])**2)
#             k += 1
#
#     return sep - space*rotorDiameter[0]




def obj_func(xdict):
    global turbineX
    global turbineY
    global rotorDiameter
    global hubheight
    global generator_efficiency
    global rho
    global windroseDirections
    global windFrequencies
    global funcs

    # x = xdict['xvars']
    # y = xdict['yvars']
    yaw_i = xdict['yaw']
    nturbines = 60
    nwind = np.size(nSamp)

    yaw = np.zeros(nturbines)
    k = 0
    for i in range(nwind):
        for j in range(nturbines):
            yaw[i,j] = yaw_i[k]
            k += 1

    funcs = {}
    Vin = [8]#np.array([7.44398937276582, 7.0374734676978, 7.09820487528301, 6.97582720257973, 7.09906387977456, 6.79384039857, 6.61386862442827, 6.88724003004024, 7.28011938604077,
                    # 7.36188716192998, 8.71116543234788, 8.48712054489046, 8.29266617193479, 8.20118583117946, 8.37368435340895, 7.58132127356313, 7.22083060467946, 6.84375121139698,
                    # 6.97104934411891, 7.00252433975894, 6.41311939314553, 5.27507213263217, 5.76890203958823, 0, 0, 0, 6.59969330523529, 6.61658124076272, 6.64733829530252,
                    # 7.47235805224027, 8.61805918044583, 8.75984697836557, 9.67706672825695, 9.32352191719384, 9.21667534912845, 8.71900777891329, 8.81730433408493, 8.88752434776097,
                    # 9.4922154283017, 9.93464650311183, 10.8422915888111, 11.0667932760273, 10.6112557711479, 10.9274723871776, 11.2891354914567, 10.6026281377234, 10.4529006549746,
                    # 9.73585063158273, 9.05802517543981, 8.78398165372577, 8.59346790556242, 8.73641023854555, 8.25111136888054, 7.92409540545332, 7.74262390808914, 7.88863072434713,
                    # 7.86205679337884, 7.90922676689813, 8.62487719148967, 8.41035461780799, 8.22082093235957, 8.3217415391023, 8.43023332279823, 8.5809564344, 8.84885079304578,
                    # 8.19056310577993, 8.26653822789919, 8.58177787396992, 8.0580183145337, 7.79176332186897, 7.41456822405026, 7.83003940838069])


    # Calculate the AEP of the wind farm
    power_dir = np.zeros(nwind)
    veleff = np.zeros(nturbines)


    windDirections = np.zeros_like(R)
    for d in range(nwind):
        # Adjusting coordinate system
        windDirections[d] = 270. - R[d]
        if windDirections[d] < 0.:
            windDirections[d] += 360.
        windDirectionRad = np.pi*windDirections[d]/180.0
        xw = turbineX*np.cos(-windDirectionRad) - turbineY*np.sin(-windDirectionRad)
        yw = turbineX*np.sin(-windDirectionRad) + turbineY*np.cos(-windDirectionRad)

        _, _, power = floris_power(xw, yw, rotorDiameter, Vin[d], rho, generator_efficiency, yaw[d], hubheight)

        power_dir[d] = power*windFrequencies[d]
    AEP = sum(power_dir)*8760.

    funcs['obj'] = (-1.*AEP)

    # print AEP

    # sep = sep_func(np.append(x,y))
    # funcs['sep'] = sep

    fail = False

    return funcs, fail


## Main
if __name__ == "__main__":

    global rotorDiameter
    global hubheight
    global generator_efficiency
    global rho
    global windroseDirections
    global windFrequencies
    global funcs

    nturbines = 60
    rotorDiameter = np.zeros(nturbines)
    # hubheight = np.zeros(nturbines)
    generator_efficiency = np.zeros(nturbines)
    axialInduction = np.zeros(nturbines)
    Cp = np.zeros(nturbines)
    Ct = np.zeros(nturbines)
    yaw = np.zeros(nturbines)
    ## Wind farm data

    farmname = 'Amalia'
    # farmname = 'Rosiere'

    # original example case
    # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
    # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m
    # obs = np.array([[800.,1500.,2.],[2300.,1500.,2.]])
    # xlow = np.ones(np.size(turbineX))*min(turbineX)
    # xupp = np.ones(np.size(turbineX))*max(turbineX)
    # ylow = np.ones(np.size(turbineY))*min(turbineY)
    # yupp = np.ones(np.size(turbineY))*max(turbineY)

    if farmname == 'Amalia':
        turbineX = np.array([972.913, 1552.628, 2157.548, 720.863, 1290.496, 1895.416, 2515.459, 3135.502, 468.813, 1033.405,
                             1623.202, 2238.204, 2853.206, 3483.331, 216.763, 776.314, 1356.029, 1955.908, 2570.91, 3185.912,
                             3800.914, 519.223, 1093.897, 1683.694, 2283.573, 2893.534, 3503.495, 257.091, 821.683, 1406.439,
                             1996.236, 2601.156, 3201.035, 3780.75, 0, 559.551, 1129.184, 1713.94, 2308.778, 2903.616, 3468.208,
                             287.337, 851.929, 1431.644, 2006.318, 2596.115, 3155.666, 3710.176, 20.164, 574.674, 1139.266,
                             1718.981, 2283.573, 2843.124, 297.419, 851.929, 1426.603, 1986.154, 1124.143, 1683.694])

        turbineY = np.array([[10.082, 0, 20.164, 509.141, 494.018, 509.141, 539.387, 579.715, 1003.159,
                              977.954, 988.036, 1023.323, 1058.61, 1103.979, 1497.177, 1471.972, 1477.013,
                              1497.177, 1532.464, 1572.792, 1633.284, 1960.949, 1960.949, 1986.154, 2006.318,
                              2046.646, 2097.056, 2460.008, 2449.926, 2460.008, 2485.213, 2520.5, 2565.869,
                              2611.238, 2948.985, 2933.862, 2943.944, 2954.026, 2989.313, 3024.6, 3069.969,
                              3422.839, 3427.88, 3437.962, 3458.126, 3493.413, 3533.741, 3569.028, 3911.816,
                              3911.816, 3911.816, 3926.939, 3962.226, 3992.472, 4390.711, 4385.67, 4400.793,
                              4420.957, 4869.606, 4889.77]])

        # print obs.ravel()
        windFrequencies = [1.]#np.array([0.01178129090726, 0.010995871513443, 0.009606283355151, 0.012123653207129, 0.010472258584231, 0.010069479407915, 0.009686839190414, 0.010009062531467,
        #                             0.010371563790152, 0.011217400060417, 0.015225052864767, 0.015627832041084, 0.015748665793979, 0.017057698117007, 0.019353539422012, 0.01419796596516,
        #                             0.012063236330682, 0.01202295841305, 0.013211156983184, 0.017460477293324, 0.017299365622797, 0.014399355553318, 0.007874332896989, 0, 2.01389588158292E-05,
        #                             0, 0.000342362299869, 0.003564595710402, 0.007189608297251, 0.008800725002517, 0.011358372772128, 0.014157688047528, 0.016695196858323, 0.016312556640822,
        #                             0.013170879065552, 0.01091531567818, 0.009485449602256, 0.010109757325546, 0.011881985701339, 0.012606988218709, 0.015889638505689, 0.017702144799114,
        #                             0.020420904239251, 0.022797301379519, 0.029543852582822, 0.030288994059007, 0.026986204813211, 0.022152854697412, 0.0212466015507, 0.018286174604773,
        #                             0.016614641023059, 0.019011177122143, 0.019051455039775, 0.016393112476085, 0.017621588963851, 0.016534085187796, 0.014459772429765, 0.014036854294633,
        #                             0.016574363105428, 0.015627832041084, 0.015345886617662, 0.017520894169772, 0.015970194340953, 0.015104219111872, 0.014520189306213, 0.013452824488974,
        #                             0.014781995770819, 0.013392407612527, 0.01105628838989, 0.010452119625415, 0.011620179236734, 0.01105628838989])

        for turbI in range(0, nturbines):
            rotorDiameter[turbI] = 126.4            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generator_efficiency[turbI] = 1.0     #0.944
            for i in range(len(yaw)):
                yaw[turbI] = yaw[i]     # deg.


        # yaw = np.zeros(60)


        rho = 1.1716    # kg/m^3
        hubheight = np.ones(nturbines)*140.    # m
        rotorDiameter = np.ones(nturbines)*126.4     # m

        # xlow = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        # xupp = np.array([1200.,1200.,1200.,1200.,1200.,1200.,1200.,1200.,1200.,1200.,1200.,1200.])
        # ylow = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        # yupp = np.array([1650.,1650.,1650.,1650.,1650.,1650.,1650.,1650.,1650.,1650.,1650.,1650.])


    # increment = 360./np.size(windFrequencies)
    # windroseDirections = np.zeros(np.size(windFrequencies))
    # for i in range(len(windFrequencies)):
    #     windroseDirections[i] = windroseDirections[i-1] + increment
    # mu, sigma = 0., 2. # mean and standard deviation
    nSamp = 1500
    # L = np.random.normal(mu, sigma, nSamp)
    lhd = lhs(1, samples=nSamp, criterion='center')
    L = norm(loc=0, scale=2).ppf(lhd)
    # R = windroseDirections + L
    windroseDirections = np.array([45.]) + L

    nturbines = np.size(turbineX)
    nwind = np.size(nSamp)
    input = {'yaw':yaw}
    funcs, _ =obj_func(input)
    AEP = funcs['obj']*-1
    print AEP

# ## Optimization

    for i in range(nSamp):

        optProb[i] = Optimization('Wind_Farm_AEP', obj_func)
        optProb.addObj('obj')[i]

        optProb.addVarGroup('yaw', nturbines, 'c', lower=-30., upper=30., value=yaw)[i]
        # optProb.addVarGroup('xvars', nturb, 'c', lower=xlow, upper=xupp, value=turbineX)
        # optProb.addVarGroup('yvars', nturb, 'c', lower=ylow, upper=yupp, value=turbineY)

        # num_cons_sep = (nturb-1)*nturb/2
        # optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)

        opt = SNOPT()
        opt.setOption('Scale option',0)
        # opt.setOption('Iterations limit',1000000)
        # opt.setOption('Print file','/Users/ning1/Dropbox/Acoustics/OpenMDAO_hubvels/ParetoLissett/SNOPT_print_SPL'+str(SPLlim)+'.out')
        # opt.setOption('Summary file','/Users/ning1/Dropbox/Acoustics/OpenMDAO_hubvels/ParetoLissett/SNOPT_summary_SPL'+str(SPLlim)+'.out')
        res = np.mean(opt(optProb))  #, sens=None)
    print res

    pow = np.array(-1*res.fStar)
    yaw = res.xStar['yaw']
    # xf = res.xStar['xvars']*100.
    # yf = res.xStar['yvars']*100.


    print 'Wind Directions:',windroseDirections
    print 'AEP:',pow,'kWh'
    print 'Yaw:',yaw,'degrees'
#     # print 'X-locations (initial):',turbineX
#     # print 'X-locations (final):',xf
#     # print 'Y-locations (initial):',turbineY
#     # print 'Y-locations (final):',yf
