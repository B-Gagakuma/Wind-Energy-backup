#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:43:32 2016

@author: gagakuma
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import *
from scipy.stats.distributions import norm
import chaospy as cp

from openmdao.api import Problem,ScipyOptimizer,Group , pyOptSparseDriver, IndepVarComp
#from florisse.OptimizationGroups import OptAEP
from florisse.floris import AEPGroup
from florisse.OptimizationGroups import OptAEP

import math
import time
import numpy as np
import pylab as plt

#import chaospy as cp
from matplotlib.patches import Circle
import cProfile
import random
import sys

if __name__ == "__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([972.913, 1552.628, 2157.548, 720.863, 1290.496, 1895.416, 2515.459, 3135.502, 468.813, 1033.405,
                         1623.202, 2238.204, 2853.206, 3483.331, 216.763, 776.314, 1356.029, 1955.908, 2570.91, 3185.912,
                         3800.914, 519.223, 1093.897, 1683.694, 2283.573, 2893.534, 3503.495, 257.091, 821.683, 1406.439,
                         1996.236, 2601.156, 3201.035, 3780.75, 0, 559.551, 1129.184, 1713.94, 2308.778, 2903.616, 3468.208,
                         287.337, 851.929, 1431.644, 2006.318, 2596.115, 3155.666, 3710.176, 20.164, 574.674, 1139.266,
                         1718.981, 2283.573, 2843.124, 297.419, 851.929, 1426.603, 1986.154, 1124.143, 1683.694])

    turbineY = np.array([10.082, 0, 20.164, 509.141, 494.018, 509.141, 539.387, 579.715, 1003.159,
                          977.954, 988.036, 1023.323, 1058.61, 1103.979, 1497.177, 1471.972, 1477.013,
                          1497.177, 1532.464, 1572.792, 1633.284, 1960.949, 1960.949, 1986.154, 2006.318,
                          2046.646, 2097.056, 2460.008, 2449.926, 2460.008, 2485.213, 2520.5, 2565.869,
                          2611.238, 2948.985, 2933.862, 2943.944, 2954.026, 2989.313, 3024.6, 3069.969,
                          3422.839, 3427.88, 3437.962, 3458.126, 3493.413, 3533.741, 3569.028, 3911.816,
                          3911.816, 3911.816, 3926.939, 3962.226, 3992.472, 4390.711, 4385.67, 4400.793,
                          4420.957, 4869.606, 4889.77])

    # define turbine size
    rotor_diameter = 126.4  # (m)

    # Scaling grid case
    # nRows = 3     # number of rows and columns in grid
    # spacing = 3.     # turbine grid spacing in diameters

    # Set up position arrays


    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    windFrequencies = np.array([0.01178129090726, 0.010995871513443, 0.009606283355151, 0.012123653207129, 0.010472258584231, 0.010069479407915, 0.009686839190414, 0.010009062531467,
                                0.010371563790152, 0.011217400060417, 0.015225052864767, 0.015627832041084, 0.015748665793979, 0.017057698117007, 0.019353539422012, 0.01419796596516,
                                0.012063236330682, 0.01202295841305, 0.013211156983184, 0.017460477293324, 0.017299365622797, 0.014399355553318, 0.007874332896989, 0, 2.01389588158292E-05,
                                0, 0.000342362299869, 0.003564595710402, 0.007189608297251, 0.008800725002517, 0.011358372772128, 0.014157688047528, 0.016695196858323, 0.016312556640822,
                                0.013170879065552, 0.01091531567818, 0.009485449602256, 0.010109757325546, 0.011881985701339, 0.012606988218709, 0.015889638505689, 0.017702144799114,
                                0.020420904239251, 0.022797301379519, 0.029543852582822, 0.030288994059007, 0.026986204813211, 0.022152854697412, 0.0212466015507, 0.018286174604773,
                                0.016614641023059, 0.019011177122143, 0.019051455039775, 0.016393112476085, 0.017621588963851, 0.016534085187796, 0.014459772429765, 0.014036854294633,
                                0.016574363105428, 0.015627832041084, 0.015345886617662, 0.017520894169772, 0.015970194340953, 0.015104219111872, 0.014520189306213, 0.013452824488974,
                                0.014781995770819, 0.013392407612527, 0.01105628838989, 0.010452119625415, 0.011620179236734, 0.01105628838989])    # probability of wind in this direction at this speed

    increment = 360./np.size(windFrequencies)
    windDirections = np.zeros(np.size(windFrequencies))
    for i in range(len(windFrequencies)):
        windDirections[i] = windDirections[i-1] + increment

    nDirections = len(windDirections)
    yaw = np.zeros(nTurbs*nDirections)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = 126.4            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    windSpeeds = np.array([7.44398937276582, 7.0374734676978, 7.09820487528301, 6.97582720257973, 7.09906387977456, 6.79384039857, 6.61386862442827, 6.88724003004024, 7.28011938604077,
                            7.36188716192998, 8.71116543234788, 8.48712054489046, 8.29266617193479, 8.20118583117946, 8.37368435340895, 7.58132127356313, 7.22083060467946, 6.84375121139698,
                            6.97104934411891, 7.00252433975894, 6.41311939314553, 5.27507213263217, 5.76890203958823, 0, 0, 0, 6.59969330523529, 6.61658124076272, 6.64733829530252,
                            7.47235805224027, 8.61805918044583, 8.75984697836557, 9.67706672825695, 9.32352191719384, 9.21667534912845, 8.71900777891329, 8.81730433408493, 8.88752434776097,
                            9.4922154283017, 9.93464650311183, 10.8422915888111, 11.0667932760273, 10.6112557711479, 10.9274723871776, 11.2891354914567, 10.6026281377234, 10.4529006549746,
                            9.73585063158273, 9.05802517543981, 8.78398165372577, 8.59346790556242, 8.73641023854555, 8.25111136888054, 7.92409540545332, 7.74262390808914, 7.88863072434713,
                            7.86205679337884, 7.90922676689813, 8.62487719148967, 8.41035461780799, 8.22082093235957, 8.3217415391023, 8.43023332279823, 8.5809564344, 8.84885079304578,
                            8.19056310577993, 8.26653822789919, 8.58177787396992, 8.0580183145337, 7.79176332186897, 7.41456822405026, 7.83003940838069])        # m/s

    air_density = 1.1716    # kg/m^3

    # set up problem
    prob = Problem()
    root = prob.root = Group()
    root.add('OptAEP', OptAEP(nTurbs, nDirections = 1, differentiable=True, use_rotor_components=False), promotes=['*'])
    root.add('yaw0', IndepVarComp('yaw0', np.zeros(nTurbs)), promotes=['*'])
    # prob = Problem()
    # root = prob.root = Group()
    # root.add(AEPGroup(nTurbs, nDirections=1))
    # initialize problem

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.add_objective('obj', scaler = 1.0e-8)
    prob.driver.opt_settings['Summary file'] = 'snopt_log.out'
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 1.0e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
    # prob.driver.opt_settings['Function precission'] = 1.0e-8

    # root.deriv_options['form'] = 'central'
    # root.deriv_options['step_size'] = 1.0e-5
    # root.deriv_options['step_calc'] = 'relative'
    # root.deriv_options['type'] = 'fd'


    # select design variables
    for direction_id in range(0, nDirections):
        prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

    prob.setup()

    # assign values to turbine states
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = np.array([wind_speed])
    prob['air_density'] = air_density
    prob['windDirections'] = np.array([windDirections])
    prob['windFrequencies'] = np.array([wind_frequency])
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12         # turns off cosine spread (just needs to be very large)
    # prob['floris_params:useWakeAngle'] = True
    # run the problem
    print 'start FLORIS run'
    tic = time.time()
    prob.run()
    #toc = time.time()

    ap =  prob['AEP']

    print prob['turbineX']

    # print the results
    #   print 'FLORIS calculation took %.06f sec.' % (toc-tic)
    print 'turbine X positions in wind frame (m): %s' % prob['turbineX']
    print 'turbine Y positions in wind frame (m): %s' % prob['turbineY']
    print 'yaw (deg) = ', prob['yaw0']
    print 'Effective hub velocities (m/s) = ', prob['wtVelocity0']
    print 'Turbine powers (kW) = ', (prob['wtPower0'])
    print 'wind farm power (kW): %s' % (prob['dir_power0'])
    print 'wind farm AEP for this direction and speed (kWh): %s' % prob['AEP']
    print 'Effective hub velocities (m/s) = ', prob['wtVelocity0']

    xbounds = [min(turbineX)/rotor_diameter, min(turbineX)/rotor_diameter, max(turbineX)/rotor_diameter, max(turbineX)/rotor_diameter, min(turbineX)/rotor_diameter]
    ybounds = [min(turbineY)/rotor_diameter, max(turbineY)/rotor_diameter, max(turbineY)/rotor_diameter, min(turbineY)/rotor_diameter, min(turbineX)/rotor_diameter]



    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()

    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    R = rotor_diameter/2.
    for i in range(nTurbs):
        x = prob['turbineX'][i]
        y = prob['turbineY'][i]
        theta = math.radians(prob['windDirections'][0]) + math.radians(prob['yaw0'][i])
        # theta = math.radians(225.)
        # theta = 0.
        xt = x-R*np.cos(theta)
        xb = x+R*np.cos(theta)
        yt = y+R*np.sin(theta)
        yb = y-R*np.sin(theta)
        plt.plot([xt,xb],[yt,yb],'b',linewidth=4)
    # plt.plot(np.array([x,x]),np.array([y-R,y+R]),'k',linewidth=4)
    ax.set_aspect('equal')
    # ax.set_xticks(np.arange(500, 1751, 250))
    # ax.set_yticks(np.arange(500, 1501, 250))
    ax.autoscale(tight=True)
    ax.axis([min(prob['turbineX'])-200,max(prob['turbineX'])+200,min(prob['turbineY'])-200,max(prob['turbineY'])+200])
    plt.title('optimized')
    plt.axis('off')


    plt.show()


np.savetxt('deterministic.txt', np.c_[ap]) #header="turbineX, turbineY, turbineZ")
# np.savetxt('dt_XYZdt_%s.txt'%spacing, np.c_[prob['d_paramH1'], prob['d_paramH2'], prob['t_paramH1'], prob['t_paramH2'],], header="d_paramH1, t_paramH1")
