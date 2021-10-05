#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:43:32 2016

@author: gagakuma
"""

import numpy as np
import matplotlib.pyplot as plt


from openmdao.api import Problem,ScipyOptimizer,Group , pyOptSparseDriver,IndepVarComp
from florisse.floris import AEPGroup
from florisse.OptimizationGroups import OptAEP
from florisse.GeneralWindFarmComponents import MeanAEP, yawCorrect

import math
import time
import numpy as np
import pylab as plt

if __name__ == "__main__":

    mu, sigma = 0., 2. # mean and standard deviation
    nSamp = 100
    s = np.random.normal(mu, sigma, nSamp)

    # define turbine size
    rotor_diameter = 126.4  # (m)

    # Scaling grid case
    nRows = 3     # number of rows and columns in grid
    spacing = 3.     # turbine grid spacing in diameters

    # Set up position arrays
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = 126.4            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0     #0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    windSpeeds = np.array([6.53163342, 6.11908394, 6.13415514, 6.0614625,  6.21344602,
                                5.87000793, 5.62161519, 5.96779107, 6.33589422, 6.4668016,
                                7.9854581,  7.6894432,  7.5089221,  7.48638098, 7.65764618,
                                6.82414044, 6.36728201, 5.95982999, 6.05942132, 6.1176321,
                                5.50987893, 4.18461796, 4.82863115, 0.,         0.,         0.,
                                5.94115843, 5.94914252, 5.59386528, 6.42332524, 7.67904937,
                                7.89618066, 8.84560463, 8.51601497, 8.40826823, 7.89479475,
                                7.86194762, 7.9242645,  8.56269962, 8.94563889, 9.82636368,
                               10.11153102, 9.71402212, 9.95233636,  10.35446959, 9.67156182,
                                9.62462527, 8.83545158, 8.18011771, 7.9372492,  7.68726143,
                                7.88134508, 7.31394723, 7.01839896, 6.82858346, 7.06213432,
                                7.01949894, 7.00575122, 7.78735165, 7.52836352, 7.21392201,
                                7.4356621,  7.54099962, 7.61335262, 7.90293531, 7.16021596,
                                7.19617087, 7.5593657,  7.03278586, 6.76105501, 6.48004694,
                                6.94716392])        # m/s
    air_density = 1.1716    # kg/m^3
    # wind_direction = 240    # deg (N = 0 deg., using direction FROM, as in met-mast data)



    #   print wind_direction
    wind_frequency = np.array([1.17812570e-02, 1.09958570e-02, 9.60626600e-03, 1.21236860e-02,
                               1.04722450e-02, 1.00695140e-02, 9.68687400e-03, 1.00090550e-02,
                               1.03715390e-02, 1.12172280e-02, 1.52249700e-02, 1.56279300e-02,
                               1.57488780e-02, 1.70577560e-02, 1.93535770e-02, 1.41980570e-02,
                               1.20632100e-02, 1.20229000e-02, 1.32111160e-02, 1.74605400e-02,
                               1.72994400e-02, 1.43993790e-02, 7.87436000e-03, 0.00000000e+00,
                               2.01390000e-05, 0.00000000e+00, 3.42360000e-04, 3.56458900e-03,
                               7.18957000e-03, 8.80068000e-03, 1.13583200e-02, 1.41576700e-02,
                               1.66951900e-02, 1.63125500e-02, 1.31709000e-02, 1.09153300e-02,
                               9.48553000e-03, 1.01097900e-02, 1.18819700e-02, 1.26069900e-02,
                               1.58895900e-02, 1.77021600e-02, 2.04208100e-02, 2.27972500e-02,
                               2.95438600e-02, 3.02891700e-02, 2.69861000e-02, 2.21527500e-02,
                               2.12465500e-02, 1.82861400e-02, 1.66147400e-02, 1.90111800e-02,
                               1.90514500e-02, 1.63932050e-02, 1.76215200e-02, 1.65341460e-02,
                               1.44597600e-02, 1.40370300e-02, 1.65745000e-02, 1.56278200e-02,
                               1.53459200e-02, 1.75210100e-02, 1.59702700e-02, 1.51041500e-02,
                               1.45201100e-02, 1.34527800e-02, 1.47819600e-02, 1.33923300e-02,
                               1.10562900e-02, 1.04521380e-02, 1.16201970e-02, 1.10562700e-02])

    # probability of wind in this direction at this speed
    nDirections = len(windSpeeds)
    windDirections = np.linspace(0.,360.-360./nDirections, nDirections)   # deg (N = 0 deg., using direction FROM, as in met-mast data)

    AEPsolved = np.zeros(nDirections)

    for k in range(nDirections):
        R = windDirections[k] + s  #windDirections

        # print(R)

        # set up problem

        prob = Problem()
        root = prob.root = Group()
        for direction_id in range(0, nDirections):
            root.add('yaw%i' %direction_id, IndepVarComp('yaw%i' %direction_id, np.ones(nTurbs)*0.), promotes=['*'])
        for i in range(nSamp):
            root.add("AEP_sample%s" %i, AEPGroup(nTurbs, nDirections = 1, differentiable = True))
        root.add("Mean_AEP", MeanAEP(nSamp), promotes = ["MeanAEP"])
        for i in range(nSamp):
            root.add('yawCorrect%s'%i, yawCorrect(nTurbs), promotes=['meanDirection','inputYaw'])

        root.connect('yaw0','inputYaw')
        for i in range(nSamp):
            root.connect("AEP_sample%s.AEP"%i, 'Mean_AEP.AEP%s'%i)
            root.connect('yawCorrect%s.globalYaw'%i,'AEP_sample%s.yaw0'%i)

        #
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('MeanAEP', scaler = 1.0e-8)
        prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
        prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
        prob.driver.opt_settings['Print file'] = 'printFile.out'

        prob.driver.add_desvar('yaw0', lower=-30.0, upper=30.0, scaler=1)
        prob.setup()

        prob['meanDirection'] = windDirections[k]

        for i in range(nSamp):
            prob['yawCorrect%s.sampleDirection'%i] = R[i]
            prob['AEP_sample%s.turbineX'%i] = turbineX
            prob['AEP_sample%s.turbineY'%i] = turbineY
            prob['AEP_sample%s.rotorDiameter'%i] = rotorDiameter
            prob['AEP_sample%s.axialInduction'%i] = axialInduction
            prob['AEP_sample%s.generatorEfficiency'%i] = generatorEfficiency
            prob['AEP_sample%s.windSpeeds'%i] = np.array([windSpeeds[i]])
            prob['AEP_sample%s.air_density'%i] = air_density
            prob['AEP_sample%s.windDirections'%i] = np.array([R[i]])
            prob['AEP_sample%s.windFrequencies'%i] = np.array([wind_frequency[i]])
            prob['AEP_sample%s.Ct_in'%i] = Ct
            prob['AEP_sample%s.Cp_in'%i] = Cp
            prob['AEP_sample%s.floris_params:cos_spread'%i] = 1E12         # turns off cosine spread (just needs to be very large)
            # prob['floris_params:useWakeAngle'] = True
            # run the problem

        print 'start FLORIS run'
        tic = time.time()
        prob.run()
        toc = time.time()
        AEPsolved[k] = prob['MeanAEP']

    print 'AEPsolved: %s' %AEPsolved
    # print 'MEAN AEP: ', prob['MeanAEP']
    # print 'YAW: ', prob['yaw0']
    # print 'Time to run: ', toc-tic
    # print 'windSpeeds: ', prob['AEP_sample0.windSpeeds']
    # print 'windDirections: ', prob['AEP_sample0.windDirections']
    # print 'windFrequencies: ', prob['AEP_sample0.windFrequencies']
