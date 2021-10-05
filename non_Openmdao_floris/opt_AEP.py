from pyoptsparse import Optimization, SNOPT, pyOpt_solution, NSGA2
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from non_openmdao_floris import AEP_obj
from position_constraints import SpacingConstraint, circularBoundary
from FLORISSE3D.setupOptimization import amaliaRose


def obj_func(xdict):
    global rotorDiameter
    global turbineZ
    global ratedPower
    global windDirections
    global windSpeeds
    global windFrequencies
    global shearExp
    global minSpacing
    global circle_radius

    turbineX = xdict['xvars']*1.
    turbineY = xdict['yvars']*1.

    funcs = {}

    AEP = AEP_obj(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
    funcs['obj'] = AEP/1.E8

    funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=minSpacing)/1.E5
    funcs['bound'] = circularBoundary(turbineX, turbineY, circle_radius)/100.

    fail = False

    return funcs, fail


## Main
if __name__ == "__main__":

    global rotorDiameter
    global turbineZ
    global ratedPower
    global windDirections
    global windSpeeds
    global windFrequencies
    global shearExp
    global minSpacing
    global circle_radius

    """grid setup"""
    nRows = 4
    nTurbines = nRows**2
    loc_array = np.arange(nRows)/2.
    loc_array = loc_array - max(loc_array)/2.
    turbineX = np.zeros(nTurbines)
    turbineY = np.zeros(nTurbines)
    for i in range(nRows):
        turbineX[i*nRows:nRows*(i+1)] = loc_array[i]
        for j in range(nRows):
            turbineY[i*nRows+j] = loc_array[j]
    print turbineX

    circle_radius = (abs(loc_array[0])*1000.)*np.sqrt(2)

    turbineZ = np.ones(nTurbines)*100.
    rotorDiameter = np.ones(nTurbines)*100.
    ratedPower = np.ones(nTurbines)*3500.


    # windSpeeds = np.array([8.,5.,10.,14.])
    # nDirections = len(windSpeeds)
    # windDirections = np.linspace(0.,360.-360./float(nDirections),nDirections)
    # windFrequencies = np.array([0.1,0.2,0.3,0.4])
    # windFrequencies = windFrequencies/sum(windFrequencies)
    #
    # windSpeeds = np.array([10.])
    # windDirections = np.array([0.])
    # windFrequencies = np.array([1.])

    windDirections, windFrequencies, windSpeeds = amaliaRose(30)

    shearExp = 0.15
    minSpacing = 2.0

    input = {'xvars':turbineX*1000.,'yvars':turbineY*1000.}
    funcs,_ = obj_func(input)
    print 'AEP start: ', -funcs['obj']
    # print funcs['sep']
    # print funcs['bound']

    """Optimization"""
    optProb = Optimization('Wind_Farm_AEP', obj_func)
    optProb.addObj('obj')


    optProb.addVarGroup('xvars', nTurbines, 'c', lower=None, upper=None, value=turbineX)
    optProb.addVarGroup('yvars', nTurbines, 'c', lower=None, upper=None, value=turbineY)

    num_cons_sep = (nTurbines-1)*nTurbines/2
    optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
    optProb.addConGroup('bound', nTurbines, lower=0., upper=None)

    # opt = NSGA2()
    # opt.setOption('maxGen', 200)
    # opt.setOption('PopSize', 15*(2*nturb+nrpm))
    # opt.setOption('pMut_real', 0.01)
    # opt.setOption('pCross_real', 1.0)
    opt = SNOPT()
    opt.setOption('Scale option',0)
    opt.setOption('Iterations limit',1000000)

    opt.setOption('Print file','Print.out')
    opt.setOption('Summary file','Summary.out')

    res = opt(optProb)#, sens=None)

    xf = res.xStar['xvars']
    yf = res.xStar['yvars']

    input = {'xvars':xf,'yvars':yf}
    funcs,_ = obj_func(input)
    print 'AEP opt: ', -funcs['obj']

    for i in range(nTurbines):
        circ_start = plt.Circle((turbineX[i]*1000.,turbineY[i]*1000.), rotorDiameter[i]/2., linestyle='dashed',facecolor="none",edgecolor="black")
        circ_opt = plt.Circle((xf[i]*1.,yf[i]*1.), rotorDiameter[i]/2.,facecolor="red",edgecolor="red",alpha=0.2)
        plt.gca().add_patch(circ_start)
        plt.gca().add_patch(circ_opt)
    circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed',facecolor='none',label='Boundaries')
    plt.gca().add_patch(circ_outer)
    plt.axis('equal')
    plt.show()
