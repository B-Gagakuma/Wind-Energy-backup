from pyoptsparse import Optimization, SNOPT, pyOpt_solution, NSGA2
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from non_openmdao_floris import AEP_obj
from statistics import *
from position_constraints import SpacingConstraint, circularBoundary
from FLORISSE3D.setupOptimization import amaliaRose


"""This is your objective function. You need your objective and all your constraints in here"""
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

    mean, var = mean_variance_rect(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
    funcs['mean'] = -mean/1.E4
    funcs['var'] = var/1.E8

    funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=minSpacing)/1.E5
    funcs['bound'] = circularBoundary(turbineX, turbineY, circle_radius)/100.

    fail = False

    return funcs, fail


## Main
if __name__ == "__main__":

    """not sure if you need to have global variables, but I did. anything that isn't a
    design variable I made global"""

    global rotorDiameter
    global turbineZ
    global ratedPower
    global windDirections
    global windSpeeds
    global windFrequencies
    global shearExp
    global minSpacing
    global circle_radius

    """this sets up initial turbine locations in a grid"""
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

    index_num = 50
    mean_values = np.zeros(index_num)
    var_values = np.zeros(index_num)
    x_values = np.zeros((index_num,nTurbines))
    y_values = np.zeros((index_num,nTurbines))

    indices = np.arange(index_num)

    """This sets up the initial turbine locations randomly"""
    for index in range(index_num):

        """random setup"""
        rotor_diameter = 100.
        turbineX = np.array([])
        turbineY = np.array([])
        for i in range(nTurbines):
            good=False
            while good==False:
                x = float(np.random.rand(1)*900.)
                y = float(np.random.rand(1)*900.)
                good_sum = 0
                for j in range(len(turbineX)):
                    dist = 0.
                    dist = np.sqrt((turbineX[j]-x)**2+(turbineY[j]-y)**2)
                    if dist > rotor_diameter:
                        good_sum += 1
                if good_sum == len(turbineX):
                    turbineX = np.append(turbineX,x)
                    turbineY = np.append(turbineY,y)
                    good=True
                else:
                    print 'BAD'

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
        """define the wind rose, I have mine in a function but you can do it manually"""
        windDirections, windFrequencies, windSpeeds = amaliaRose(30)

        shearExp = 0.15
        minSpacing = 2.0

        """initially call my objective function to get initial values and see if it's working"""
        input = {'xvars':turbineX*1000.,'yvars':turbineY*1000.}
        funcs,_ = obj_func(input)
        print 'mean start: ', -funcs['mean']
        print 'var: ', funcs['var']
        # print funcs['sep']
        # print funcs['bound']

        """Optimization"""
        """setup problem and give it the function"""
        optProb = Optimization('Wind_Farm_mean', obj_func)

        """say what the objective is (remember this will always be MINIMIZED)"""
        optProb.addObj('mean')

        """define the design variables"""
        optProb.addVarGroup('xvars', nTurbines, 'c', lower=None, upper=None, value=turbineX)
        optProb.addVarGroup('yvars', nTurbines, 'c', lower=None, upper=None, value=turbineY)

        """define the constraints"""
        num_cons_sep = (nTurbines-1)*nTurbines/2
        optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
        optProb.addConGroup('bound', nTurbines, lower=0., upper=None)
        optProb.addConGroup('var', 1, lower=None, upper=0.2)

        # opt = NSGA2()
        # opt.setOption('maxGen', 200)
        # opt.setOption('PopSize', 15*(2*nturb+nrpm))
        # opt.setOption('pMut_real', 0.01)
        # opt.setOption('pCross_real', 1.0)
        """define the optimizer and pass optimizer options"""
        opt = SNOPT()
        opt.setOption('Scale option',0)
        opt.setOption('Iterations limit',1000000)

        opt.setOption('Print file','Print.out')
        opt.setOption('Summary file','Summary%s.out'%(index))

        """post process, pull out results"""
        res = opt(optProb)#, sens=None)

        xf = res.xStar['xvars']
        yf = res.xStar['yvars']

        input = {'xvars':xf,'yvars':yf}
        funcs,_ = obj_func(input)
        mean =  -funcs['mean']
        var =  funcs['var']

        mean_values[index] = mean
        var_values[index] = var
        x_values[index] = xf
        y_values[index] = yf

        # for i in range(nTurbines):
        #     circ_start = plt.Circle((turbineX[i]*1000.,turbineY[i]*1000.), rotorDiameter[i]/2., linestyle='dashed',facecolor="none",edgecolor="black")
        #     circ_opt = plt.Circle((xf[i]*1.,yf[i]*1.), rotorDiameter[i]/2.,facecolor="red",edgecolor="red",alpha=0.2)
        #     plt.gca().add_patch(circ_start)
        #     plt.gca().add_patch(circ_opt)
        # circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed',facecolor='none',label='Boundaries')
        # plt.gca().add_patch(circ_outer)
        # plt.axis('equal')
        # plt.show()

    print 'mean_values: ', mean_values
    print 'var_values: ', var_values
    print 'best index: ', indices[np.argmax(mean_values)]
    print 'best mean: ', max(mean_values)
    print 'var of best mean: ', var_values[np.argmax(mean_values)]

    print 'x: ', x_values[np.argmax(mean_values)]
    print 'y: ', y_values[np.argmax(mean_values)]



    # for i in range(nTurbines):
    #     circ_opt = plt.Circle((x_values[np.argmax(mean_values)][i],y_values[np.argmax(mean_values)][i]), rotorDiameter[i]/2.,facecolor="red",edgecolor="red",alpha=0.2)
    #     plt.gca().add_patch(circ_opt)
    # circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed',facecolor='none',label='Boundaries')
    # plt.gca().add_patch(circ_outer)
    # plt.axis('equal')
    # plt.show()
