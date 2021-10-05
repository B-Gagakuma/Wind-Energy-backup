from pyoptsparse import Optimization, SNOPT, pyOpt_solution, NSGA2
import numpy as np
from scipy.optimize import fsolve
from scipy import interpolate
import matplotlib.pyplot as plt
from non_openmdao_floris import AEP_obj, optVariance, power_call
from position_constraints import SpacingConstraint, circularBoundary
from windRoses import reddingRose, denverRose, amaliaRose
import scipy.io as io


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

    turbineX = xdict['xvars']*100.
    turbineY = xdict['yvars']*100.

    funcs = {}

    # AEP = AEP_obj(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)

    meanPower, varPower = optVariance(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)


    m1 = 3.03616692303902
    funcs['mn'] = -m1
    funcs['mean'] = -meanPower/1.E4
    funcs['var'] = varPower/1.E8
    # funcs['obj'] = np.percentile(dir_powers, 98)

    funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=minSpacing)/1.E5
    funcs['bound'] = circularBoundary(turbineX, turbineY, circle_radius)/100.

    fail = False

    return funcs, fail


''' MAIN '''
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


    nRows = 8
    # width = 1700.
    nTurbines = nRows**2
    # turbineX = np.zeros(nTurbines)
    # turbineY = np.zeros(nTurbines)
    # rotorDiameter = np.ones(nTurbines)*126.4
    #
    # loc_array = np.arange(nRows)/2.
    # loc_array = loc_array - max(loc_array)/2.
    #
    # for i in range(nRows):
    #     turbineX[i*nRows:nRows*(i+1)] = loc_array[i]*width
    #     for j in range(nRows):
    #         turbineY[i*nRows+j] = loc_array[j]*width

    # nTurbines = len(turbineX)
    # turbineZ = np.ones(nTurbines)*100.
    # rotorDiameter = np.ones(nTurbines)*126.4
    # ratedPower = np.ones(nTurbines)*3500.

    # circle_radius = (abs(loc_array[0])*100.)*np.sqrt(2)
    circle_radius = 2975. # max(turbineX)
    # circle_radius = 0.5 * np.sqrt((2*max(turbineX))**2 + (2*max(turbineY)**2))
    #
    # print 'turbineX: ', max(turbineX)
    # print 'turbineY: ', turbineY
    #
    # for i in range(nTurbines):
    #
    #     circ_start = plt.Circle((turbineX[i], turbineY[i]), rotorDiameter[i]/2., color='sandybrown')
    #     plt.gcf().gca().add_patch(circ_start)
    #
    # plt.axis('equal')
    #
    # plt.show()


    """wind data"""
    # inputfile = '../../code/ouu/Simple Wind Farm Applications_files/WindRoses/windrose_amalia_8ms.txt'

    # wind_data = np.loadtxt(inputfile)

    # Directions = wind_data[:,0]
    # Speeds = wind_data[:,1]  # Speed is a constant for this file.
    # Frequencies = wind_data[:,2]

    # freq_func = interpolate.CubicSpline(Directions, Frequencies)

    # nSamples = 50
    #
    # windDirections, windFrequencies, windSpeeds = amaliaRose(nSamples)
    # # windDirections = np.array([0., 45, 90., 200.])
    # # windFrequencies = np.array([0.2, 0.4, 0.1, 0.3])
    # # windSpeeds = np.array([5., 8., 4, 6. ])
    #
    # # print 'wind', windDirections, windFrequencies, windSpeeds
    #
    # shearExp = 0.15
    # minSpacing = 2.0
    # # circle_radius = max(turbineX)*np.sqrt(2)
    #
    # input = {'xvars':turbineX/100., 'yvars':turbineY/100.}
    # funcs,_ = obj_func(input)
    #
    # print funcs['bound']
    # print "mean", funcs['mean']
    # print "var", funcs['var']
    # print "sep_con", funcs['sep']
    #
    #
    # """Optimization"""
    # #
    # optProb = Optimization('Wind_Farm_mean', obj_func)
    # optProb.addObj('mean')
    #
    # optProb.addVarGroup('xvars', nTurbines, 'c', lower=None, upper=None, value=turbineX/100.)
    # optProb.addVarGroup('yvars', nTurbines, 'c', lower=None, upper=None, value=turbineY/100.)
    #
    # d = 0.01
    #
    # num_cons_sep = (nTurbines-1)*nTurbines/2
    # optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
    # optProb.addConGroup('bound', nTurbines, lower=0., upper=None)
    # # optProb.addConGroup('var', 1, lower=None, upper=6.65)
    # optProb.addConGroup('var', 1, lower=None, upper=(funcs['var'] - (d*funcs['var'])))

    ## gradient-free
    # opt = NSGA2()
    # opt.setOption('maxGen', 200)
    # opt.setOption('PopSize', 20*nTurbines)
    # opt.setOption('pMut_real', 0.01)
    # opt.setOption('pCross_real', 1.0)


    ## gradient-based
    # opt = SNOPT()
    # opt.setOption('Scale option',0)
    # opt.setOption('Iterations limit',1000000)
    # opt.setOption('Verify level', 3)
    # opt.setOption('Major optimality tolerance', 1.E-4)
    #
    # opt.setOption('Print file','Print.out')
    # opt.setOption('Summary file','Summary.out')
    #
    # res = opt(optProb)#, sens=None)
    #
    #
    # # print "mean_opt", res.fStar
    #
    #
    # # pow = np.array(-1*res.fStar)*1e4
    # xf = res.xStar['xvars']
    # yf = res.xStar['yvars']
    #
    #
    # for i in range(nTurbines):
    #     circ_start = plt.Circle((turbineX[i], turbineY[i]), rotorDiameter[i]/2., color='sandybrown')
    #     circ_opt = plt.Circle((xf[i]*100., yf[i]*100.), rotorDiameter[i]/2., color='dodgerblue')
    #
    #     plt.gcf().gca().add_patch(circ_start)
    #     plt.gcf().gca().add_patch(circ_opt)
    #
    # circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed', fill=False, label='Boundaries')
    # plt.gcf().gca().add_patch(circ_outer)
    # plt.axis('equal')
    # plt.show()


    ''' random starts '''

    # circle_radius = (abs(loc_array[0])*1000.)*np.sqrt(2)

    d = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10])/100.

    index_num = 2
    mean_values = np.zeros((index_num, len(d)))
    var_values = np.zeros((index_num, len(d)))
    x_values = np.zeros((index_num,nTurbines))
    y_values = np.zeros((index_num,nTurbines))


    indices = np.arange(index_num)

    for index in range(index_num):

        rotor_diameter = 126.4
        turbineX = np.array([])
        turbineY = np.array([])
        for i in range(nTurbines):
            good=False
            while good==False:
                x = float(np.random.rand(1)*2975.)
                y = float(np.random.rand(1)*2975.)
                good_sum = 0
                new_rad = np.sqrt(x**2 + y**2)
                if new_rad <= circle_radius:
                    for j in range(len(turbineX)):
                        dist = 0.
                        dist = np.sqrt((turbineX[j]-x)**2+(turbineY[j]-y)**2)
                        if dist > rotor_diameter:
                            good_sum += 1
                    if good_sum == len(turbineX):
                        turbineX = np.append(turbineX, x)
                        turbineY = np.append(turbineY, y)
                        good=True
                    # else:
                    #     print 'BAD'
        turbineX = turbineX/100.
        turbineY = turbineY/100.
        turbineZ = np.ones(nTurbines)*100.
        rotorDiameter = np.ones(nTurbines)*126.4
        ratedPower = np.ones(nTurbines)*3500.

        windDirections, windFrequencies, windSpeeds = amaliaRose(100)

        shearExp = 0.15
        minSpacing = 2.0

        input = {'xvars':turbineX, 'yvars':turbineY}
        funcs,_ = obj_func(input)
        print 'mean start: ', -funcs['mean']
        print 'var: ', funcs['var']
        print funcs['sep']
        print funcs['bound']

        ups = np.zeros(len(d))
        for j in range(len(ups)):
            ups[j] = 1.95417355850624 - (d[j]*1.95417355850624)

        for k in range(len(ups)):
            """Optimization"""
            optProb = Optimization('Wind_Farm_mean', obj_func)
            optProb.addObj('mn')

            optProb.addVarGroup('xvars', nTurbines, 'c', lower=None, upper=None, value=turbineX/100.)
            optProb.addVarGroup('yvars', nTurbines, 'c', lower=None, upper=None, value=turbineY/100.)


            # d = 0.02
            num_cons_sep = (nTurbines-1)*nTurbines/2
            optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
            optProb.addConGroup('bound', nTurbines, lower=0., upper=None)
            # optProb.addConGroup('var', 1, lower=None, upper=0.8)
            # optProb.addConGroup('var', 1, lower=None, upper=(funcs['var'] - (d*funcs['var'])))
            optProb.addConGroup('var', 1, lower=None, upper=ups[k])


            # opt = NSGA2()
            # opt.setOption('maxGen', 200)
            # opt.setOption('PopSize', 15*(2*nturb+nrpm))
            # opt.setOption('pMut_real', 0.01)
            # opt.setOption('pCross_real', 1.0)

            opt = SNOPT()
            opt.setOption('Scale option',0)
            opt.setOption('Iterations limit',1000000)
            opt.setOption('Print file','Print_rand%s_%s.out'%(index, k))
            opt.setOption('Summary file','Summary%s_%s.out'%(index, k))
            opt.setOption('Major optimality tolerance', 1.E-4)
            # print "sep_con", funcs['sep']

            res = opt(optProb)#, sens=None)

            xf = res.xStar['xvars']
            yf = res.xStar['yvars']

            input = {'xvars':xf,'yvars':yf}
            funcs,_ = obj_func(input)
            mean =  -funcs['mean']
            var =  funcs['var']

            mean_values[index][k] = mean
            var_values[index][k] = var
            x_values[index] = xf
            y_values[index] = yf


        io.savemat('mean_constr_circ64.mat', {'mean':mean_values, 'variance':var_values})#, 'turbineX':x_values[np.argmax(mean_values)], 'turbineY':y_values[np.argmax(mean_values)]})

        # print 'mean_values: ', mean_values
        # print 'var_values: ', var_values
        # print 'best index: ', indices[np.argmax(mean_values)]
        # print 'best mean: ', max(mean_values)
        # print 'var of best mean: ', var_values[np.argmax(mean_values)]



            # print 'mean', mean
            # print 'var', var
            # print 'x', xf
            # print 'y', yf


            # for i in range(nTurbines):
            #     circ_start = plt.Circle((turbineX[i]*1000.,turbineY[i]*1000.), rotorDiameter[i]/2., linestyle='dashed',facecolor="none",edgecolor="black")
            #     circ_opt = plt.Circle((xf[i]*1.,yf[i]*1.), rotorDiameter[i]/2.,facecolor="red",edgecolor="red",alpha=0.2)
            #     plt.gca().add_patch(circ_start)
            #     plt.gca().add_patch(circ_opt)
            # circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed',facecolor='none',label='Boundaries')
            # plt.gca().add_patch(circ_outer)
            # plt.axis('equal')
            # plt.show()


        # print 'mean_values: ', mean_values
        # print 'var_values: ', var_values
        # print 'best index: ', indices[np.argmax(mean_values)]
        # print 'best mean: ', max(mean_values)
        # print 'var of best mean: ', var_values[np.argmax(mean_values)]


        # print 'x: ', x_values[np.argmax(mean_values)]
        # print 'y: ', y_values[np.argmax(mean_values)]

        # for l in range(len(d)):
            # io.savemat('mean_constr_circ16_%s.mat'%(d[l]), {'mean':mean_values, 'variance':var_values, 'turbineX':x_values[np.argmax(mean_values)], 'turbineY':y_values[np.argmax(mean_values)]})
        # io.savemat('mean_constr_circ16.mat', {'mean':mean_values, 'variance':var_values, 'turbineX':x_values[np.argmax(mean_values)], 'turbineY':y_values[np.argmax(mean_values)]})
