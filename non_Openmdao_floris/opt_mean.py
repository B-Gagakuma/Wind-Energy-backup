# from pyoptsparse import Optimization, SNOPT, pyOpt_solution, NSGA2
import numpy as np
# from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# from non_openmdao_floris import AEP_obj
# from statistics import *
# from position_constraints import SpacingConstraint, circularBoundary
# from FLORISSE3D.setupOptimization import amaliaRose
#
#
# def obj_func(xdict):
#     global rotorDiameter
#     global turbineZ
#     global ratedPower
#     global windDirections
#     global windSpeeds
#     global windFrequencies
#     global shearExp
#     global minSpacing
#     global circle_radius
#
#     turbineX = xdict['xvars']
#     turbineY = xdict['yvars']
#
#     funcs = {}
#
#     mean, var = mean_variance_rect(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
#     funcs['mean'] = -mean/1.E4
#     funcs['var'] = var/1.E8
#
#     funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=minSpacing)/1.E5
#     # funcs['bound'] = circularBoundary(turbineX, turbineY, circle_radius)/100.
#
#     fail = False
#
#     return funcs, fail
#
#
# ## Main
# if __name__ == "__main__":
#
#     global rotorDiameter
#     global turbineZ
#     global ratedPower
#     global windDirections
#     global windSpeeds
#     global windFrequencies
#     global shearExp
#     global minSpacing
#     global circle_radius
#
#     """grid setup"""
#
#     # nRows = 4
#     # nTurbines = nRows**2
#     # loc_array = np.arange(nRows)/2.
#     # loc_array = loc_array - max(loc_array)/2.
#     # turbineX = np.zeros(nTurbines)
#     # turbineY = np.zeros(nTurbines)
#     # for i in range(nRows):
#     #     turbineX[i*nRows:nRows*(i+1)] = loc_array[i]
#     #     for j in range(nRows):
#     #         turbineY[i*nRows+j] = loc_array[j]
#
#     # size = 4 # number of processors (and number of wind directions to run)
#
#     rotor_diameter = 100.  # (m)
#
#     turbineX = np.array([0.,0.,0.,0.,500.,500.,500.,500.,1000.,1000.,1000.,1000.,1500.,1500.,1500.,1500.])#*0.6
#     turbineY = np.array([0.,500.,1000.,1500.,0.,500.,1000.,1500.,0.,500.,1000.,1500.,0.,500.,1000.,1500.])#*0.6
#
#     # turbineX = turbineX * 1000.
#     # turbineY = turbineY * 1000.
#
#     nTurbines = len(turbineX)
#     plt.figure(1)
#     plt.plot(turbineX, turbineY, 'o', color='sandybrown', markersize=13, label='initial')
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     """ single run """
#
#     turbineZ = np.ones(nTurbines)*100.
#     rotorDiameter = np.ones(nTurbines)*rotor_diameter
#     ratedPower = np.ones(nTurbines)*3500.
#     windDirections, windFrequencies, windSpeeds = amaliaRose(100)
#
#     shearExp = 0.15
#     minSpacing = 2.0
#     # circle_radius = (abs(loc_array[0]))*np.sqrt(2)
#
#     input = {'xvars':turbineX*1000.,'yvars':turbineY*1000.}
#     funcs,_ = obj_func(input)
#     print 'mean start: ', -funcs['mean']
#     print 'var: ', funcs['var']
#
#
#     """Optimization"""
#     optProb = Optimization('Wind_Farm_mean', obj_func)
#     optProb.addObj('mean')
#
#     optProb.addVarGroup('xvars', nTurbines, 'c', lower=min(turbineX), upper=max(turbineX), value=turbineX)
#     optProb.addVarGroup('yvars', nTurbines, 'c', lower=min(turbineY), upper=max(turbineY), value=turbineY)
#
#     num_cons_sep = (nTurbines-1)*nTurbines/2
#     optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
#     # optProb.addConGroup('bound', nTurbines, lower=0., upper=None)
#     optProb.addConGroup('var', 1, lower=None, upper=0.2)
#
#     opt = NSGA2()
#     opt.setOption('maxGen', 200)
#     opt.setOption('PopSize', 15*(5*nTurbines))
#     opt.setOption('pMut_real', 0.01)
#     opt.setOption('pCross_real', 1.0)
#
#     opt = SNOPT()
#     opt.setOption('Scale option',0)
#     opt.setOption('Iterations limit',1000000)
#
#     opt.setOption('Print file','Print_single_run_test.out')
#     opt.setOption('Summary file','Summary_single_run_test.out')
#     opt.setOption('Verify level', 3)
#
#
#     res = opt(optProb)#, sens=None)
#
#     xf = res.xStar['xvars']
#     yf = res.xStar['yvars']
#
#     input = {'xvars':xf,'yvars':yf}
#     funcs,_ = obj_func(input)
#     mean =  -funcs['mean']
#     var =  funcs['var']
#
#     print 'obj', funcs
#     print 'result', res
#     print 'x', xf
#     print 'y', yf
#
#     plt.plot(xf, yf, 'o', color = 'dodgerblue', markersize=13, label='optimized')
#     plt.legend()
#     #
#     plt.show()

    # for i in range(nTurbines):
    #     circ_start = plt.Circle((turbineX[i]*1000.,turbineY[i]*1000.), rotorDiameter[i]/2., linestyle='dashed',facecolor="none",edgecolor="black")
    #     circ_opt = plt.Circle((xf[i]*1.,yf[i]*1.), rotorDiameter[i]/2.,facecolor="red",edgecolor="red",alpha=0.2)
    #     plt.gca().add_patch(circ_start)
    #     plt.gca().add_patch(circ_opt)
    # circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed',facecolor='none',label='Boundaries')
    # plt.gca().add_patch(circ_outer)
    # plt.axis('equal')
    # plt.show()



    #     """random setup"""

    # circle_radius = (abs(loc_array[0])*1000.)*np.sqrt(2)
    #
    # index_num = 10
    # mean_values = np.zeros(index_num)
    # var_values = np.zeros(index_num)
    # x_values = np.zeros((index_num,nTurbines))
    # y_values = np.zeros((index_num,nTurbines))
    #
    # indices = np.arange(index_num)
    #
    # for index in range(index_num):
    #
    #     rotor_diameter = 100.
    #     turbineX = np.array([])
    #     turbineY = np.array([])
    #     for i in range(nTurbines):
    #         good=False
    #         while good==False:
    #             x = float(np.random.rand(1)*900.)
    #             y = float(np.random.rand(1)*900.)
    #             good_sum = 0
    #             for j in range(len(turbineX)):
    #                 dist = 0.
    #                 dist = np.sqrt((turbineX[j]-x)**2+(turbineY[j]-y)**2)
    #                 if dist > rotor_diameter:
    #                     good_sum += 1
    #             if good_sum == len(turbineX):
    #                 turbineX = np.append(turbineX,x)
    #                 turbineY = np.append(turbineY,y)
    #                 good=True
    #             else:
    #                 print 'BAD'
    #
    #     turbineZ = np.ones(nTurbines)*100.
    #     rotorDiameter = np.ones(nTurbines)*100.
    #     ratedPower = np.ones(nTurbines)*3500.
    #     # windSpeeds = np.array([8.,5.,10.,14.])
    #     # nDirections = len(windSpeeds)
    #     # windDirections = np.linspace(0.,360.-360./float(nDirections),nDirections)
    #     # windFrequencies = np.array([0.1,0.2,0.3,0.4])
    #     # windFrequencies = windFrequencies/sum(windFrequencies)
    #     #
    #     # windSpeeds = np.array([10.])
    #     # windDirections = np.array([0.])
    #     # windFrequencies = np.array([1.])
    #     windDirections, windFrequencies, windSpeeds = amaliaRose(100)
    #
    #     shearExp = 0.15
    #     minSpacing = 2.0
    #
    #     input = {'xvars':turbineX*1000.,'yvars':turbineY*1000.}
    #     funcs,_ = obj_func(input)
    #     print 'mean start: ', -funcs['mean']
    #     print 'var: ', funcs['var']
    #     # print funcs['sep']
    #     # print funcs['bound']
    #
    #     """Optimization"""
    #     optProb = Optimization('Wind_Farm_mean', obj_func)
    #     optProb.addObj('mean')
    #
    #     optProb.addVarGroup('xvars', nTurbines, 'c', lower=None, upper=None, value=turbineX)
    #     optProb.addVarGroup('yvars', nTurbines, 'c', lower=None, upper=None, value=turbineY)
    #
    #     num_cons_sep = (nTurbines-1)*nTurbines/2
    #     optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
    #     optProb.addConGroup('bound', nTurbines, lower=0., upper=None)
    #     optProb.addConGroup('var', 1, lower=None, upper=0.2)
    #
    #     # opt = NSGA2()
    #     # opt.setOption('maxGen', 200)
    #     # opt.setOption('PopSize', 15*(2*nturb+nrpm))
    #     # opt.setOption('pMut_real', 0.01)
    #     # opt.setOption('pCross_real', 1.0)
    #     opt = SNOPT()
    #     opt.setOption('Scale option',0)
    #     opt.setOption('Iterations limit',1000000)
    #
    #     opt.setOption('Print file','Print.out')
    #     opt.setOption('Summary file','Summary%s.out'%(index))
    #
    #     res = opt(optProb)#, sens=None)
    #
    #     xf = res.xStar['xvars']
    #     yf = res.xStar['yvars']
    #
    #     input = {'xvars':xf,'yvars':yf}
    #     funcs,_ = obj_func(input)
    #     mean =  -funcs['mean']
    #     var =  funcs['var']
    #
    #     mean_values[index] = mean
    #     var_values[index] = var
    #     x_values[index] = xf
    #     y_values[index] = yf
    #
    #     # for i in range(nTurbines):
    #     #     circ_start = plt.Circle((turbineX[i]*1000.,turbineY[i]*1000.), rotorDiameter[i]/2., linestyle='dashed',facecolor="none",edgecolor="black")
    #     #     circ_opt = plt.Circle((xf[i]*1.,yf[i]*1.), rotorDiameter[i]/2.,facecolor="red",edgecolor="red",alpha=0.2)
    #     #     plt.gca().add_patch(circ_start)
    #     #     plt.gca().add_patch(circ_opt)
    #     # circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed',facecolor='none',label='Boundaries')
    #     # plt.gca().add_patch(circ_outer)
    #     # plt.axis('equal')
    #     # plt.show()
    #
    # print 'mean_values: ', mean_values
    # print 'var_values: ', var_values
    # print 'best index: ', indices[np.argmax(mean_values)]
    # print 'best mean: ', max(mean_values)
    # print 'var of best mean: ', var_values[np.argmax(mean_values)]
    #
    # print 'x: ', x_values[np.argmax(mean_values)]
    # print 'y: ', y_values[np.argmax(mean_values)]
    #
    #
    #
    # for i in range(nTurbines):
    #     circ_opt = plt.Circle((x_values[np.argmax(mean_values)][i],y_values[np.argmax(mean_values)][i]), rotorDiameter[i]/2.,facecolor="red",edgecolor="red",alpha=0.2)
    #     plt.gca().add_patch(circ_opt)
    # circ_outer = plt.Circle((0,0), circle_radius, linestyle='dashed',facecolor='none',label='Boundaries')
    # plt.gca().add_patch(circ_outer)
    # plt.axis('equal')
    # plt.show()


x = np.array([6.5000, -6.5000, -6.5000, 2.3069, -5.8561, -6.5000, -6.4998, 6.5000, 6.5000, 4.9247, -6.5000, 3.2486, 0.7207, 6.5000, 2.0306, 6.3946])
y = np.array([-6.5000, 6.5000, 6.5000, 6.5000, 2.3512, 6.5000, -5.3770, -6.5000, 6.5000, -0.4653, 6.5000, -5.4501, -5.6604, -6.5000, 3.8920, 3.6241])

plt.figure()
plt.plot(x, y, 'o', color='dodgerblue', markersize=13)
plt.show()
