import numpy as np
import matplotlib.pyplot as plt
import _floris
from scipy.interpolate import CubicSpline
from non_openmdao_floris import *
from akima import Akima, akima_interp
from scipy.integrate import quadrature, romberg, romb, fixed_quad, quad, simps
from FLORISSE3D.setupOptimization import amaliaRose, amaliaLayout
from windRoses import reddingRose, denverRose
import time

def mean_variance_monte_carlo(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp):
    nTurbines = len(turbineX)
    nDirections = len(windDirections)

    dir_powers = np.zeros(nDirections)
    for i in range(nDirections):
        turbineXw, turbineYw = WindFrame(windDirections[i], turbineX,turbineY)
        Vinf = PowWind(windSpeeds[i], turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_powers[i] = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    mu = np.sum(dir_powers)/len(dir_powers)

    var = 0.

    for i in range(nDirections):
        var += (dir_powers[i] - mu)**2

    variance = var/(len(dir_powers) - 1)

    return mu, variance

def mean_variance_rect(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp):
    nTurbines = len(turbineX)
    nDirections = len(windDirections)

    dir_powers = np.zeros(nDirections)
    for i in range(nDirections):
        turbineXw, turbineYw = WindFrame(windDirections[i], turbineX,turbineY)
        Vinf = PowWind(windSpeeds[i], turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_powers[i] = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)

    # h = ((2*np.pi) - 0.)/float(nDirections)
    mean = np.sum(dir_powers*windFrequencies)
    # var = np.sum((dir_powers**2)*windFrequencies) - mean**2

    var = 0.
    for i in range(nDirections):
        var += windFrequencies[i]*(dir_powers[i]-mean)**2

# def rect_rule (f, a, b, n):
# 	total = 0.0
# 	dx = (b-a)/float(n)
# 	for k in range(0, n):
#         	total = total + f((a + (k*dx)))
# 	return dx*total

    return mean, var

def mean_variance_simps(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp):
    nTurbines = len(turbineX)
    nDirections = len(windDirections)

    dir_powers = np.zeros(nDirections)
    for i in range(nDirections):
        turbineXw, turbineYw = WindFrame(windDirections[i], turbineX,turbineY)
        Vinf = PowWind(windSpeeds[i], turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_powers[i] = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    mu = simps(dir_powers*windFrequencies)

    var = simps(windFrequencies*dir_powers**2) - mu**2


    return mu, var


def mean_variance_trapz(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp):
    nTurbines = len(turbineX)
    nDirections = len(windDirections)

    dir_powers = np.zeros(nDirections)
    for i in range(nDirections):
        turbineXw, turbineYw = WindFrame(windDirections[i], turbineX,turbineY)
        Vinf = PowWind(windSpeeds[i], turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_powers[i] = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    mu = np.trapz(dir_powers*windFrequencies)

    var = np.trapz(windFrequencies*dir_powers**2) - mu**2


    return mu, var



def gaus_mean_integrate_func(direction,*args):
    global func_calls

    turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func = args
    val = np.zeros(len(direction))
    for i in range(len(direction)):
        func_calls += 1
        dir = direction[i]
        speed,_,_,_ = speed_func.interp(dir)
        turbineXw, turbineYw = WindFrame(np.rad2deg(dir), turbineX,turbineY)
        Vinf = PowWind(speed, turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_power = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
        freq,_,_,_ = freq_func.interp(dir)
        val[i] = dir_power*freq

    return val

def variance_integrate_func(direction,*args):
    global func_calls

    turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func = args
    val = np.zeros(len(direction))
    for i in range(len(direction)):
        func_calls += 1
        dir = direction[i]
        speed,_,_,_ = speed_func.interp(dir)
        turbineXw, turbineYw = WindFrame(np.rad2deg(dir), turbineX,turbineY)
        Vinf = PowWind(speed, turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_power = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
        freq,_,_,_ = freq_func.interp(dir)
        val[i] = freq*dir_power**2

    return val

def mean_variance_gaus_quad(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, maxiter):
    args = turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func
    # mean = quadrature(gaus_mean_integrate_func,0.,2.*np.pi,args=(args),maxiter=maxiter)
    mu = fixed_quad(gaus_mean_integrate_func,0.,2.*np.pi, args=(args),n=maxiter)[0]
    sigma2 = fixed_quad(variance_integrate_func, 0., 2.*np.pi, args=(args), n=maxiter)[0] - mu**2

    return mu, sigma2


def romb_mean_integrate_func(direction,*args):
    global func_calls
    func_calls += 1
    turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func = args

    speed,_,_,_ = speed_func.interp(direction)
    turbineXw, turbineYw = WindFrame(np.rad2deg(direction), turbineX,turbineY)
    Vinf = PowWind(speed, turbineZ, shearExp)
    wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
    _,dir_power = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    freq,_,_,_ = freq_func.interp(direction)

    return dir_power*freq

def romb_variance_integrate_func(direction,*args):
    global func_calls
    func_calls += 1
    turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func = args

    speed,_,_,_ = speed_func.interp(direction)
    turbineXw, turbineYw = WindFrame(np.rad2deg(direction), turbineX,turbineY)
    Vinf = PowWind(speed, turbineZ, shearExp)
    wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
    _,dir_power = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    freq,_,_,_ = freq_func.interp(direction)

    return freq*dir_power**2


def mean_variance_romberg(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, maxiter):
    global ndir

    args = turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func
    mu = romberg(romb_mean_integrate_func,0.,2.*np.pi,args=(args), divmax=2)#,show=1)
    variance = romberg(romb_variance_integrate_func,0.,2.*np.pi,args=(args), divmax=2) - mu**2

    return mu, variance

def mean_variance_romb(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp):
    nTurbines = len(turbineX)
    nDirections = len(windDirections)

    # mean = 0.
    # var = 0.
    dir_powers = np.zeros(nDirections)
    for i in range(nDirections):
        turbineXw, turbineYw = WindFrame(windDirections[i], turbineX,turbineY)
        Vinf = PowWind(windSpeeds[i], turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_powers[i] = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)

    h = ((2*np.pi) - 0.)/nDirections
    mean = romb(dir_powers*windFrequencies)
    var = romb((dir_powers**2)*windFrequencies) - mean**2
    print 'oner', var

    return mean, var


def cc_mean_integrate_func(direction,*args):
    global func_calls
    func_calls += 1
    turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func = args

    speed,_,_,_ = speed_func.interp(direction)
    turbineXw, turbineYw = WindFrame(np.rad2deg(direction), turbineX,turbineY)
    Vinf = PowWind(speed, turbineZ, shearExp)
    wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
    _,dir_power = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    freq,_,_,_ = freq_func.interp(direction)

    return dir_power*freq

def cc_variance_integrate_func(direction,*args):
    global func_calls

    func_calls += 1
    turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func = args

    speed,_,_,_ = speed_func.interp(direction)
    turbineXw, turbineYw = WindFrame(np.rad2deg(direction), turbineX,turbineY)
    Vinf = PowWind(speed, turbineZ, shearExp)
    wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
    _,dir_power = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    freq,_,_,_ = freq_func.interp(direction)

    return freq*dir_power**2

def mean_variance_cc(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, maxiter):

    global ndir

    args = turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,shearExp,speed_func,freq_func
    mu = quad(cc_mean_integrate_func,0.,2.*np.pi,args=(args),limit=ndir)[0]
    variance = quad(cc_variance_integrate_func,0.,2.*np.pi,args=(args),limit=ndir)[0] - mu**2

    return mu, variance


def frequency_function():

    windFrequencies = np.array([1.17812570e-02, 1.09958570e-02, 9.60626600e-03, 1.21236860e-02,
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
                               1.10562900e-02, 1.04521380e-02, 1.16201970e-02, 1.10562700e-02])/(0.0871828428793*0.999814399093*0.999999979031*0.999999999998)

    windDirections = np.linspace(0.,(2.*np.pi)-(2.*np.pi)/float(len(windFrequencies)), len(windFrequencies))
    spline_freq = Akima(windDirections, windFrequencies)

    return spline_freq


def speed_function():

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
                                6.94716392])

    windDirections = np.linspace(0.,(2.*np.pi)-(2.*np.pi)/float(len(windSpeeds)), len(windSpeeds))
    spline_speed = Akima(windDirections, windSpeeds)

    return spline_speed

# def frequency_function():
#
#     likelihood = np.array([5.9, 4.8, 3.4, 1.9, 1.2, 1., 0.7, 0.5, 0.6, 0.5, 0.6, 0.5,
#                             0.6, 1.1, 1.3, 2.3, 3.9, 4.1, 3.6, 3.1, 2.7, 1.8, 1.3, 0.8,
#                             0.9, 1.1, 1.1, 1.3, 0.9, 1.2, 1.8, 2.3, 3.6, 4.9, 6.5, 7.3])
#
#     windFrequencies = likelihood/np.sum(likelihood)
#
#     windDirections = np.linspace(0.,(2.*np.pi)-(2.*np.pi)/float(len(windFrequencies)), len(windFrequencies))
#
#     spline_freq = Akima(windDirections, windFrequencies)
#
#     return spline_freq
#
# def speed_function():
#
#     windSpeeds = np.array([3.9, 4., 3.6, 3.6, 3., 2.8, 2.2, 2.4, 2., 2.5, 2.1, 2.1, 2.3, 2.5,
#                            3.3, 3.9, 5., 5.2, 4.8, 4.6, 3.7, 3.5, 3., 2.9, 2.9, 3.1, 2.9,
#                            2.8, 2.7, 2.7, 2.5, 2.5, 2.6, 2.8, 3.4, 3.8])
#
#     windDirections = np.linspace(0.,(2.*np.pi)-(2.*np.pi)/float(len(windSpeeds)), len(windSpeeds))
#     spline_speed = Akima(windDirections, windSpeeds)
#
#     return spline_speed
#

if __name__=="__main__":

    # nRows = 8
    # width = 1000.
    # loc_array = np.arange(nRows)/2.
    # loc_array = loc_array - max(loc_array)/2.
    # turbineX = np.zeros(nTurbines)
    # turbineY = np.zeros(nTurbines)
    # turbineX, turbineY = amaliaLayout()
    turbineX = np.array([0.,0.,0.,0.,500.,500.,500.,500.,1000.,1000.,1000.,1000.,1500.,1500.,1500.,1500.])
    turbineY = np.array([0.,500.,1000.,1500.,0.,500.,1000.,1500.,0.,500.,1000.,1500.,0.,500.,1000.,1500.])
    nTurbines = len(turbineX)

    #
    # for i in range(nRows):
    #     turbineX[i*nRows:nRows*(i+1)] = loc_array[i]*width
    #     for j in range(nRows):
    #         turbineY[i*nRows+j] = loc_array[j]*width

    # turbineX = np.random.rand(nTurbines)*1500.-750.
    # turbineY = np.random.rand(nTurbines)*1500.-750.
    turbineZ = np.ones(nTurbines)*100.
    rotorDiameter = np.ones(nTurbines)*126.4
    ratedPower = np.ones(nTurbines)*3500.
    shearExp = 0.15
    # plt.figure(1)
    # plt.plot(turbineX, turbineY, 'o', color='dodgerblue', markersize=13)
    # plt.show()

    freq_func = frequency_function()
    speed_func = speed_function()

    windDirections, windFrequencies, windSpeeds = amaliaRose(1000)
    trueMean, trueVar = mean_variance_rect(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
    print 'once', trueMean, trueVar
    # true = 19697.3144901 #4x4 grid
    # true = 73259.1473992 #8x8 grid

    # trueMean = 30157.260065057657
    # trueVar = 248037259.56089923

    # Amalia wind Rose (amalia layout)
    # trueMean = 70221.9101464137
    # trueVar = 1490207016.9447415

    # 1000 point gauss quad (amaliaRose and layout)
    # func_calls=0
    # trueMean, trueVar = mean_variance_gaus_quad(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, 1000)
    # print 'first', trueMean, trueVar
    # print 'f_calls', func_calls
    trueMean = 70173.12911656126
    trueVar = 1489679615.9385786

    # Amalia wind Rose (8x8 grid)
    # trueMean = 73268.75624880308
    # trueVar = 1684746997.4778266


    # Redding wind rose (8x8 grid)
    # trueMean = 6842.849567857633
    # trueVar = 37694912.18197613

    # Denver wind rose (8x8 grid)
    # trueMean = 20568.301583467404
    # trueVar = 76364948.55766141



    # 200000 sample Monte Carlo mean/varince
    # windDirections, windFrequencies, windSpeeds = amaliaRose(200000)
    # start_mc = time.time()
    # mean_mc, var_mc =  mean_variance_monte_carlo(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
    # end_mc = time.time()
    # mct = end_mc - start_mc
    #
    # print mean_mc, var_mc
    # print 'time', mct
    # trueMean = 25466.383814904242
    # # time 115.438555002
    #
    # trueVar = 218622412.09999433

    # global func_calls
    # func_calls=0


    """timing and function calls"""
    global func_calls
    global ndir

    func_calls=0
    #
    num = 30
    accuracy = np.linspace(10, 300, num)#*5

    gaussian_mean_error = np.zeros(num)
    gaussian_var_error = np.zeros(num)
    gaussian_time = np.zeros(num)

    curtis_mean_error = np.zeros(num)
    curtis_var_error = np.zeros(num)
    curtis_time = np.zeros(num)

    rectangle_mean_error = np.zeros(num)
    rectangle_var_error = np.zeros(num)
    rectangle_time = np.zeros(num)

    romberg_mean_error = np.zeros(num)
    romberg_var_error = np.zeros(num)
    romberg_time = np.zeros(num)

    simpsons_mean_error = np.zeros(num)
    simpsons_var_error = np.zeros(num)
    simpsons_time = np.zeros(num)

    trapezoidal_mean_error = np.zeros(num)
    trapezoidal_var_error = np.zeros(num)
    trapezoidal_time = np.zeros(num)

    # # monte_mean_error = np.zeros(num)
    # # monte_var_error = np.zeros(num)
    # # monte_time = np.zeros(num)
    #
    #
    #
    # # plt.figure()
    # # plt.xlabel('# of samples')
    # # plt.ylabel('metric')

    # plt.xlabel('# of points')
    # plt.ylabel('metric')
    # plt.title('rect mean/variance convergence')

    for i in range(num):
        # print i
        acc = int(accuracy[i]) #accuracy
        ndir = acc
        print 'check', acc


    #     # # Monte Carlo (need to sample over probability distribution)
    #     # windDirections, windFrequencies, windSpeeds = amaliaRose(acc)
    #     # start_mc = time.time()
    #     # mean_mc, var_mc =  mean_variance_monte_carlo(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
    #     # end_mc = time.time()
    #     #
    #     # mc_time = end_mc - start_mc
    #     # mc_mean_err = abs((trueMean - mean_mc)/trueMean*100.)
    #     # mc_var_err = abs((trueVar - var_mc)/trueVar*100.)
    #     # # print 'mc error: ', mc_err
    #     # # print 'mc time: ', mc_time
    #     #
    #     # plt.plot(acc, mean_mc, '.', color='dodgerblue')#, label='mean')
    #     # plt.plot(acc, var_mc, '.', color='sandybrown')#, label='variance')
    #     # plt.plot(acc, mc_time, '.', color='darkorchid')#, label='time')
    #     #
    #     # monte_time[i] = mc_time
    #     # monte_mean_error[i] = mc_mean_err
    #     # monte_var_error[i] = mc_var_err
    #
        # Gauss quadrature
        start_gaus = time.time()
        mean_gauss, var_gauss = mean_variance_gaus_quad(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, acc)
        end_gaus = time.time()

        gaus_time = end_gaus - start_gaus
        gaussMean_err = abs((trueMean - mean_gauss)/trueMean*100.)
        gaussVar_err = abs((trueVar - var_gauss)/trueVar*100.)

        # print 'gauss error in mean: ', gaussMean_err
        # print 'gauss error in Var: ', gaussVar_err
        # print "gaus time: ", gaus_time
        gaussian_time[i] = gaus_time
        gaussian_mean_error[i] = gaussMean_err
        gaussian_var_error[i] = gaussVar_err


        # Clenshaw-Curtis
        start_cc = time.time()
        mean_cc, var_cc =  mean_variance_cc(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, acc)
        end_cc = time.time()

        cc_time = end_cc - start_cc
        cc_mean_err = abs((trueMean - mean_cc)/trueMean*100.)
        cc_var_err = abs((trueVar - var_cc)/trueVar*100.)

        # print 'cc error: ', cc_err
        # print "cc time: ", cc_time
        curtis_time[i] = cc_time
        curtis_mean_error[i] = cc_mean_err
        curtis_var_error[i] = cc_var_err


        # Romberg integration
        start_romb = time.time()
        mean_romb, var_romb = mean_variance_romberg(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, acc)
        end_romb = time.time()

        romb_time = end_romb - start_romb
        romb_mean_err = abs((trueMean - mean_romb)/trueMean*100.)
        romb_var_err = abs((trueVar - var_romb)/trueVar*100.)

        # print 'romberg error: ', romberg_err
        # print "romberg time: ", romberg_time
        romberg_time[i] = romb_time
        romberg_mean_error[i] = romb_mean_err
        romberg_var_error[i] = romb_var_err


        # Rectangle rule
        windDirections, windFrequencies, windSpeeds = amaliaRose(acc)
        start_rect = time.time()
        mean_rect, var_rect =  mean_variance_rect(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)

        # plt.plot(acc, mean_rect, '.', color='dodgerblue')
        # plt.plot(acc, var_rect, '.', color='sandybrown')

        # plt.legend(['mean', 'variance'])
        # plt.show()

        end_rect = time.time()

        rect_time = end_rect - start_rect
        rect_mean_err = abs((trueMean - mean_rect)/trueMean*100.)
        rect_var_err = abs((trueVar - var_rect)/trueVar*100.)
        # print 'rect error: ', rect_err
        # print "rect time: ", rect_time
        rectangle_time[i] = rect_time
        rectangle_mean_error[i] = rect_mean_err
        rectangle_var_error[i] = rect_var_err

        # Romberg data set
        # start_romb = time.time()
        # mean_romb, var_romb = mean_variance_romb(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
        # end_romb = time.time()
        #
        # romb_time = end_romb - start_romb
        # romb_mean_err = abs((trueMean - mean_romb)/trueMean*100.)
        # romb_var_err = abs((trueVar - var_romb)/trueVar*100.)
        #
        # # print 'romberg error: ', romberg_err
        # # print "romberg time: ", romberg_time
        # romberg_time[i] = romb_time
        # romberg_mean_error[i] = romb_mean_err
        # romberg_var_error[i] = romb_var_err


        # Simpson's rule
        start_simps = time.time()
        mean_simps, var_simps = mean_variance_simps(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
        end_simps = time.time()

        simps_time = end_simps - start_simps
        simps_mean_err = abs((trueMean - mean_simps)/trueMean*100.)
        simps_var_err = abs((trueVar - var_simps)/trueVar*100.)
        # print 'simps error: ', simps_err
        # print "simps time: ", simps_time
        simpsons_time[i] = simps_time
        simpsons_mean_error[i] = simps_mean_err
        simpsons_var_error[i] = simps_var_err


        # Trapezoidal rule
        start_trapz = time.time()
        mean_trapz, var_trapz = mean_variance_trapz(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
        end_trapz = time.time()

        trapz_time = end_trapz - start_trapz
        trapz_mean_err = abs((trueMean - mean_trapz)/trueMean*100.)
        trapz_var_err = abs((trueVar - var_trapz)/trueVar*100.)
        # print 'trapz error: ', trapz_err
        # print "trapz time: ", trapz_time
        trapezoidal_time[i] = trapz_time
        trapezoidal_mean_error[i] = trapz_mean_err
        trapezoidal_var_error[i] = trapz_var_err



    # time
    plt.figure(2)
    plt.title('time')
    plt.xlabel('# of quadrature points')
    plt.ylabel('time (s)')
    plt.semilogy(accuracy, rectangle_time, linewidth=2, color='dodgerblue', label='rect')
    plt.semilogy(accuracy, gaussian_time, linewidth=2, color='darkslategray', label='gauss')
    # plt.semilogy(accuracy, romberg_time, linewidth=2, color='limegreen', label='romberg')
    plt.semilogy(accuracy, curtis_time, linewidth=2, color='red', label='cc')
    plt.semilogy(accuracy, simpsons_time, linewidth=2, color='orchid', label='simps')
    plt.semilogy(accuracy, trapezoidal_time, linewidth=2, color='sandybrown', label='trapz')
    plt.legend()

    # mean
    plt.figure(3)
    plt.title('% error in mean')
    plt.xlabel('# of quadrature points')
    plt.ylabel('% error')
    plt.semilogy(accuracy, rectangle_mean_error, linewidth=2, color='dodgerblue', label='rect')
    plt.semilogy(accuracy, gaussian_mean_error, linewidth=2, color='darkslategray', label='gauss')
    # plt.semilogy(accuracy, romberg_mean_error, linewidth=2, color='limegreen', label='romberg')
    plt.semilogy(accuracy, curtis_mean_error, linewidth=2, color='red', label='cc')
    plt.semilogy(accuracy, simpsons_mean_error, linewidth=2, color='orchid', label='simps')
    plt.semilogy(accuracy, trapezoidal_mean_error, linewidth=2, color='sandybrown', label='trapz')
    plt.legend()

    # variance
    plt.figure(4)
    plt.title('% error in variance')
    plt.xlabel('# of quadrature points')
    plt.ylabel('% error')
    plt.semilogy(accuracy, rectangle_var_error, linewidth=2, color='dodgerblue', label='rect')
    plt.semilogy(accuracy, gaussian_var_error, linewidth=2, color='darkslategray', label='gauss')
    # plt.semilogy(accuracy, romberg_var_error, linewidth=2, color='limegreen', label='romberg')
    plt.semilogy(accuracy, curtis_var_error, linewidth=2, color='red', label='cc')
    plt.semilogy(accuracy, simpsons_var_error, linewidth=2, color='orchid', label='simps')
    plt.semilogy(accuracy, trapezoidal_var_error, linewidth=2, color='sandybrown', label='trapz')
    plt.legend()

    plt.show()


    # num = 9
    # maxiter = np.linspace(2,10,num)
    # error_rect = np.zeros(num)
    # error_gaus = np.zeros(num)
    # error_romb = np.zeros(num)
    # error_cc = np.zeros(num)
    # for i in range(num):
    #     iter = int(maxiter[i])
    #     mean_gaus,_ =  mean_variance_gauss_quad(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, iter)
    #     mean_cc =  mean_variance_cc(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, iter)
    #     # mean_romb =  mean_variance_romberg(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, shearExp, speed_func, freq_func, iter)
    #     windDirections, windFrequencies, windSpeeds = amaliaRose(iter)
    #     mean_rect,_ = mean_variance_rect(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)
    #     print iter
    #     print 'percent gaus: ', abs((true-mean_gaus)/true*100.)
    #     print 'percent rect: ', abs((true-mean_rect)/true*100.)
    #     print 'percent cc: ', abs((true-mean_cc)/true*100.)
    #     # print 'percent romb: ', abs((true-mean_romb)/true*100.)
    #     error_gaus[i] = abs((true-mean_gaus)/true*100.)
    #     error_rect[i] = abs((true-mean_rect)/true*100.)
    #     error_cc[i] = abs((true-mean_cc)/true*100.)
    #     # error_romb[i] = abs((true-mean_romb)/true*100.)


"""
# Decatur IL

likelihood = np.array([2.1, 1.5, 1.7, 1.9,	2., 2.5, 2.7, 2.9, 2.8, 2.5, 2.5, 1.8,
                       1.7, 1.6, 1.4, 1.2, 2.3, 3.1, 4.6, 4.8, 4.7, 5.3, 3.6, 2.9,
                       2.3, 1.8, 2.3, 1.9, 3, 3.5, 3., 3.6, 2.8, 2.6, 1.8, 2.6])

windFrequencies = likelihood/np.sum(likelihood)


windDirections = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140.,
                           150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270.,
                           280., 290., 300., 310., 320., 330., 340., 350.])

windSpeeds = np.array([4.6, 4.6, 4.5, 4.6, 4.1, 4.4, 4.5, 4.9, 5.2, 4.9, 4.8, 4.7, 4.2,
                      3.8, 4.1, 4., 4.3, 5., 5.5, 5.8, 6.5, 5.6, 5, 4.5, 4.6, 4.2, 4.1,
                      4.5, 4.8, 5.5, 4.8, 5.1, 5.5, 5.7, 4.8, 5.6])

plt.figure(1)
plt.plot(windDirections, windFrequencies, color='dodgerblue')
plt.xlabel('wind direction (deg)')
plt.ylabel('frequency')


# Denver CO

likelihood = np.array([2.8, 2.8, 2., 2.3, 2.5, 3., 2.5, 2.5, 2.5, 2.6, 2.7, 1.9,
                       2.1, 2., 2.6, 3., 4.4, 4.4, 2.9, 3.1, 4.2, 5., 4., 3.8, 3.4,
                       2.3, 2.8, 2.3, 2.4, 2.1, 1.9, 1.9, 1.2, 2, 2.3, 2.1])

windFrequencies = likelihood/np.sum(likelihood)


windDirections = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140.,
                           150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270.,
                           280., 290., 300., 310., 320., 330., 340., 350.])

windSpeeds = np.array([5.7, 6.5, 5.6, 5.1, 5.2, 5.5, 4.8, 4.5, 4.4, 4.6, 4.3, 4.5, 4.1,
                       4.8, 5.1, 5., 5.6, 5.5, 5.3, 5.4, 5.5, 5.3, 4.4, 4.3, 3.8, 3.4,
                       4.5, 5.7, 6.2, 5.9, 5.1, 5.3, 4.9, 5.1, 5.5, 6.2])

# Redding CA

likelihood = np.array([5.9, 4.8, 3.4, 1.9, 1.2, 1., 0.7, 0.5, 0.6, 0.5, 0.6, 0.5,
                            0.6, 1.1, 1.3, 2.3, 3.9, 4.1, 3.6, 3.1, 2.7, 1.8, 1.3, 0.8,
                            0.9, 1.1, 1.1, 1.3, 0.9, 1.2, 1.8, 2.3, 3.6, 4.9, 6.5, 7.3])

windFrequencies = likelihood/np.sum(likelihood)


windDirections = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140.,
                           150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270.,
                           280., 290., 300., 310., 320., 330., 340., 350.])

windSpeeds = np.array([3.9, 4., 3.6, 3.6, 3., 2.8, 2.2, 2.4, 2., 2.5, 2.1, 2.1, 2.3, 2.5,
                       3.3, 3.9, 5., 5.2, 4.8, 4.6, 3.7, 3.5, 3., 2.9, 2.9, 3.1, 2.9,
                       2.8, 2.7, 2.7, 2.5, 2.5, 2.6, 2.8, 3.4, 3.8])


plt.show()

"""


# def rect_rule (f, a, b, n):
# 	total = 0.0
# 	dx = (b-a)/float(n)
# 	for k in range (0, n):
#         	total = total + f((a + (k*dx)))
# 	return dx*total
