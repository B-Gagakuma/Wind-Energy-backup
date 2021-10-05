import numpy as np
import _floris
from scipy.interpolate import CubicSpline


def WindFrame(wind_direction, turbineX, turbineY):
    """ Calculates the locations of each turbine in the wind direction reference frame """
    nTurbines = len(turbineX)
    windDirectionDeg = wind_direction
    # adjust directions
    windDirectionDeg = 270. - windDirectionDeg
    if windDirectionDeg < 0.:
        windDirectionDeg += 360.
    windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

    # convert to downwind(x)-crosswind(y) coordinates
    turbineXw = turbineX*np.cos(-windDirectionRad)-turbineY*np.sin(-windDirectionRad)
    turbineYw = turbineX*np.sin(-windDirectionRad)+turbineY*np.cos(-windDirectionRad)

    return turbineXw, turbineYw


def PowWind(Uref, turbineZ, shearExp, zref=50., z0=0.):
    """
    wind shear power law
    """
    nTurbines = len(turbineZ)

    turbineSpeeds = np.zeros(nTurbines)

    for turbine_id in range(nTurbines):
        turbineSpeeds[turbine_id]= Uref*((turbineZ[turbine_id]-z0)/(zref-z0))**shearExp

    return turbineSpeeds


def Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf, yaw=False, Ct=False, kd=0.15, bd=-0.01, initialWakeDisplacement=-4.5,\
            useWakeAngle=False, initialWakeAngle=1.5, ke=0.065, adjustInitialWakeDiamToYaw=False, MU=np.array([0.5, 1.0, 5.5]),\
            useaUbU=True, aU=5.0, bU=1.66, me=np.array([-0.5, 0.22, 1.0]), cos_spread=1.e+12, Region2CT=0.888888888889, axialInduction=False, \
            keCorrCT=0.0, keCorrArray=0.0, axialIndProvided=True, shearCoefficientAlpha=0.10805, shearZh=90.):
    """floris wake model"""
    nTurbines = len(turbineXw)

    if yaw == False:
        yaw = np.zeros(nTurbines)
    if axialInduction == False:
        axialInduction = np.ones(nTurbines)*1./3.
    if Ct == False:
        Ct = np.ones(nTurbines)*4.0*1./3.*(1.0-1./3.)

    # yaw wrt wind dir.
    yawDeg = yaw

    nSamples = 1
    wsPositionXYZw = np.zeros([3, nSamples])

    # call to fortran code to obtain output values
    wtVelocity, wsArray, wakeCentersYT, wakeCentersZT, wakeDiametersT, wakeOverlapTRel = \
                _floris.floris(turbineXw, turbineYw, turbineZ, yawDeg, rotorDiameter, Vinf,
                                               Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                               MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                               Region2CT, keCorrArray, useWakeAngle,
                                               adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU, wsPositionXYZw,
                                               shearCoefficientAlpha, shearZh)

    return wtVelocity, wakeCentersYT, wakeCentersZT, wakeDiametersT, wakeOverlapTRel


def WindDirectionPower(rotorDiameter,ratedPower,wtVelocity,Cp=False,generatorEfficiency=False,cut_in_speed=False,air_density=1.1716):
    """calculate power from a given wind direction"""

    nTurbines = len(rotorDiameter)
    if Cp == False:
        # Cp = np.zeros(nTurbines)+(0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp = np.zeros(nTurbines)+0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
    if generatorEfficiency == False:
        generatorEfficiency = np.ones(nTurbines)
    if cut_in_speed == False:
        cut_in_speed = np.ones(nTurbines)*3.

    rotorArea = 0.25*np.pi*np.power(rotorDiameter, 2)
    # calculate initial values for wtPower (W)
    wtPower = np.zeros(nTurbines)
    for i in range(nTurbines):
        # wtPower[i] = generatorEfficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power((wtVelocity[i]-cut_in_speed[i]), 3))
        wtPower[i] = generatorEfficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power((wtVelocity[i]), 3))
    # adjust units from W to kW
    wtPower /= 1000.0

    if np.any(wtPower) >= (np.any(ratedPower)-100.):
        for i in range(0, nTurbines):
            if (ratedPower[i]-100.) <= wtPower[i] <= (ratedPower[i]+100.):
                x = np.array([(ratedPower[i]-100.),ratedPower[i],(ratedPower[i]+100.)])
                y = np.array([(ratedPower[i]-100.),ratedPower[i]-25.,ratedPower[i]])
                cs = CubicSpline(x,y)
                power = wtPower[i]
                wtPower[i] = cs(power)

            elif wtPower[i] > (ratedPower[i]+100.):
                wtPower[i] = ratedPower[i]

    for i in range(nTurbines):
        if wtVelocity[i] < cut_in_speed[i]:
            wtPower[i] = 0.

    # calculate total power for this direction
    dir_power = np.sum(wtPower)

    # pass out results
    return wtPower, dir_power


def calcAEP_floris(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp):
    nTurbines = len(turbineX)
    nDirections = len(windDirections)

    dir_powers = np.zeros(nDirections)
    for i in range(nDirections):
        turbineXw, turbineYw = WindFrame(windDirections[i], turbineX,turbineY)
        Vinf = PowWind(windSpeeds[i], turbineZ, shearExp)
        wtVelocity,_,_,_,_ = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        _,dir_powers[i] = WindDirectionPower(rotorDiameter,ratedPower,wtVelocity)
    AEP = np.sum(dir_powers*windFrequencies*24.*365.)
    return AEP


def AEP_obj(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp):
    return -calcAEP_floris(turbineX, turbineY, turbineZ, rotorDiameter, ratedPower, windDirections, windSpeeds, windFrequencies, shearExp)



if __name__=="__main__":

    turbineX = np.array([0.,0.,0.])
    turbineY = np.array([0.,500.,1000.])
    turbineZ = np.array([100.,100.,100.])
    rotorDiameter = np.array([100.,100.,100.])
    ratedPower = np.array([5000.,5000.,5000.])


    windSpeeds = np.array([2.,5.,10.,30.])
    ndirs = len(windSpeeds)
    windDirections = np.linspace(0.,360.-360./float(ndirs),ndirs)
    windFrequencies = np.array([0.1,0.2,0.3,0.4])
    windFrequencies = windFrequencies/sum(windFrequencies)

    shearExp = 0.15

    AEP = calcAEP_floris(turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,windDirections,windSpeeds,windFrequencies,shearExp)
    print AEP
    print AEP_obj(turbineX,turbineY,turbineZ,rotorDiameter,ratedPower,windDirections,windSpeeds,windFrequencies,shearExp)

    # import matplotlib.pyplot as plt
    # plt.plot(x,y,'ok')
    # plt.plot(xw,yw,'or')
    # plt.show()
