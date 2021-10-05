import numpy as np


def turbineSpacingSquared(turbineX, turbineY):
    """
    Calculates inter-turbine spacing for all turbine pairs
    """
    nTurbines = len(turbineX)
    separation_squared = np.zeros((nTurbines-1)*nTurbines/2)

    k = 0
    for i in range(0, nTurbines):
        for j in range(i+1, nTurbines):
            separation_squared[k] = (turbineX[j]-turbineX[i])**2+(turbineY[j]-turbineY[i])**2
            k += 1
    return separation_squared


def SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=2.0):
    """
    inter turbine spacing constraint
    """
    nTurbines = len(rotorDiameter)
    separation_squared = turbineSpacingSquared(turbineX, turbineY)
    spacing_con = np.zeros(int((nTurbines-1)*nTurbines/2))

    k = 0
    for i in range(0, nTurbines):
        for j in range(i+1, nTurbines):
            spacing_con[k] = separation_squared[k] - (0.5*minSpacing*rotorDiameter[i]+0.5*minSpacing*rotorDiameter[j])**2
            k += 1
    return spacing_con


def circularBoundary(turbineX, turbineY, circle_radius, circle_center=np.array([0.,0.])):
    """
    circular boundary constraint
    """
    nTurbines = len(turbineX)
    circle_boundary_constraint = np.zeros(nTurbines)
    for i in range(nTurbines):
        R = np.sqrt((turbineX[i]-circle_center[0])**2+(turbineY[i]-circle_center[1])**2)
        circle_boundary_constraint[i] = circle_radius-R
    return circle_boundary_constraint


if __name__=="__main__":

    turbineX = np.array([0.,0.,0.])
    turbineY = np.array([0.,500.,1000.])
    rotorDiameter = np.array([100.,100.,100.])

    print SpacingConstraint(turbineX, turbineY, rotorDiameter,minSpacing=2.)
    print circularBoundary(turbineX, turbineY, 10000.)
