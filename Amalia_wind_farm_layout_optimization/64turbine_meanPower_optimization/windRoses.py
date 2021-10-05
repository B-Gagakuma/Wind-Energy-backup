import numpy as np
from akima import Akima, akima_interp


def amaliaRose(nDirections):
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
                               1.10562900e-02, 1.04521380e-02, 1.16201970e-02, 1.10562700e-02])

    windDirections = np.linspace(0.,360.-360./float(len(windSpeeds)), len(windSpeeds))
    spline_freq = Akima(windDirections, windFrequencies)
    spline_speed = Akima(windDirections, windSpeeds)
    num = nDirections
    dirs = np.linspace(0.,360.-360./float(num), num)
    ddir = dirs[1]-dirs[0]

    frequencies = np.zeros(num)
    speeds = np.zeros(num)

    num_int = 1000

    dir_int1 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int1 = np.zeros(num_int/2)
    speed_freq_int1 = np.zeros(num_int/2)
    for j in range(num_int/2):
        freq_int1[j],_,_,_ = spline_freq.interp(dir_int1[j])
        ws,_,_,_ = spline_speed.interp(dir_int1[j])
        speed_freq_int1[j] = freq_int1[j]*ws

    dir_int2 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int2 = np.zeros(num_int/2)
    speed_freq_int2 = np.zeros(num_int/2)
    for j in range(num_int/2):
        freq_int2[j],_,_,_ = spline_freq.interp(dir_int2[j])
        ws,_,_,_ = spline_speed.interp(dir_int2[j])
        speed_freq_int2[j] = freq_int2[j]*ws

    frequencies[0] = np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2)
    speeds[0] = (np.trapz(speed_freq_int1,dir_int1)+np.trapz(speed_freq_int2,dir_int2))/\
        (np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2))

    for i in range(1,num):
        dir_int = np.linspace(dirs[i]-ddir/2.,dirs[i]+ddir/2.,num_int)
        freq_int = np.zeros(num_int)
        speed_freq_int = np.zeros(num_int)
        for j in range(num_int):
            freq_int[j],_,_,_ = spline_freq.interp(dir_int[j])
            ws,_,_,_ = spline_speed.interp(dir_int[j])
            speed_freq_int[j] = freq_int[j]*ws
        frequencies[i] = np.trapz(freq_int,dir_int)
        speeds[i] = np.trapz(speed_freq_int,dir_int)/np.trapz(freq_int,dir_int)

    frequencies = frequencies/sum(frequencies)
    for i in range(len(frequencies)):
        if speeds[i] < 0.:
            speeds[i] = 0.
        if frequencies[i] < 0.:
            frequencies[i] = 0.

    return dirs, frequencies, speeds


def denverRose(nDirections):

    windSpeeds = np.array([5.7, 6.5, 5.6, 5.1, 5.2, 5.5, 4.8, 4.5, 4.4, 4.6, 4.3, 4.5, 4.1,
                           4.8, 5.1, 5., 5.6, 5.5, 5.3, 5.4, 5.5, 5.3, 4.4, 4.3, 3.8, 3.4,
                           4.5, 5.7, 6.2, 5.9, 5.1, 5.3, 4.9, 5.1, 5.5, 6.2])

    likelihood = np.array([2.8, 2.8, 2., 2.3, 2.5, 3., 2.5, 2.5, 2.5, 2.6, 2.7, 1.9,
                           2.1, 2., 2.6, 3., 4.4, 4.4, 2.9, 3.1, 4.2, 5., 4., 3.8, 3.4,
                           2.3, 2.8, 2.3, 2.4, 2.1, 1.9, 1.9, 1.2, 2, 2.3, 2.1])

    windFrequencies = likelihood/np.sum(likelihood)

    windDirections = np.linspace(0.,360.-360./float(len(windSpeeds)), len(windSpeeds))
    spline_freq = Akima(windDirections, windFrequencies)
    spline_speed = Akima(windDirections, windSpeeds)

    num = nDirections
    dirs = np.linspace(0.,360.-360./float(num), num)
    ddir = dirs[1]-dirs[0]

    frequencies = np.zeros(num)
    speeds = np.zeros(num)

    num_int = 1000

    dir_int1 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int1 = np.zeros(num_int/2)
    speed_freq_int1 = np.zeros(num_int/2)

    for j in range(num_int/2):
        freq_int1[j],_,_,_ = spline_freq.interp(dir_int1[j])
        ws,_,_,_ = spline_speed.interp(dir_int1[j])
        speed_freq_int1[j] = freq_int1[j]*ws

    dir_int2 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int2 = np.zeros(num_int/2)
    speed_freq_int2 = np.zeros(num_int/2)

    for j in range(num_int/2):
        freq_int2[j],_,_,_ = spline_freq.interp(dir_int2[j])
        ws,_,_,_ = spline_speed.interp(dir_int2[j])
        speed_freq_int2[j] = freq_int2[j]*ws

    frequencies[0] = np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2)
    speeds[0] = (np.trapz(speed_freq_int1,dir_int1)+np.trapz(speed_freq_int2,dir_int2))/\
        (np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2))

    for i in range(1,num):
        dir_int = np.linspace(dirs[i]-ddir/2.,dirs[i]+ddir/2.,num_int)
        freq_int = np.zeros(num_int)
        speed_freq_int = np.zeros(num_int)
        for j in range(num_int):
            freq_int[j],_,_,_ = spline_freq.interp(dir_int[j])
            ws,_,_,_ = spline_speed.interp(dir_int[j])
            speed_freq_int[j] = freq_int[j]*ws
        frequencies[i] = np.trapz(freq_int,dir_int)
        speeds[i] = np.trapz(speed_freq_int,dir_int)/np.trapz(freq_int,dir_int)

    frequencies = frequencies/sum(frequencies)
    for i in range(len(frequencies)):
        if speeds[i] < 0.:
            speeds[i] = 0.
        if frequencies[i] < 0.:
            frequencies[i] = 0.

    return dirs, frequencies, speeds



def decaturRose(nDirections):

    windSpeeds = np.array([4.6, 4.6, 4.5, 4.6, 4.1, 4.4, 4.5, 4.9, 5.2, 4.9, 4.8, 4.7, 4.2,
                          3.8, 4.1, 4., 4.3, 5., 5.5, 5.8, 6.5, 5.6, 5, 4.5, 4.6, 4.2, 4.1,
                          4.5, 4.8, 5.5, 4.8, 5.1, 5.5, 5.7, 4.8, 5.6])

    likelihood = np.array([2.1, 1.5, 1.7, 1.9,	2., 2.5, 2.7, 2.9, 2.8, 2.5, 2.5, 1.8,
                           1.7, 1.6, 1.4, 1.2, 2.3, 3.1, 4.6, 4.8, 4.7, 5.3, 3.6, 2.9,
                           2.3, 1.8, 2.3, 1.9, 3, 3.5, 3., 3.6, 2.8, 2.6, 1.8, 2.6])

    windFrequencies = likelihood/np.sum(likelihood)

    windDirections = np.linspace(0.,360.-360./float(len(windSpeeds)), len(windSpeeds))
    spline_freq = Akima(windDirections, windFrequencies)
    spline_speed = Akima(windDirections, windSpeeds)

    num = nDirections
    dirs = np.linspace(0.,360.-360./float(num), num)
    ddir = dirs[1]-dirs[0]

    frequencies = np.zeros(num)
    speeds = np.zeros(num)

    num_int = 1000

    dir_int1 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int1 = np.zeros(num_int/2)
    speed_freq_int1 = np.zeros(num_int/2)

    for j in range(num_int/2):
        freq_int1[j],_,_,_ = spline_freq.interp(dir_int1[j])
        ws,_,_,_ = spline_speed.interp(dir_int1[j])
        speed_freq_int1[j] = freq_int1[j]*ws

    dir_int2 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int2 = np.zeros(num_int/2)
    speed_freq_int2 = np.zeros(num_int/2)

    for j in range(num_int/2):
        freq_int2[j],_,_,_ = spline_freq.interp(dir_int2[j])
        ws,_,_,_ = spline_speed.interp(dir_int2[j])
        speed_freq_int2[j] = freq_int2[j]*ws

    frequencies[0] = np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2)
    speeds[0] = (np.trapz(speed_freq_int1,dir_int1)+np.trapz(speed_freq_int2,dir_int2))/\
        (np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2))

    for i in range(1,num):
        dir_int = np.linspace(dirs[i]-ddir/2.,dirs[i]+ddir/2.,num_int)
        freq_int = np.zeros(num_int)
        speed_freq_int = np.zeros(num_int)
        for j in range(num_int):
            freq_int[j],_,_,_ = spline_freq.interp(dir_int[j])
            ws,_,_,_ = spline_speed.interp(dir_int[j])
            speed_freq_int[j] = freq_int[j]*ws
        frequencies[i] = np.trapz(freq_int,dir_int)
        speeds[i] = np.trapz(speed_freq_int,dir_int)/np.trapz(freq_int,dir_int)

    frequencies = frequencies/sum(frequencies)
    for i in range(len(frequencies)):
        if speeds[i] < 0.:
            speeds[i] = 0.
        if frequencies[i] < 0.:
            frequencies[i] = 0.

    return dirs, frequencies, speeds


def reddingRose(nDirections):

    windSpeeds = np.array([3.9, 4., 3.6, 3.6, 3., 2.8, 2.2, 2.4, 2., 2.5, 2.1, 2.1, 2.3, 2.5,
                           3.3, 3.9, 5., 5.2, 4.8, 4.6, 3.7, 3.5, 3., 2.9, 2.9, 3.1, 2.9,
                           2.8, 2.7, 2.7, 2.5, 2.5, 2.6, 2.8, 3.4, 3.8])

    likelihood = np.array([5.9, 4.8, 3.4, 1.9, 1.2, 1., 0.7, 0.5, 0.6, 0.5, 0.6, 0.5,
                           0.6, 1.1, 1.3, 2.3, 3.9, 4.1, 3.6, 3.1, 2.7, 1.8, 1.3, 0.8,
                           0.9, 1.1, 1.1, 1.3, 0.9, 1.2, 1.8, 2.3, 3.6, 4.9, 6.5, 7.3])

    windFrequencies = likelihood/np.sum(likelihood)

    windDirections = np.linspace(0.,360.-360./float(len(windSpeeds)), len(windSpeeds))
    spline_freq = Akima(windDirections, windFrequencies)
    spline_speed = Akima(windDirections, windSpeeds)

    num = nDirections
    dirs = np.linspace(0.,360.-360./float(num), num)
    ddir = dirs[1]-dirs[0]

    frequencies = np.zeros(num)
    speeds = np.zeros(num)

    num_int = 1000

    dir_int1 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int1 = np.zeros(num_int/2)
    speed_freq_int1 = np.zeros(num_int/2)

    for j in range(num_int/2):
        freq_int1[j],_,_,_ = spline_freq.interp(dir_int1[j])
        ws,_,_,_ = spline_speed.interp(dir_int1[j])
        speed_freq_int1[j] = freq_int1[j]*ws

    dir_int2 = np.linspace(dirs[0],dirs[0]+ddir/2.,num_int/2)
    freq_int2 = np.zeros(num_int/2)
    speed_freq_int2 = np.zeros(num_int/2)

    for j in range(num_int/2):
        freq_int2[j],_,_,_ = spline_freq.interp(dir_int2[j])
        ws,_,_,_ = spline_speed.interp(dir_int2[j])
        speed_freq_int2[j] = freq_int2[j]*ws

    frequencies[0] = np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2)
    speeds[0] = (np.trapz(speed_freq_int1,dir_int1)+np.trapz(speed_freq_int2,dir_int2))/\
        (np.trapz(freq_int1,dir_int1)+np.trapz(freq_int2,dir_int2))

    for i in range(1,num):
        dir_int = np.linspace(dirs[i]-ddir/2.,dirs[i]+ddir/2.,num_int)
        freq_int = np.zeros(num_int)
        speed_freq_int = np.zeros(num_int)
        for j in range(num_int):
            freq_int[j],_,_,_ = spline_freq.interp(dir_int[j])
            ws,_,_,_ = spline_speed.interp(dir_int[j])
            speed_freq_int[j] = freq_int[j]*ws
        frequencies[i] = np.trapz(freq_int,dir_int)
        speeds[i] = np.trapz(speed_freq_int,dir_int)/np.trapz(freq_int,dir_int)

    frequencies = frequencies/sum(frequencies)
    for i in range(len(frequencies)):
        if speeds[i] < 0.:
            speeds[i] = 0.
        if frequencies[i] < 0.:
            frequencies[i] = 0.

    return dirs, frequencies, speeds
