import numpy as np
import matplotlib.pyplot as plt

# optimizing mean
mean = np.array([2.05657127256,1.21874851076,2.05182946781,2.01852735648,1.96914362563,1.8868815221,1.75638065335])*1.e4
var = np.array([1.2763939842,0.189473494294,1.2,1.0,0.8,0.6,0.4])*1.e8

lm = np.min(mean)*1.e4
lv = np.min(var)*1.e8
mm = np.max(mean)*1.e4
mv = np.var(var)*1.e8

plt.plot(mean,var,'o')
plt.plot(lm, lv, 'or')
plt.plot(mm, mv, 'og')
plt.xlabel('mean')#,fontsize=18,family='serif')
plt.ylabel('variance')#,fontsize=18,family='serif')
plt.show()
