# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:26:16 2017

@author: msiwek
"""

from scipy import interp # ?
# ???
import numpy as np
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xvals = np.linspace(0, 2*np.pi, 50)
yinterp = np.interp(xvals, x, y)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o') # original data points
plt.plot(xvals, yinterp, '-x') # interpolated data points
plt.show()