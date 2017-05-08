# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:07:18 2017

@author: msiwek
"""

# unpacking numpy array
import numpy as np
x = np.arange(15).reshape(5, 3)
x
def add_cols(a, b, c):
    return a+b+c
add_cols(*x.T)
def f(a,b,c):
    print(a,b,c)
f(*x.T)

# interpolation
import numpy as np
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xvals = np.linspace(0, 2*np.pi, 50)
yinterp = np.interp(xvals, x, y)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o') # original data points
plt.plot(xvals, yinterp, '-x') # interpolated data points
plt.show()