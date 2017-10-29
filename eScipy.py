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

# sparce matrices
from scipy.sparse import csc_matrix
a = csc_matrix([[1,0,0,0],[0,0,10,11],[0,0,0,99]])
a
# print whole array
print(a) # sparse matrix
a.todense() # dense matrix
a.toarray() # numpy array
# print a row
print(a[1,:])
a.todense()[1,:]
a.toarray()[1,:]
# print a column
print(a[:,2])
a.todense()[:,2]
a.toarray()[:,2][:,np.newaxis] # necessary to preserve the number of dimensions

