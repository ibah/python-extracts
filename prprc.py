# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:24:33 2017

@author: a
"""

import numpy as np
import pandas as pd

# dummy variables
s = pd.Series(list('abca'))
s
x = pd.get_dummies(s)
x # strange, it is displayed as float
# all levels are represented (so there's collinearity)
type(x.ix[0,0]) # np.float64
s1 = ['a','b', np.nan]
pd.get_dummies(s1)
pd.get_dummies(s1, dummy_na=True) # NaN as a separate value
df = pd.DataFrame({'A': list('abc'),
                   'B': list('bac'),
                   'C': range(1,4)})
pd.get_dummies(df, prefix='x')
# -> bad, here we have columns with identical names...
#    you can use it only when the set of possible values are
#    disparate / no overlapping of values.
pd.get_dummies(df, prefix=['col1', 'col2'])
# -> good, every column has different name
pd.get_dummies(pd.Series(list('abcaa')))
pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
# -> a is dropped, this is to remove collinearity between the columns


