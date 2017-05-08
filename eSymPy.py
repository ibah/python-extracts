# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:32:56 2017
SymPy Tutorial
http://docs.sympy.org/dev/tutorial/index.html
"""

"""
Basic Operations

SymPy objects are immutable.
"""

from sympy import *
x, y, z = symbols("x y z")
'''substitution'''
expr = cos(x) + 1
expr
expr.subs(x, y)
# evaluating expression at a point
expr.subs(x, 0)
# Replacing a subexpression with another subexpression
# e.g. some symmetry in expresssion we want to build
expr = x**y
expr
expr = expr.subs(y, x**y)
expr
expr = expr.subs(y, x**x)
expr
# e.g. to perform a very controlled simplification, or perhaps a simplification that SymPy is otherwise unable to do
expr = sin(2*x) + cos(2*x)
expr
expand_trig(expr) # SymPy automatic expansion/simplification
expr.subs(sin(2*x), 2*sin(x)*cos(x)) # manual simplification using substitution
# fact: subs returns new object
expr = cos(x)
expr.subs(x, 0)
expr
x
# multiple substitutions at once
expr = x**3 + 4*x*y - z
expr.subs([(x, 2), (y, 4), (z, 0)]) # multiple substitution
# combining this with a list comprehension (multiple multiple substitutions)
expr = x**4 - 4*x**3 + 4*x**2 - 2*x + 3
expr
replacements = [(x**i, y**i) for i in range(5) if i % 2 == 0]
# -> replace all instances of xx that have an even power with yy
expr.subs(replacements)

"""Converting Strings to SymPy Expressions"""
str_expr = "x**2 + 3*x - 1/2"
expr = sympify(str_expr)
expr
expr.subs(x, 2)
# evalf: evaluate a numerical expression into a floating point number
expr = sqrt(8)
expr
expr.evalf()
# evaluate floating point expressions to arbitrary precision
pi.evalf(100)
# To numerically evaluate an expression with a Symbol at a point
expr = cos(2*x)
expr
expr.evalf(subs={x: 2.4})
# removing roundoff errors
one = cos(1)**2 + sin(1)**2
(one - 1).evalf()
(one - 1).evalf(chop=True)
# lambdify: evaluate an expression at many points
# The easiest way to convert a SymPy expression to an expression that can be numerically evaluated
import numpy
a = numpy.arange(10)
expr = sin(x)
expr
f = lambdify(x, expr, "numpy") 
f(a)
f = lambdify(x, expr, "math")
f(0.1)
# To use lambdify with numerical libraries that it does not know about, pass a dictionary of sympy_name:numerical_function pairs
def mysin(x):
    """
    My sine. Note that this is only accurate for small x.
    """
    return x
f = lambdify(x, expr, {"sin":mysin})
f(0.1)
