# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:20:13 2017
Visualisations in Python
(extracts)
"""

"""
Sample data sets
http://deparkes.co.uk/2016/11/11/python-sample-datasets/
1. CSV from Internet
import pandas as pd
data = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
2. Seaborn library
import seaborn.apionly as sns
iris = sns.load_dataset('iris')
3. Sklearn package
from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
column_names = iris.feature_names
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

%matplotlib inline
plt.rcParams['figure.figsize'] = [10,6]
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# scatter plot
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


# histogrtam, density plot
https://plot.ly/matplotlib/histograms/
http://seaborn.pydata.org/tutorial/distributions.html
x = np.random.normal(size=100)
plt.hist(x)
sns.kdeplot(x)
sns.distplot(x)
sns.distplot(x, kde=False, rug=True)
sns.distplot(x, bins=20, kde=False, rug=True)




# categorical data
# <sns>
# 1
x = np.array(['a','b','a','b','a','a','c'])
sns.countplot(x)
# 2
# sns.set(style="ticks")
exercise = sns.load_dataset("exercise")
# ticks
sns.factorplot(x="time", y="pulse", data=exercise)
sns.factorplot(x="time", y="pulse", hue="kind", data=exercise)
# violin
sns.factorplot(x="time", y="pulse", hue="kind", data=exercise, kind="violin")
# adding facets
sns.factorplot(x="time", y="pulse", hue="kind", col="diet", data=exercise)
sns.factorplot(x="time", y="pulse", hue="kind", col="diet", data=exercise, size=5, aspect=.8)
titanic = sns.load_dataset("titanic")
g = sns.factorplot("alive", col="deck", col_wrap=4,
                   data=titanic[titanic.deck.notnull()],
                   kind="count", size=2.5, aspect=.8)
g = sns.factorplot(x="age", y="embark_town",
                   hue="sex", row="class",
                   data=titanic[titanic.embark_town.notnull()],
                   orient="h", size=2, aspect=3.5, palette="Set3",
                   kind="violin", split=True, cut=0, bw=.2)
# <plt>
plt.bar([0,1,2], np.unique(x, return_counts=True)[1])
unique, counts = np.unique(x, return_counts=True)
plt.bar(range(len(unique)), counts, tick_label=unique)

# preparations
Counter(x)
unique, counts = np.unique(x, return_counts=True)
dict(zip(unique,counts))
plt.bar(range(len(unique)),counts)

# full example
gender = ['male','male','female','male','female']
import matplotlib.pyplot as plt
from collections import Counter
c = Counter(gender)
men = c['male']
women = c['female']
bar_heights = (men, women)
x = (1, 2)
fig, ax = plt.subplots()
width = 0.4
ax.bar(x, bar_heights, width)
ax.set_xlim((0, 3))
ax.set_ylim((0, max(men, women)*1.1))
ax.set_xticks([i+width/2 for i in x])
ax.set_xticklabels(['male', 'female'])
plt.show()






# Density plot
import seaborn as sns
data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
plt.hist(data)
sns.set_style('whitegrid')
sns.kdeplot(np.array(data), bw=0.5)
sns.distplot(data)



# Boxplot
# Boxplot from Pandas groupby object
df = pd.DataFrame(np.random.random(10), index=list('abaabbaabb'), columns=['x'])
# plt
tmp = df.groupby(df.index).x.apply(lambda x: x.values)
plt.boxplot(tmp, labels=tmp.index)
# sns
sns.boxplot(df.index, df.x)
# plt - filled in
data = [np.random.normal(0, std, 1000) for std in range(1, 6)]
plt.boxplot(data, notch=True, patch_artist=True)
plt.show()
# plt - controling the colors
data = [np.random.normal(0, std, 1000) for std in range(1, 6)]
box = plt.boxplot(data, notch=True, patch_artist=True)
colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()




# Ploting vecotrs
def plot_vectors(vs):
    """Plot vectors in vs assuming origin at (0,0)."""
    n = len(vs)
    X, Y = np.zeros((n, 2))
    U, V = np.vstack(vs).T
    plt.quiver(X, Y, U, V, range(n), angles='xy', scale_units='xy', scale=1)
    xmin, xmax = np.min([U, X]), np.max([U, X])
    ymin, ymax = np.min([V, Y]), np.max([V, Y])
    xrng = xmax - xmin
    yrng = ymax - ymin
    xmin -= 0.05*xrng
    xmax += 0.05*xrng
    ymin -= 0.05*yrng
    ymax += 0.05*yrng
    plt.axis([xmin, xmax, ymin, ymax])
A, B = [1,3], [5,4]
plot_vectors([A, B])
# checking
X, Y = np.zeros((2,2))
U, V = np.vstack([A,B]).T
plt.quiver(X, Y, U, V, [2,3], angles='xy', scale_units='xy', scale=1) # colors, scale
# -> angles:
#    'xy': the arrow points from (x,y) to (x+u, y+v)
xmin, xmax = np.min([U,X])-0.5, np.max([U,X])+0.5
ymin, ymax = np.min([V,Y])-0.5, np.max([V,Y])+0.5
plt.axis([xmin, xmax, ymin, ymax])





# Plotting functions and formulas
# plot basic functions and lines
x = np.arange(-5,5, 0.1);
y = (x**3-2*x+7)/(x**4+2)
plt.plot(x, y)
plt.xlim(-4, 4); plt.ylim(-.5, 4)
plt.xlabel('x'); plt.ylabel('y')
plt.axhline(0); plt.axvline(1)
# plt.axhspan(2, 3.5, xmin=-3, xmax=-1)
plt.title('Example Function')
plt.show()
# plot a function
from scipy.stats import norm
def plot_func(func, x_range=(-1,1)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = func(x)
    plt.plot(x, y)
    plt.show()
plot_func(norm.pdf)
# plot a formula
def plot_form(form, x_range=(-1,1)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = eval(form)
    plt.plot(x, y)
    plt.show()
plot_form('x**2')





# Heatmaps
from sklearn.datasets import load_iris; import pandas as pd
iris = load_iris(); data = iris.data; column_names = iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# plot
sns.heatmap(df.corr())
# customization
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.corr(),
            linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
            linecolor='white', annot=True)





# Pair plots
from sklearn.datasets import load_iris; import pandas as pd
iris = load_iris(); data = iris.data; column_names = iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# adding the target column
df = df.assign(Species=iris.target)
df['Species'] = df.Species.astype('category')
df['Species'].cat.categories = iris.target_names
df.head()
# plot
sns.pairplot(df)
sns.pairplot(df, hue='Species')
sns.pairplot(df, hue='Species',
             diag_kind='kde')
sns.pairplot(df, hue='Species',
             diag_kind='kde',
             diag_kws=dict(shade=True))
sns.pairplot(df, hue='Species',
             diag_kind='kde',
             diag_kws=dict(shade=True),
             plot_kws=dict(s=10))
g = sns.pairplot(df,
        hue='Species', palette = 'seismic', size=1.2, diag_kind = 'kde',
        diag_kws=dict(shade=True), plot_kws=dict(s=10) )
g.set(xticklabels=[])




# using log scale
t = np.arange(0.01, 20.0, 0.01)
# one plot
plt.semilogx(t, np.sin(2*np.pi*t))
# four plots (different axes)
plt.subplots_adjust(hspace=0.4)
# log y axis
plt.subplot(221)
plt.semilogy(t, np.exp(-t/5.0))
plt.title('semilogy')
plt.grid(True)
# log x axis
plt.subplot(222)
plt.semilogx(t, np.sin(2*np.pi*t))
plt.title('semilogx')
plt.grid(True)
# log x and y axis
plt.subplot(223)
plt.loglog(t, 20*np.exp(-t/10.0), basex=2)
plt.grid(True)
plt.title('loglog base 2 on x')
# with errorbars: clip non-positive values
ax = plt.subplot(224)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
x = 10.0**np.linspace(0.0, 2.0, 20)
y = x**2.0
plt.errorbar(x, y, xerr=0.1*x, yerr=5.0 + 0.75*y)
ax.set_ylim(ymin=0.1)
ax.set_title('Errorbars go negative')
plt.show()
# both axes in log scale
plt.loglog(t, np.exp(t), 'r:', label='Some funny dots');plt.legend();plt.title('The great plot');plt.xlabel('ants');plt.ylabel('Germans')







# ribbons
x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2*np.pi*x)
y2 = 1.2*np.sin(4*np.pi*x)

# one ribbon
plt.fill_between(x, 0, y1)

# 3 ribbons
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.fill_between(x, 0, y1)
ax1.set_ylabel('between y1 and 0')
ax2.fill_between(x, y1, 1)
ax2.set_ylabel('between y1 and 1')
ax3.fill_between(x, y1, y2)
ax3.set_ylabel('between y1 and y2')
ax3.set_xlabel('x')

# Now fill between y1 and y2 where a logical condition is met. Note this is different than calling
#   fill_between(x[where], y1[where],y2[where]
# because of edge effects over multiple contiguous regions.
fig, (ax, ax1) = plt.subplots(2, 1, sharex=True)
ax.plot(x, y1, x, y2, color='black')
ax.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
ax.set_title('fill between where')
# Test support for masked arrays.
y2 = np.ma.masked_greater(y2, 1.0)
ax1.plot(x, y1, x, y2, color='black')
ax1.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
ax1.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
ax1.set_title('Now regions with y2>1 are masked')

# This example illustrates a problem; because of the data
# gridding, there are undesired unfilled triangles at the crossover
# points.  A brute-force solution would be to interpolate all
# arrays to a very fine grid before plotting.
# show how to use transforms to create axes spans where a certain condition is satisfied
fig, ax = plt.subplots()
y = np.sin(4*np.pi*x)
ax.plot(x, y, color='black')
# use the data coordinates for the x-axis and the axes coordinates for the y-axis
import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
theta = 0.9
ax.axhline(theta, color='green', lw=2, alpha=0.5)
ax.axhline(-theta, color='red', lw=2, alpha=0.5)
ax.fill_between(x, 0, 1, where=y > theta, facecolor='green', alpha=0.5, transform=trans)
ax.fill_between(x, 0, 1, where=y < -theta, facecolor='red', alpha=0.5, transform=trans)
plt.show()








# multiple plots
from sklearn.datasets import load_iris; import pandas as pd
iris = load_iris(); data = iris.data; column_names = iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# plot
plt.subplot(1,2,1) # numrows, numcols, fignum
plt.plot(df.filter(regex='length')) # but no legend here and same colors...
plt.subplot(1,2,2)
plt.plot(df.filter(regex='width'))
plt.tight_layout()

# exammple
# plt.figure()
plt.subplot(211)
plt.plot([1,2,3],[2,3,1], 'r*-', label='Red line')
plt.legend()
plt.subplot(212)
plt.plot([3,4,5],[2,4,6], 'cp-', label='Cyan line')
plt.legend()
#plt.show()

# example - shared axes
a, b = plt.subplots(2, 1, sharex=True, sharey=True)
b[0].plot([1,2,3],[2,3,1], 'r*-', label='Red line')
b[1].plot([3,4,5],[2,4,6], 'cp-', label='Cyan line')
b[0].legend(); b[1].legend()

"""
Subplots

http://matplotlib.org/examples/pylab_examples/subplots_demo.html
"""
# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)
# Just a figure and one subplot
f, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')
# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, y)
axarr[0].set_title('Sharing X axis')
axarr[1].scatter(x, y)
# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)
# Three subplots sharing both x/y axes
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing both axes')
ax2.scatter(x, y)
ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)





# plotting in Pandas
from sklearn.datasets import load_iris; import pandas as pd
iris = load_iris(); data = iris.data; column_names = iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# plots
df.filter(regex='length').plot()  # see nice legends
df.filter(regex='width').plot()




# Plotly
import plotly; plotly.__version__
from plotly.graph_objs import Scatter, Figure, Layout
plot([Scatter(x=[1, 2, 3], y=[3, 1, 6])])







"""

3D plots

"""

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# matplotlib - contour plots
plt.contour(X, Y, Z)
plt.contour(X, Y, Z, 1) # number of countour lines
plt.contour(X, Y, Z, 10)
plt.contour(X, Y, Z, 100)
plt.contour(X, Y, Z, np.linspace(Z.min(),Z.max(),10)) # ditto, 10
plt.contour(X, Y, Z, np.linspace(Z.min(),Z.max(),100)) # ditto, 100
plt.contourf(X, Y, Z)
plt.contourf(X, Y, Z, 1) # number of countour lines (borders)
plt.contourf(X, Y, Z, 2)
plt.contourf(X, Y, Z, 10)
plt.contourf(X, Y, Z, 100)


# matplotlib - Axes3D - Surface Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)#,
                       #linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


##################################################

# matplotlib - annotations

import numpy as np
import matplotlib.pyplot as plt
# Example data
a = np.arange(0,3, .02)
b = np.arange(0,3, .02)
c = np.exp(a)
d = c[::-1]
# Create plots with pre-defined labels.
# Alternatively, you can pass labels explicitly when calling `legend`.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c+d, 'k', label='Total message length')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.show()

# rotate tick labels
g = sns.countplot(data.loan_status)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
# Adjusting Labels

# automatic rotation of labels
df = pd.DataFrame(np.random.random(3),
                  index=['kjfhflkjdahjheajklehjkhaekjh',
                         'fjlhadjkhfljkhdlkjhadlkhgljkakhdka',
                         'aalkljharlkjlakjkljkerjaaj'], columns=['x'])
fig, ax = plt.subplots(1)
plt.bar(range(3), df.x, tick_label=df.index)
plt.title('Some nice plot')
fig.autofmt_xdate()

# shortening names
def shorten(x):
    if len(x) > 22:
        x = x[:6] + '..' + x[-11:]
    return x
v_shorten = np.vectorize(shorten)









"""
================    ===============================
character           description
================    ===============================
``'-'``             solid line style
``'--'``            dashed line style
``'-.'``            dash-dot line style
``':'``             dotted line style
``'.'``             point marker
``','``             pixel marker
``'o'``             circle marker
``'v'``             triangle_down marker
``'^'``             triangle_up marker
``'<'``             triangle_left marker
``'>'``             triangle_right marker
``'1'``             tri_down marker
``'2'``             tri_up marker
``'3'``             tri_left marker
``'4'``             tri_right marker
``'s'``             square marker
``'p'``             pentagon marker
``'*'``             star marker
``'h'``             hexagon1 marker
``'H'``             hexagon2 marker
``'+'``             plus marker
``'x'``             x marker
``'D'``             diamond marker
``'d'``             thin_diamond marker
``'|'``             vline marker
``'_'``             hline marker
================    ===============================

==========  ========
character   color
==========  ========
'b'         blue
'g'         green
'r'         red
'c'         cyan
'm'         magenta
'y'         yellow
'k'         black
'w'         white
==========  ========

Legend
        ===============   =============
        Location String   Location Code
        ===============   =============
        'best'            0
        'upper right'     1
        'upper left'      2
        'lower left'      3
        'lower right'     4
        'right'           5
        'center left'     6
        'center right'    7
        'lower center'    8
        'upper center'    9
        'center'          10
        ===============   =============

"""

"""
seaborn

palette
deep, muted, bright, pastel, dark, colorblind
Other options:
hls, husl, any named matplotlib palette, list of colors
"""