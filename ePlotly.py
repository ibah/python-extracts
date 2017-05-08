# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:20:48 2017
Plotly
https://plot.ly/python/
You can set up Plotly to work in online or offline mode, or in jupyter notebooks.

Getting started, follow these steps
https://plot.ly/python/getting-started/
https://plot.ly/python/line-and-scatter/
"""



# https://plot.ly/python/getting-started/

import plotly
plotly.__version__
plotly.tools.set_credentials_file(username='ibah@o2.pl', api_key='BPTFCnr5yt68Cfio52T2')
"""
The initialization step places a special .plotly/.credentials file in your home directory. Your ~/.plotly/.credentials file should look something like this [...]
Privacies: public, private, secret
For PRO users
import plotly
plotly.tools.set_config_file(world_readable=False,
                             sharing='private')
"""

"""
Special Instructions for Plotly On-Premise Users

Your API key for account on the public cloud will be different than the API key in Plotly On-Premise. Visit https://plotly.your-company.com/settings/api/ to find your Plotly On-Premise API key. Remember to replace "your-company.com" with the URL of your Plotly On-Premise server. If your company has a Plotly On-Premise server, change the Python API endpoint so that it points to your company's Plotly server instead of Plotly's cloud.
"""

"""
Start Plotting Online
Two methods for plotting online: py.plot() and py.iplot(). Both options create a unique url for the plot and save it in your Plotly account.
- py.plot() to return the unique url and optionally open the url.
- py.iplot() when working in a Jupyter Notebook to display the plot in the notebook.
"""

# py.plot -> online (plotly account)
import plotly.plotly as py
from plotly.graph_objs import *
trace0 = Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = Data([trace0, trace1])
py.plot(data, filename = 'basic-line')
help(py.plot)

# py.iplot -> jupyter notebook
import plotly.plotly as py
from plotly.graph_objs import *
trace0 = Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17]
)
trace1 = Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = Data([trace0, trace1])
py.iplot(data, filename = 'basic-line')

"""
Initialization for Offline Plotting

Plotly Offline allows you to create graphs offline and save them locally
- Use plotly.offline.plot() to create and standalone HTML that is saved locally and opened inside your web browser.
- Use plotly.offline.iplot() when working offline in a Jupyter Notebook to display the plot in the notebook.
"""

# offline.plot -> save html file
import plotly
from plotly.graph_objs import Scatter, Layout
plotly.offline.plot({
    "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": Layout(title="hello world")
})

# offline.iplot -> offline jupyter notebook
import plotly
from plotly.graph_objs import Scatter, Layout
plotly.offline.init_notebook_mode(connected=True)
# -> required at the start of every session
plotly.offline.iplot({
    "data": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": Layout(title="hello world")
})


# <--------------------------------------------------------











# https://plot.ly/python/line-and-scatter/



import plotly.plotly as py
import plotly.graph_objs as go

# Simple Scatter Plot
# Create random data with numpy
import numpy as np
N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)
# Create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)
data = [trace]
# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')
# -> see: https://plot.ly/~ibah/0/
# or plot with:
# plot_url = py.plot(data, filename='basic-line')







"""
Couldn't find a 'username', 'api-key' pair for you on your local machine. To sign in temporarily (until you stop running Python), run:
import plotly.plotly as py
py.sign_in('username', 'api_key')

Even better, save your credentials permanently using the 'tools' module:
import plotly.tools as tls
tls.set_credentials_file(username='username', api_key='api-key')

For more help, see https://plot.ly/python.
"""