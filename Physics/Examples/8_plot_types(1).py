# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8 Examples of plots

Showcases different plot types

Lloyd Cawthorne 02/09/21

"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111)

x_values = np.linspace(-np.pi, np.pi, 10)

ax.plot(x_values, np.sin(x_values), marker='x')

ax.scatter(x_values, 0.1 + np.sin(x_values), marker='v', color='r')

ax.errorbar(x_values, np.sin(x_values) - 0.1, yerr=np.arange(10)/10, fmt='D')

plt.show()
