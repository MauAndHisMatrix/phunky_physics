# -*- coding: utf-8 -*-
"""
PHYS20161 Week 10 exmaple of figure object

PLots two lines using figure and axes objects

Lloyd Cawthorne 21/11/19
"""

import numpy as np
import matplotlib.pyplot as plt

FIGURE = plt.figure()

AXES = FIGURE.add_subplot(111)

X_VALUES = np.linspace(-2, 2, 1000)

AXES.plot(X_VALUES, np.cosh(X_VALUES) - 1)
AXES.plot(X_VALUES, X_VALUES**2 / 2)

plt.savefig('example_plot.png', dpi=300)
plt.show()
