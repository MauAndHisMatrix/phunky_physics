# -*- coding: utf-8 -*-
"""
PHYS20161 Week 10 exmaple of multiple plots

Lloyd Cawthorne 21/11/19

"""
import numpy as np
import matplotlib.pyplot as plt

figure  = plt.figure()

axes_1 = figure.add_subplot(211)

x_values = np.linspace(-2,2,1000)

axes_1.plot(x_values, np.cosh(x_values)-1)
axes_1.plot(x_values, x_values**2 / 2)

axes_2 = figure.add_subplot(212)

axes_2.plot(x_values, np.sinh(x_values))
axes_2.plot(x_values, x_values)

plt.savefig('example_plot_3.png', dpi = 300)
plt.show()
