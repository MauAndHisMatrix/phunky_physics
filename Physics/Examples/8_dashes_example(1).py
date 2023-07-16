# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8 more exmaples of plotting

Primarily shows a series of dashed lines.

Lloyd Cawthorne 08/11/19

"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x_variable, mean, standard_deviation):
    """Returns value of gaussian at x centred over mean with standard
    deviation.

    x_variable (float)
    mean (float)
    standard_deviation (float)
    """
    exponent = -(x_variable - mean)**2 / (2 * standard_deviation**2)
    normalisation = (1. / (standard_deviation * np.sqrt(2 * np.pi)))

    return normalisation * np.exp(exponent)


x_values = np.linspace(-2 * np.pi, 2 * np.pi, 100)
means = np.array([0, 2, -1])
standard_deviations = np.array([0.5, 0.3, 0.2])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x_values, gaussian(x_values, means[0], standard_deviations[0]),
        label='\mu = {0:d}, \sigma = {1:.2f}'.format(means[0],
                                                     standard_deviations[0]),
        dashes=[1, 1])

ax.plot(x_values, gaussian(x_values, means[1], standard_deviations[1]),
          label='\mu = {0:d}, \sigma = {1:2.1f}'.
          format(means[1], standard_deviations[1]), dashes=[4, 2])

ax.plot(x_values, gaussian(x_values, means[2], standard_deviations[2]),
          label='\mu = {0:d}, \sigma = {1:2.1f}'.
          format(means[2], standard_deviations[2]), dashes=[8, 1, 2, 4])
ax.legend()
plt.show()
