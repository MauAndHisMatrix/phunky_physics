# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8 more exmaples of plotting

Iterate plots with properties set in arrays..

Structure follows more proffesional style in that constants are defined
between import statements and function definitions.

Lloyd Cawthorne 08/11/19

"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters for three gaussians
MEANS = np.array([0, 2, -1, 3])
STANDARD_DEVIATIONS = np.array([0.5, 0.3, 0.2, 0.6])
DASHES = [[1, 1], [4, 2], [8, 1, 2, 4], [2, 2]]


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

fig = plt.figure()
ax = fig.add_subplot(111)

# Iterate the three plots.
for index, mean in enumerate(MEANS):

    ax.plot(x_values, gaussian(x_values, mean,
                                STANDARD_DEVIATIONS[index]),
             dashes=DASHES[index],
             label=r'\mu = {0:d}, \sigma = {1:2.1f}'.
             format(mean, STANDARD_DEVIATIONS[index]))

ax.legend()
plt.show()
