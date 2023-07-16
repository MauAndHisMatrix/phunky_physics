# -*- coding: utf-8 -*-
"""
PHYS20161 Week 10 example of fmin to find maximum

Finds maximum of same parabola from week 8 this time using fmin.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

# Fitting Parameters

X_START = -1.5

# Function defintions

def parabola(x_variable):
    """Returns -2x^4 - 4x^2 + 5x - 10
    x_variable (float)
    """
    return -2 * x_variable**4 - 4. * x_variable**2 + 5. * x_variable + 10.

# Below function is redundant if we use lambda when fitting
def negative_parabola(x_variable):
    """Returns -parabola(x)
    x (float)
    """
    return - parabola(x_variable)


def plot_parabola(x_values, parabola_function):
    """Returns plot object for parabola and x values given
    x_values np.array(float)
    parabola function that returns float
    """

    plot, = plt.plot(x_values, parabola_function(x_values))
    plt.xlabel('x values')
    plt.ylabel('f(x)')

    return plot


# Main code

X_VALUES = np.linspace(-2., 2., 100)

# Set initial parameters
X_MAXIMUM = X_START
PARABOLA_MAXIMUM = parabola(X_START)

# Display plot with startig point
plt.scatter(X_MAXIMUM, PARABOLA_MAXIMUM, label='Starting point', color='k')

# Find maximum with fmin
# Find minimum of -function
# FIT_RESULTS = fmin(negative_parabola, X_START, full_output=True)

FIT_RESULTS = fmin(lambda x: -parabola(x), X_START, full_output=True)

X_MAXIMUM = FIT_RESULTS[0][0]
# multiply by minus sign to get back to function of interest
PARABOLA_MAXIMUM = -FIT_RESULTS[1]
COUNTER = FIT_RESULTS[2]

# Plot result
plot_parabola(X_VALUES, parabola)
plt.scatter(X_MAXIMUM, PARABOLA_MAXIMUM, label='Maximum', color='r')
plt.legend()
plt.show()

print('f({0:4.3f}) = {1:5.3f} is the maximum point'.format(X_MAXIMUM,
                                                           PARABOLA_MAXIMUM))
print('This took {0:d} iterations.'.format(COUNTER))
