# -*- coding: utf-8 -*-
"""
PHYS20161 Week 10 fmin example

Find minimum of polynomial using inbuilt funciton fmin

Lloyd Cawthorne 20/11/19

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

#  Fitting parameters

X_START = -1.5

# Function definitions


def parabola(x_variable):
    """Returns x^4 + 4x^3 - 5x - 10
    x_variable (float)
    """
    return x_variable**4 + 4. * x_variable**2 + 10.


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
X_MINIMUM = X_START
PARABOLA_MINIMUM = parabola(X_START)

# Display plot with startig point
plt.scatter(X_MINIMUM, PARABOLA_MINIMUM, label='Starting point', color='k')

# find minimum

FIT_RESULTS = fmin(parabola, X_START, full_output=True, disp=0)

X_MINIMUM = FIT_RESULTS[0][0]
PARABOLA_MINIMUM = FIT_RESULTS[1]
COUNTER = FIT_RESULTS[2]

# Plot result
plot_parabola(X_VALUES, parabola)
plt.scatter(X_MINIMUM, PARABOLA_MINIMUM, label='Minimum', color='r')
plt.legend()
plt.show()

print('f({0:4.3f}) = {1:5.3f} is the minimum point'.format(X_MINIMUM,
                                                           PARABOLA_MINIMUM))
print('This took {0:d} iterations.'.format(COUNTER))
