# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8: Newton Raphson example

Simple example of implementing Newton Raphson to find sqrt(2);
i.e. root of x^2 - 2

Lloyd Cawthorne 17/10/20

"""

import numpy as np
import matplotlib.pyplot as plt

# Initial parameters

X_START = 4.
ACCURACY = 15 # decimal places solution is accurate to
TOLERANCE = 10**-ACCURACY

# Function definitions


def function(x_variable):
    """Returns x^2 - 2
    x_variable (float)
    """
    return x_variable**2 - 2.


def derivative(x_variable):
    """Returns  2x, derivative of x^2 - 2
    x_variable (float)
    """
    return 2. * x_variable


def next_x(previous_x):
    """
    Returns next value for x according to algorithm

    previous_x (float)

    """

    return previous_x - function(previous_x) / derivative(previous_x)


# Main code

# Main algorithm

# set up parameters
difference = 1
counter = 0

x_root = X_START

# Repeatedly find x_n until the tolerance threshold is met.

while difference > TOLERANCE:

    counter += 1

    x_test = x_root
    x_root = next_x(x_root)

    difference = abs(x_test - x_root)

# Final plot

x_values = np.linspace(0, 5, 100)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.plot(x_values, function(x_values))
ax.plot(x_values, 0 * x_values, c='grey', dashes=[4, 4])
ax.scatter(x_root, function(x_root), c='k', label='x_root')

ax.scatter(X_START, function(X_START), c='red', label='x_0')

ax.set_xlim(0, 5)
ax.set_ylim(-5, 20)
ax.legend()
plt.show()

print('2^0.5 = {0:.{decimals}f}'.format(x_root, decimals=ACCURACY))
print('This took {:d} iterations'.format(counter))
