# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8: Newton Raphson example

Simple example of implementing Newton Raphson to find root of e^2x - x - 6

Need to be careful as starting point will return different values.

Lloyd Cawthorne 17/10/20

"""

import numpy as np
import matplotlib.pyplot as plt

# Initial parameters

#X_START = 0.5
X_START = -1. # returns different result
TOLERANCE = 0.001

#  Function definitions


def function(x_variable):
    """Returns e^2x -x -6
    x (float)
    """
    return np.exp(2. * x_variable) - x_variable - 6.


def derivative(x_variable):
    """Returns 2 e^2x - 1, derivative of e^2x - x - 6
    x_variable (float)
    """
    return 2. * np.exp(2. * x_variable) - 1


def next_x_function(previous_x):
    """
    Returns next value for x according to algorithm

    previous_x (float)

    """

    return previous_x - function(previous_x) / derivative(previous_x)


def newton_raphson(x_start=X_START, tolerance=TOLERANCE,
                   next_x=next_x_function):
    """Iterates Newton Raphson algorithm until difference between succesive
    solutions is less than tolerance.
    Args:
        x_start: float, kwarg
        tolerance: float, kwarg
        next_x: function returning float, kwarg
    Returns:
        x_root: float
        counter: int
    """
    # set up parameters
    difference = 1
    counter = 0
    x_root = x_start

    # Repeatedly find x_n until the tolerance threshold is met.
    while difference > tolerance:

        counter += 1

        x_test = x_root
        x_root = next_x(x_root)

        difference = abs(x_test - x_root)

    return x_root, counter


# Main code

x_values = np.linspace(-10, 5, 100)

x_solution, count = newton_raphson()

# Final plot

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
# Plot function
ax.plot(x_values, function(x_values))
# Plot dashed line on x axis
ax.plot(x_values, 0 * x_values, c='grey', dashes=[4, 4])
ax.scatter(x_solution, function(x_solution), c='k', label='x_root')

ax.scatter(X_START, function(X_START), c='red', label='x_0')


ax.set_xlim(-10, 5)
ax.set_ylim(-6, 6)
ax.legend()
plt.show()

print('Root = {:.3f}'.format(x_solution))
print('This took {:d} iterations'.format(count))
