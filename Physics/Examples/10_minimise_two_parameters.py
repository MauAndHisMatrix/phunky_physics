# -*- coding: utf-8 -*-
"""
PHYS20161 Week 10 example of fitting two parameters

Finds the minimum of f(x,y) = x * exp(-x^2 -y^2) using fmin

Displays result in contour plot.

Lloyd Cawthorne 22/11/19

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

#  Starting parameters
X_START = 0.
Y_START = 0.

#  Function definitions


def mesh_arrays(x_array, y_array):
    """Returns two meshed arrays of size len(x_array)
    by len(y_array)
    x_array array[floats]
    y_array array[floats]
    """
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in y_array:  # PyLint accepts _ as an uncalled variable.
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for dummy_element in x_array:  # PyLint accepts dummy_anything as well.
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh


def function(xy_values):
    """
    Requires array as [x,y]
    Returns x * exp(-x^2 -y^2)
    xy_values = [(float), (float)]
    """
    x_variable = xy_values[0]
    y_variable = xy_values[1]

    return x_variable * np.exp(-x_variable**2 - y_variable**2)


#  Main code

# Initial plot of function
X_VALUES = np.linspace(-2, 2, 100)
Y_VALUES = X_VALUES

X_VALUES_MESH, Y_VALUES_MESH = mesh_arrays(X_VALUES, Y_VALUES)

INITIAL_FIGURE = plt.figure()

INITIAL_PLOT = INITIAL_FIGURE.add_subplot(111)

INITIAL_PLOT.set_title('f(x,y)')
INITIAL_PLOT.set_xlabel('x')
INITIAL_PLOT.set_ylabel('y')

CONTOUR_PLOT = INITIAL_PLOT.contour(X_VALUES_MESH, Y_VALUES_MESH,
                                    function([X_VALUES_MESH, Y_VALUES_MESH]),
                                    20)
CONTOUR_PLOT.clabel()

plt.show()

# Perform fit

FIT_RESULTS = fmin(function, (X_START, Y_START),
                   full_output=True)

[X_MINIMUM, Y_MINIMUM] = FIT_RESULTS[0]
FUNCTION_MINIMUM = FIT_RESULTS[1]

# Plot result of fit

FINAL_FIGURE = plt.figure()

FINAL_PLOT = FINAL_FIGURE.add_subplot(111)

FINAL_PLOT.set_title('f(x,y)')
FINAL_PLOT.set_xlabel('x')
FINAL_PLOT.set_ylabel('y')

MINIMUM_POINT = FINAL_PLOT.scatter(X_MINIMUM, Y_MINIMUM,
                                   color='k', label='Minimum')
plt.legend()
CONTOUR_PLOT_2 = FINAL_PLOT.contour(X_VALUES_MESH, Y_VALUES_MESH,
                                    function([X_VALUES_MESH, Y_VALUES_MESH]),
                                    20)
CONTOUR_PLOT_2.clabel()
plt.savefig('2 parameter fit.png', dpi=300)
plt.show()

print('f({0:.3f},{1:.3f}) = {2:.3f} is the minimum.'.
      format(X_MINIMUM, Y_MINIMUM, FUNCTION_MINIMUM))
