# -*- coding: utf-8 -*-
"""
PHYS20161 Lecture 10 Quiz, Broadcasting error 4

Illustrates potential issues when broadcasting numpy arrays using fmin

Lloyd Cawthorne 07/12/20
"""

import numpy as np
from scipy.optimize import fmin

OFFSETS = np.array([0., 3., 4.])
SLOPES = np.array([1., 1.5, 2.3])

X_START = 1.
Y_START = 1.


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


def function(x_variable, y_variable, coefficients, offsets):
    """
    Returns <coefficients> * (x_variable + y_variable)^2 + <offsets>
    Parameters
    ----------
    x_variable : float
    y_variable : float
    coefficient : np.array(float)
    offset : np.array(float)
    Returns
    -------
    float
    """
    return (np.average(coefficients) * (x_variable + y_variable)**2
            + np.average(offsets))

def main():
    """
    Minimises function

    Returns
    -------
    int : 0
    """
    results = fmin(lambda x: function(x[0], x[1], SLOPES, OFFSETS),
                   x0=(X_START, Y_START))
    print(results)

    return 0

main()
