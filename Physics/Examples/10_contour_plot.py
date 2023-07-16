# -*- coding: utf-8 -*-
"""
PHYS20161 example of contour plotting

Plots f(x,y) = 2x^2 + y^2

Lloyd Cawthorne 21/11/19

"""

import numpy as np
import matplotlib.pyplot as plt


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

    for dummy_element in x_array:  # Pylint accepts dummy_anything as well.
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh


def function(x_variable, y_variable):
    """Returns 2x^2 + y^2
    x_variable (float)
    y_variable (float)
    """
    return 2 * x_variable**2 + y_variable**2


X_VALUES = np.linspace(-10, 10, 100)
Y_VALUES = X_VALUES.copy()

X_VALUES_MESH, Y_VALUES_MESH = mesh_arrays(X_VALUES, Y_VALUES)

FIGURE = plt.figure(figsize=(4, 4))

AXIS = FIGURE.add_subplot(111)

AXIS.contour(X_VALUES_MESH, Y_VALUES_MESH,
             function(X_VALUES_MESH, Y_VALUES_MESH))

plt.savefig('contour_1.png', dpi=300)
plt.show()
