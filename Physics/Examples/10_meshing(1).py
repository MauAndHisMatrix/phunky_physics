# -*- coding: utf-8 -*-
"""
PHYS20161 Week 10 example of creating a mesh

Creates a mesh of x and y arrays as needed for plotting
over two variables.

Lloyd Cawthorne 22/11/19

"""

import numpy as np

# Even values between 0 and 10
X_VALUES = np.arange(0, 11, 2)
# Odd values between 0 and 20
Y_VALUES = np.arange(1, 21, 2)

X_VALUES_MESH = np.empty((0, len(X_VALUES)))

# Create a len(x_values) by len(y_values) array of x_values.
# I.e. stack len(y_values) rows of x_values together
for i in range(len(Y_VALUES)):
    X_VALUES_MESH = np.vstack((X_VALUES_MESH, X_VALUES))

print('Meshed x values:')
print(X_VALUES_MESH, '\n')

# y_values_mesh more complicated as we want to stack columns

Y_VALUES_MESH = np.empty((0, len(Y_VALUES)))

# Create a len(y_values) by len(x_values) array of y_values.
# I.e. stack len(x_values) rows of y_values together
for i in range(len(X_VALUES)):
    Y_VALUES_MESH = np.vstack((Y_VALUES_MESH, Y_VALUES))

# Transpose result

Y_VALUES_MESH = np.transpose(Y_VALUES_MESH)
print('Meshed y values:')
print(Y_VALUES_MESH)

# Both 2D arrays should be the same size.
