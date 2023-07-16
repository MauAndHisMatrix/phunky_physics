# -*- coding: utf-8 -*-
"""
PHYS20161 Lecture 10 Quiz, Broadcasting error

Illustrates potential issues when broadcasting numpy arrays

Lloyd Cawthorne 03/12/20
"""

import numpy as np

def function(x_variable, y_variable, z_variable):
    """
    Returns r^2 = x^2 + y^2 + z^2
    Parameters
    ----------
    x_variable : float
    y_variable : float
    z_variable : float
    Returns
    -------
    float
    """
    return x_variable**2+ y_variable**2 + z_variable**2


x_values = np.linspace(0, 10, 10)
y_values = np.linspace(0, 20, 10)
z_values = np.linspace(0, 30, 10)

function_values = function(x_values, y_values, z_values)

print(x_values)
print(y_values)

print(function_values)
