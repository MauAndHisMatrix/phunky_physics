# -*- coding: utf-8 -*-
"""
PHYS20161 Lecture 10 Quiz, Broadcasting error 2 minimisation

Illustrates potential issues when broadcasting numpy arrays for plots

Lloyd Cawthorne 07/12/20
"""

import numpy as np
from scipy.optimize import fmin

OFFSETS = np.array([0., 3., 4.])
SLOPES = np.array([1., 1.5, 2.3])

X_START = 1.

def function(x_variable, coefficient, offset):
    """
    Returns coefficient * x_variable^2 + offset
    Parameters
    ----------
    x_variable : float
    coefficient : float
    offset : float
    Returns
    -------
    float
    """
    return coefficient * x_variable**2 + offset


def main():
    """
    Minimises function

    Returns
    -------
    int : 0
    """
    results = fmin(function, X_START, args=(SLOPES, OFFSETS))

    print(results)

    return 0

main()
