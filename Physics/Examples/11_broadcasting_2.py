# -*- coding: utf-8 -*-
"""
PHYS20161 Lecture 10 Quiz, Broadcasting error 2

Illustrates potential issues when broadcasting numpy arrays for plots

Lloyd Cawthorne 03/12/20
"""

import numpy as np
import matplotlib.pyplot as plt

OFFSETS = np.array([0., 3., 4.])
SLOPES = np.array([1., 1.5, 2.3])

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
    Creates plot of function.

    Returns
    -------
    int : 0
    """

    x_values = np.linspace(0, 5, 10)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    for index, offset in enumerate(OFFSETS):
        axes.plot(x_values, function(x_values, SLOPES[index], offset))
    plt.show()

    return 0

main()
