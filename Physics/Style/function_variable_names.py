# -*- coding: utf-8 -*-
"""
Example of good and bad variable names in Python for PHYS20161

This particular example considers defining the area of a triangle.

Lloyd Cawthorne 06/02/20
"""

import numpy as np


def a(b, h):
    """
    Returns the area of a triangle, float
    Args:
        b, base (float)
        h, height (float)
    """
    return b * h / 2


def area(base, height):
    """
    Returns the area of a triangle, float
    Args:
        base (float)
        height (float)
    """
    return base * height / 2


def areat(base, height):
    """
    Returns the area of a triangle, float
    Args:
        base (float)
        height (float)
    """
    return base * height / 2


def area_triangle(base, height):
    """
    Returns the area of a triangle, float
    #
    This is the naming convention you should adopt.
    #
    Args:
        base (float)
        height (float)
    """
    return base * height / 2


def area_square(side):
    """
    Returns the area of a square, float
     #
    Args:
        side (float)
    """
    return side**2


def area_circle(radius):
    """
    Returns the area of a cirle, float
    #
    Args:
        radius
    """
    return np.pi * radius**2


def area_rhombus(diagonal_1, diagonal_2):
    """
    Returns the area of a rhombus, float
    #
    Args:
        diagonal_1 (float)
        diagonal_2 (float)
    """
    return diagonal_1 * diagonal_2 / 2


def area_pentagon(side):
    """
    Returns the area of a regular pentagon, float
    #
    Args:
        side (float)
    """
    return 0.25 * np.sqrt(5 * (5 + 2 * np.sqrt(5))) * side**2
