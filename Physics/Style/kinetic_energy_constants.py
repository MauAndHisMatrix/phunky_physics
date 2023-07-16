# -*- coding: utf-8 -*-
"""
Title: Basic example that calculates kinetic energy of object given velocity

This code defines a function that returns the classical kinetic energy given
the velocity.

Note: NumPy is imported but mot used merely to show where constants should
reside.

Lloyd Cawthorne 31/01/2020
"""

import numpy as np

# SI units

MASS = 5.0


def kinetic_energy(velocity, mass=MASS):
    """
    Returns the classical kinetic energy (float) given the velocity and
    mass of the object. Assumes mass is constant unless key word argument
    is given otherwise.
    #
    velocity: float
    mass:    float
    """

    return 0.5 * velocity * mass**2
