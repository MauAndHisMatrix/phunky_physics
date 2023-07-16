# -*- coding: utf-8 -*-
"""
PHYS20161 Week 9: Volume of sphere zone optional problem

Problem:
    A sphere of radius 1 is cut by two parallel planes into three parts. The
    volume of the part between the two planes is exactly half the volume of
    the whole sphere. What is the smallest possible distance between the two
    planes?

Solution:
    Smallest possible distance implies the planes are symmetric about the
    centre. Futhermore, the volume of each spherical cap will be equal to
    one quarter of the volume of the sphere.

This code finds height of one cap, using Newton Raphson, then solves for the
the distance between the planes.

This is presented in a functional style that we encourage you to work towards.

Lloyd Cawthorne 19/05/20
"""

import numpy as np
import matplotlib.pyplot as plt

# Initial parameters

TOLERANCE = 0.001
RADIUS = 1.
HEIGHT_START = RADIUS / 2.  # Must be between 0 and radius for sensible results
CAP_VOLUME = RADIUS**3 * np.pi / 3.

#  Functions


def volume_spherical_cap(height, radius=RADIUS, cap_volume=CAP_VOLUME):
    """Returns volume of spherical cap minus desired volume given height
    and radius.
    Args:
        height: float
        radius: kwarg, float
    """
    return (np.pi / 3.) * height**2 * (3 * radius - height) - cap_volume


def derivative_volume_spherical_cap(height, radius=RADIUS):
    """Returns the derivative of the volume of a spherical cap w.r.t. height.
    Args:
        height: float
        radius, kwarg, float
    """
    return np.pi / 3. * (2 * height * (3 * radius * height) - height**2)


def next_x_function(previous_x, function=volume_spherical_cap,
                    derivative=derivative_volume_spherical_cap):
    """
    Returns next value for x according to Newton Raphson algorithm

    previous_x (float)
    function (function)
    derivative (function)
    """

    return previous_x - function(previous_x) / derivative(previous_x)


def newton_raphson(x_start=HEIGHT_START, tolerance=TOLERANCE,
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


def solution():
    """This function calls all the commands above, produces a plot of the fit
    prints the desired answer.
    """
    x_root, counter = newton_raphson()

    x_values = np.linspace(0, RADIUS, 100)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(x_values, volume_spherical_cap(x_values))
    plt.plot(x_values, 0 * x_values, c='grey', dashes=[4, 4])
    plt.scatter(x_root, volume_spherical_cap(x_root), c='k', label='x_root')

    plt.scatter(HEIGHT_START, volume_spherical_cap(HEIGHT_START), c='red',
                label='x_0')

    plt.xlim(0, RADIUS)
    plt.ylim(-RADIUS**3, RADIUS**3)
    plt.legend()
    plt.show()

    print('h = {:6.5f}'.format(x_root))
    print('This took {:d} iterations'.format(counter))

    print('The minimum distance between the planes is {:6.5f}.'.
          format(2*(RADIUS-x_root)))
    return None


#  Main code

solution()
