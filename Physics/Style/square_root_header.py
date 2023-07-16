# -*- coding: utf-8 -*-
"""
Example of function header for PHYS20161

Code consists of single function that outputs roots of second order polynomial.
Also shows how line breaks should be taken over mathematical operations.

Lloyd Cawthorne 29/01/20

"""

import math


def square_root(x_squared_coefficient, x_coefficient, constant):
    """
    Returns the two roots of a second order polymial.

    Coefficients should be given to conform to

    x_squared_coefficient x^2 + x_coefficient x + constant = 0.

    x = (-x_coefficient +/- sqrt[x_coefficient^2 - 4 x_squared_ coeffiecient
                               * constant]) / (2
                                               * x_squared_coefficient)

    Args:
        x_squared_coefficient: float
        x_coefficient: float
        constant: float
    Returns:
        Two solutions in a list: [float, float]
    Raises:
        ZeroDivisionError: If x_squared_coefficient = 0
        ValueError: Math domain error, imaginary solution

    L. Cawthorne 05/02/20
    """

    try:
        square_root_term = math.sqrt(x_coefficient**2 - 4
                                     * x_squared_coefficient
                                     * constant)
        solution_1 = ((-x_coefficient + square_root_term)
                      / (2 * x_squared_coefficient))
        solution_2 = ((-x_coefficient - square_root_term)
                      / (2 * x_squared_coefficient))
        return [solution_1, solution_2]
    except ZeroDivisionError:
        print('x_squared_coefficient cannot be 0.')
        return None
    except ValueError:
        print('No real solutions.')
        return None
