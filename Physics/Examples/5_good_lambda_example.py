# -*- coding: utf-8 -*-
"""
PHYS20161 Week 5 exmaples of lambda 2

Better use of lambda functions

Lloyd Cawthorne 09/10/20
"""

def power_0(x_variable):
    """
    Returns x^0

    Parameters
    ----------
    x_variable :float

    Returns
    -------
    float

    """
    return x_variable**0


def power_1(x_variable):
    """
    Returns x^1

    Parameters
    ----------
    x_variable :float

    Returns
    -------
    float

    """
    return x_variable**1


def power_2(x_variable):
    """
    Returns x^2

    Parameters
    ----------
    x_variable :float

    Returns
    -------
    float

    """
    return x_variable**2


def power_3(x_variable):
    """
    Returns x^3

    Parameters
    ----------
    x_variable :float

    Returns
    -------
    float

    """
    return x_variable**3


power_list = [power_0, power_1, power_2, power_3]


lambda_power_list = [lambda x: 1,
                     lambda x: x,
                     lambda x: x**2,
                     lambda x: x**3]
