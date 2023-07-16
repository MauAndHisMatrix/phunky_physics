# -*- coding: utf-8 -*-
"""
PHYS20161 Week example of lambda function

Demonstrates lambda function compared to normal functions

Lloyd Cawthorne 09/10/20
"""

def polynomial_function(x_variable):
    """
    Computes f(x) = x^2 - 3x + 2 using standard function
    
    Args:
        x_variable: float
    Returns:
        float
    """
    return x_variable**2 - 3 * x_variable + 2


lambda_polynomial = lambda x: x**2 - 3*x + 2
