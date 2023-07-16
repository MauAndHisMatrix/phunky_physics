"""
Useful Lab Functions
"""

# Standard library imports

# Third-party imports
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci
import sympy as sym

# Uni / my modules
import marawan_LSFR
# from Physics.Assignments.marawan_final_assignment import chi_squared, calculate_reduced_chi
# from lab_functions import straight_line_interval
# from assignments.final_assignment import round_to_sig


def calculate_variable_error(variable_expression: Callable,
                             contributing_vars: list[tuple],
                             **kwargs) -> float:
    """
    This function is a generic error calculator. It is based on the general
    form of error propagation found here: https://shorturl.at/fuL19.
    The user feeds the function an expression representing 'f' in the equation
    referenced above. The argument contributing_vars contains information
    about all the variables that contribute to the error calculation. The
    Sympy library facilitates the execution of derivatives by converting the
    expression and variables into algebraic form.

    Parameters:
        variable_expression
        contributing_vars: A list of tuples. Each tuple takes the following
                        form:
                        (variable name (str), variable value, variable error).
                        The variable names are needed by Sympy during the
                        symbolic differentiation stage.

        kwargs: Gives the user the chance to pass through any keyword
                arguments required by the variable expression.
                E.g. a mathematical expression such as a Gaussian that
                contains exp's will need a 'sym_form=True' flag.

    Returns:
        variable error
    """
    error_squared = 0.
    var_symbols = [sym.Symbol(var[0]) for var in contributing_vars]
    for var_name, _, var_error in contributing_vars:
        # Calculates the derivative with respect to the current variable.
        derivative = sym.diff(variable_expression(*var_symbols, **kwargs),
                              sym.Symbol(var_name))

        # The leftover variables in the derivative are sorted in line with
        # the initial variable order.
        free_symbols = [symbol for symbol in var_symbols
                        if symbol in derivative.free_symbols]

        values = [var[1] for var in contributing_vars
                  if sym.Symbol(var[0]) in free_symbols]

        # The derivative is converted into a numpy function.
        numerical_derivative = sym.lambdify(free_symbols, derivative, "numpy")
        deriv_value = numerical_derivative(*values)

        term = (deriv_value)**2 * var_error**2
        error_squared += term

    return np.sqrt(error_squared)


def calculate_reduced_chi(chi_result: float,
                          sample_size: int,
                          free_parameters: int=1):
    """
    This function calculates the reduced chi-squared value of a
    result given the sample size and number of free parameters.

    Parameters:
        chi_result: Chi-squared value
        sample_size: Number of data points
        free_parameters: Number of free parameters in the fitting
                        expression. For example, there are 3 in the case
                        of the BW expression: mass, width and constant.
    """
    deg_of_freedom = sample_size - free_parameters

    return chi_result / deg_of_freedom


def chi_squared(data: np.ndarray,
                fitting_expression: Callable,
                *free_parameters) -> float:
    """
    This function calculates the chi-squared value of data being fitted
    to a given function.

    Parameters:
        data
        fitting_expression: The function that the data is being fitted to.
        free_parameters: The free parameters required in the fitting
                        expression's calculations.
    Returns:
        chi_squared_value
    """
    chi_sq_value = 0.
    for data_point in data:
        prediction = fitting_expression(data_point[0], *free_parameters)
        chi_sq_value += ((prediction - data_point[1]) / data_point[2])**2

    return chi_sq_value


def fit_distribution(distribution: Callable, x, y, errors=None, *params):
    optimals, covariance = sci.curve_fit(distribution, x, y, p0=params, sigma=errors, absolute_sigma=False)
    param_errors = np.sqrt(np.diag(covariance))

    return optimals, param_errors


def chi_analysis(func, x, y, y_error, *free_params, lsfr: bool=False):
    if lsfr:
        labels = {
            'plot_title': 'Chi Analysis of the Optical Rotatory Dispersion Curve',
            'x_label': 'Wavelength of light (nm)',
            'y_label': r'Specific rotation (deg dm$^{-1}$g$^{-1}$ml))'
        }
        marawan_LSFR.main(labels, x, y, y_error)
    else:
        x = np.array(x).reshape((len(x), 1))
        y = np.array(y).reshape((len(y), 1))
        y_error = np.array(y_error).reshape((len(y_error), 1))
        data = np.hstack((x, y, y_error))

        chi_value = chi_squared(data, func, *free_params)
        red_chi = calculate_reduced_chi(chi_value, data.shape[0], len(free_params))

    return chi_value, red_chi