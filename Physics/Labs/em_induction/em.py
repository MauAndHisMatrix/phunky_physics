"""
EM Induction Script
"""

# Standard library imports
import glob

# Third-party imports
# from IPython.display import display
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
import scipy.optimize as sci
# import lmfit as lm

# Uni / my modules
# import marawan_LSFR
# from uni_made import marawan_LSFR_20
# from lab_functions import straight_line_interval
# from Physics.Assignments.marawan_final_assignment import round_to_sig
# from modules_by_marawan.useful_lab_funcs import chi_analysis, fit_distribution

PERSPEX = "perspex.csv"
SOLID_COPPER = "solid_copper.txt"
LAM_COPPER = "laminated_copper.txt"
SOLID_STEEL = "solid_steel.txt"
LAM_STEEL = "laminated_steel.txt"


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
        red_chi = calculate_reduced_chi(chi_value, data.shape[0])

    return chi_value, red_chi


def plot_data(x, y: list, yerr: list, plot_metadata, perspex=False):

    plot_title, x_label, y_label, titles, fig_name = plot_metadata
    y_label_1, y_label_2 = y_label.split(',')
    title_1, title_2 = titles.split(',')

    # x *= 1e3
    # y = [array * 1e3 for array in y]
    # yerr = [array * 1e3 for array in yerr]


    fig, (axes1, axes2) = plt.subplots(1, 2)
    fig.suptitle(plot_title)

    axes1.plot(x, y[0])
    axes1.errorbar(x, y[0], yerr=yerr[0], fmt='.', color='black')

    axes1.set_ylim(ymin=0)

    axes1.set_xlabel(x_label)
    axes1.set_ylabel(y_label_1)
    axes1.set_title(title_1)

    axes1.ticklabel_format(scilimits=(-3, 4))
    #-----------------------------------------------------------------
    axes2.plot(x, y[1])
    axes2.errorbar(x, y[1], yerr=yerr[1], fmt='.', color='black')

    axes2.set_ylim(ymin=0)
    if perspex:
        axes2.set_ylim(ymax=2*np.mean(y[1]))

    axes2.set_xlabel(x_label)
    axes2.set_ylabel(y_label_2)
    axes2.set_title(title_2)
    # Use standard form for numbers outside 10^m - 10^n for (m , n)
    axes2.ticklabel_format(scilimits=(-3, 4))

    plt.tight_layout()
    plt.savefig(fig_name, dpi=900)
    plt.show()


def import_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    return data


def R_eq(R_3_values, R2):
    return (R2 * R_3_values) / R4


def L_eq(R3_values, r_values, R2, C):
    return ((C * R2) / R4) * (R3_values*R4 + r_values*R3_values + r_values*R4)


def skin(R, freq):
    mu = 4 * np.pi * 1e-7
    A = 1.825e-4
    L = 0.273

    return np.sqrt((R * A) / (L * np.pi * mu * freq))


def main(file_path, plot_metadata, perspex=False):
    data = import_data(file_path)
    data *= 1e3
    data[:, 3] *= 1e-15

    R1 = R_eq(data[:, 1], data[:, 4])
    L = L_eq(data[:, 1], data[:, 2], data[:, 4], data[:, 3])
    sk = skin(R1, data[:, 0])

    plot_data(data[:, 0], [R1, sk], [None, None], plot_metadata, perspex=perspex)


R4 = 9.996e3


if __name__ == "__main__":
    main(PERSPEX, [
        "Perspex Core",
        "Frequency (Hz)",
        r"R$_{1}$ (Ohms),Skin Depth (m)",
        r"Frequency against R$_{1}$,Frequency against Skin Depth",
        "perspex_plot_skin.png"
    ],
    True)
    # main(SOLID_COPPER, [
    #     "Solid Copper Core",
    #     "Frequency (Hz)",
    #     r"R$_{1}$ (Ohms),Inductance (H)",
    #     r"Frequency against R$_{1}$,Frequency against Inductance",
    #     "solid_copper_plot.png"
    # ],
    # False)
    # main(LAM_COPPER, [
    #     "Laminated Copper Core",
    #     "Frequency (Hz)",
    #     r"R$_{1}$ (Ohms),Inductance (H)",
    #     r"Frequency against R$_{1}$,Frequency against Inductance",
    #     "laminated_copper_plot.png"
    # ],
    # False)
    # main(SOLID_STEEL, [
    #     "Solid Steel Core",
    #     "Frequency (Hz)",
    #     r"R$_{1}$ (Ohms),Inductance (H)",
    #     r"Frequency against R$_{1}$,Frequency against Inductance",
    #     "solid_steel_plot.png"
    # ],
    # False)
    # main(LAM_STEEL, [
    #     "Laminated Steel Core",
    #     "Frequency (Hz)",
    #     r"R$_{1}$ (Ohms),Inductance (H)",
    #     r"Frequency against R$_{1}$,Frequency against Inductance",
    #     "laminated_steel_plot.png"
    # ],
    # False)