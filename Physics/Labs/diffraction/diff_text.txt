"""
Diffraction Script
"""

# Standard library imports
import glob
import sys

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

SINGLE_SLIT_1_Z_1 = "single_slit/first_slit_z_2_1.csv"
SINGLE_SLIT_1_Z_2 = "single_slit/first_slit_z_2_2.csv"
SINGLE_SLIT_1_Z_3 = "single_slit/first_slit_z_3.csv"
SINGLE_SLIT_6_Z_1 = "single_slit/second_slit_z_1.csv"
SINGLE_SLIT_6_Z_2 = "single_slit/second_slit_z_2.csv"
SINGLE_SLIT_6_Z_3 = "single_slit/second_slit_z_3_1.csv"

DOUBLE_12 = "multi_slit/slit12_z39p4.csv"
TRIPLE_14 = "multi_slit/slit14_z39p4.csv"
TRIPLE_16 = "multi_slit/slit16_z39p4.csv"
QUAD_15 = "multi_slit/slit15_z43p4.csv"

LASER_LAMBDA = 672e-9 # m
SPEED = 4.47e-6 # m/s
SPEED_ERROR = 5.52e-8 # m/s


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


def import_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=2)

    return data


def plot_data(data, x, y: list, yerr: list=None, plot_metadata=None):

    fig_name = ""
    if plot_metadata:
        plot_title, x_label, y_label, titles, fig_name = plot_metadata

    fig = plt.figure(0)
    axes1 = fig.add_subplot(111)

    axes1.plot(x, y, label="Line of best fit", linewidth=2, color="red")
    axes1.errorbar(data[:, 0], data[:, 1], yerr=yerr, fmt='.', color='black', label="Experimental data", alpha=0.1)
    # axes1.scatter(data[:, 0], data[:, 1], color='black', label="Experimental data", alpha=0.2, s=4)

    axes1.set_ylim(ymin=0)

    axes1.set_xlabel(x_label)
    axes1.set_ylabel(y_label)
    axes1.set_title(plot_title)


    axes1.ticklabel_format(scilimits=(-3, 4))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=900)
    plt.show()


def intensity_eq_2(time, peak_intensity, coeff, shift):
    x_val = coeff * (time + shift)

    sinc_term = np.sin(x_val) / x_val

    intensity = peak_intensity * sinc_term**2

    return intensity


def double_intensity(time, peak, a_coeff, d_coeff, shift):
    x_val_a = a_coeff * (time + shift)
    x_val_d = d_coeff * (time + shift)
 
    cos_term = np.cos(x_val_d)
    sinc_term = np.sin(x_val_a) / x_val_a

    intensity = peak * cos_term**2 * sinc_term**2

    return intensity


def multi_intensity(time, peak, a_coeff, d_coeff, shift):

    x_val_a = a_coeff * (time + shift)
    x_val_d = d_coeff * (time + shift)

    sinc_term = np.sin(x_val_a) / x_val_a
    sin_term = np.sin(x_val_d)
    N_sin_term = np.sin(NN * x_val_d)

    big_sin = (N_sin_term / sin_term)**2

    front_coeff = (peak / NN**2)

    intensity = front_coeff * sinc_term**2 * big_sin

    return intensity


def solve_slit_width_a(coeff, z_distance, speed):
    return (coeff * LASER_LAMBDA * z_distance) / (np.pi * speed)


def main(file_path, z_distance, measured_a_slit_width, plot_metadata=None):
    data = import_data(file_path)

    # data = data[::10, :]
    # print(data)

    errors = np.array([0.05]*data.shape[0])

    a_estimate = measured_a_slit_width * ((np.pi*SPEED) / (LASER_LAMBDA*z_distance))
    x_max, y_max = data[np.where(data[:, 1] == max(data[:, 1]))][0]

    (peak_intensity, coeff, shift), (peak_error, coeff_error, shift_error) = fit_distribution(intensity_eq_2,
                                                                                data[:, 0],
                                                                                data[:, 1],
                                                                                errors,
                                                                                y_max,
                                                                                a_estimate,
                                                                                - 6.7)
    coeff = np.abs(coeff)

    print(f"\nFitted amplitude: {peak_intensity} +/- {peak_error}")
    print(f"Fitted coeff: {coeff} +/- {coeff_error}")
    print(f"Fitted shift: {shift} +/- {shift_error}\n")


    slit_width_a = solve_slit_width_a(coeff, z_distance, SPEED)

    a_error = calculate_variable_error(solve_slit_width_a,
                                       [("coeff", coeff, coeff_error),
                                        ("z", z_distance, 0.1e-2),
                                        ("speed", SPEED, SPEED_ERROR)])
    

    print(f"Slit width a = {slit_width_a} +/- {a_error} m")


    x_data = data[:, 0]
    y_data = intensity_eq_2(data[:, 0], peak_intensity, coeff, shift)
    plot_data(data, x_data, y_data, plot_metadata=plot_metadata)


def main2(file_path, z_distance, measured_a_slit_width, plot_metadata=None):
    data = import_data(file_path)
    # data *= 1e3
    # data[:, 3] *= 1e-13
    errors = np.array([0.05]*data.shape[0])

    (peak_intensity, a_coeff, d_coeff, shift), errors = fit_distribution(double_intensity,
                                                                                data[:, 0],
                                                                                data[:, 1],
                                                                                errors,
                                                                                max(data[:, 1]),
                                                                                0.37,
                                                                                0.37,
                                                                                -76)
    peak_error, a_error, d_error, shift_error = errors

    print(f"Fitted amplitude: {peak_intensity} +/- {peak_error}")
    print(f"Fitted a coeff: {a_coeff} +/- {a_error}")
    print(f"Fitted d coeff: {d_coeff} +/- {d_error}")
    print(f"Fitted shift: {shift} +/- {shift_error}\n")


    slit_width_a = solve_slit_width_a(a_coeff, z_distance, SPEED)

    a_error = calculate_variable_error(solve_slit_width_a,
                                       [("coeff", a_coeff, a_error),
                                        ("z", z_distance, 0.1e-2),
                                        ("speed", SPEED, SPEED_ERROR)])
    

    print(f"Slit width a = {slit_width_a} +/- {a_error} m")


    x_data = data[:, 0]
    y_data = double_intensity(data[:, 0], peak_intensity, a_coeff, d_coeff, shift)
    plot_data(data, x_data, y_data, plot_metadata=plot_metadata)


def main3(file_path, z_distance, measured_a_slit_width, plot_metadata=None):
    data = import_data(file_path)
    # data *= 1e3
    # data[:, 3] *= 1e-13
    errors = None

    x_max, intensity_max = data[np.where(data[:, 1] == max(data[:, 1]))][0]

    a_estimate = measured_a_slit_width * ((np.pi*SPEED) / (LASER_LAMBDA*z_distance))

    (peak_intensity, a_coeff, d_coeff, shift), errors = fit_distribution(multi_intensity,
                                                                                data[:, 0],
                                                                                data[:, 1],
                                                                                errors,
                                                                                intensity_max,
                                                                                0.37,
                                                                                - 0.09,
                                                                                - 150)
    peak_error, a_error, d_error, shift_error = errors

    print(f"Fitted amplitude: {peak_intensity} +/- {peak_error}")
    print(f"Fitted a coeff: {a_coeff} +/- {a_error}")
    print(f"Fitted d coeff: {d_coeff} +/- {d_error}")
    print(f"Fitted shift: {shift} +/- {shift_error}\n")

    x_data = data[:, 0]
    y_data = double_intensity(data[:, 0], peak_intensity, a_coeff, d_coeff, shift)
    # y_data = double_intensity(data[:, 0], 0.6, 0.37, 0.37, -132)
    plot_data(data, x_data, y_data)

# Number of slits
NN = 3

if __name__ == "__main__":
    # main(SINGLE_SLIT_1_Z_1, 15.3e-2, 0.02e-2)
    # main(SINGLE_SLIT_1_Z_2, 32e-2, 0.02e-2)
    # main(SINGLE_SLIT_1_Z_3, 45.3e-2, 0.02e-2)
    
    # main(SINGLE_SLIT_6_Z_1, 15.3e-2, 0.015e-2, ["Diffraction pattern for single slit number 6",
    #                                             "Time (s)",
    #                                             "Intensity (V)",
    #                                             "",
    #                                             "slit_6.png"])
    # main(SINGLE_SLIT_6_Z_2, 32e-2, 0.015e-2)
    # main(SINGLE_SLIT_6_Z_3, 45.3-2, 0.015e-2)

    main2(DOUBLE_12, 39.4e-2, 0.007, ["Diffraction pattern for double slit number 12",
                                                "Time (s)",
                                                "Intensity (V)",
                                                "",
                                                "slit_12.png"])

    # main3(TRIPLE_16, 39.4e-2, 0.004)
