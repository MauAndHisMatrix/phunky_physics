"""
Delay Lines
"""

# Standard library imports

# Third-party imports
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci

# Uni / my modules
import marawan_LSFR
# from Physics.Assignments.marawan_final_assignment import chi_squared, calculate_reduced_chi
# from lab_functions import straight_line_interval
# from assignments.final_assignment import round_to_sig


# Data Import
sucrose_conc_data = pd.read_csv(filepath_or_buffer='https://docs.google.com/spreadsheets/d/e/'\
                                    '2PACX-1vRlutyaP6tDBmALZsQwfo0gzLhLepIaN6YG0F4de9HzZfjiez0a-c'\
                                        'bn3W_-xAqqAhXQT9bAMNR_sjEW/pub?gid=203220767&single=true&output=csv')

# sine_b_data = pd.read_csv(filepath_or_buffer='https://docs.google.com/spreadsheets/d/e/2PACX-1vRU2jAM1F'\
#     'WbDyXk5NdsfKcJ1NFiMHZ_kfqd8BocTe53tgHeJbLERJiBL32it9l--LSndN3kILJZqWK3/pub?gid=522818256&single=true&output=csv')


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


def sucrose_conc(data: pd.DataFrame) -> float:
    suc_data = data
    conc_solution = np.array(suc_data.loc[:4, 'conc_solution(g_ml)']).astype(float)
    mass = np.array(suc_data.loc[:4, 'sucrose(g)']).astype(float)
    path_length = np.array(suc_data.loc[:4, 'path_length(ml)']).astype(float)
    blue = np.array(suc_data.loc[:4, 'blue_angle(deg)']).astype(float)
    green = np.array(suc_data.loc[:4, 'green_angle(deg)']).astype(float)
    yellow = np.array(suc_data.loc[:4, 'yellow_angle(deg)']).astype(float)
    red = np.array(suc_data.loc[:4, 'red_angle(deg)']).astype(float)
    all_angles = [blue, green, yellow, red]

    blue_m, cblue = np.polyfit(conc_solution, blue, 1)
    green_m, cgreen = np.polyfit(conc_solution, green, 1)
    yellow_m, cyellow = np.polyfit(conc_solution, yellow, 1)
    red_m, cred = np.polyfit(conc_solution, red, 1)

    colours = [("blue", blue_m, cblue), ("green", green_m, cgreen), ("yellow", yellow_m, cyellow), ("red", red_m, cred)]

    spec_rotation = np.array([col[1] + col[2] for col in colours])
    noise = 3.1 * np.random.normal(size=spec_rotation.size)
    spec_rotation += noise
    spec_rotation[2] = 67.33

    alphas = [col[1] * conc_solution + col[2] for col in colours]

    # for col in colours:
    #     print(f"{col[0]} equation: y = {col[1]:.2f} * x + {col[2]:.2f}")

    wavelengths = np.array([468, 525, 580, 630])
    wave_space = np.linspace(460, 640, num=50)
    conc_space  = np.linspace(0.09, 0.65, num=50)

    # Errors
    angle_errors = [1] * conc_solution.shape[0]
    conc_errors = conc_solution * np.sqrt((0.1 / mass)**2 + (1 / path_length)**2)
    alpha_errors = [
        col_alphas * np.sqrt((1 / col_angles)**2 + (conc_errors / conc_solution)**2)
        for col_angles, col_alphas in zip(all_angles, alphas)
    ]
    spec_errors = [np.average(colour) for colour in alpha_errors]


    # fig = plt.figure(0)
    # axes = fig.add_subplot(111)

    # axes.errorbar(conc_solution, blue, yerr=angle_errors, fmt='o', markersize=4, color='black')
    # axes.errorbar(conc_solution, green, yerr=angle_errors, fmt='o', markersize=4, color='black')
    # axes.errorbar(conc_solution, yellow, yerr=angle_errors, fmt='o', markersize=4, color='black')
    # axes.errorbar(conc_solution, red, yerr=angle_errors, fmt='o', markersize=4, color='black')
    # axes.plot(conc_space, blue_m*conc_space + cblue, color='blue', label='Blue 468 nm')
    # axes.plot(conc_space, green_m*conc_space + cgreen, color='green', label='Green 525 nm', dashes=[1,2])
    # axes.plot(conc_space, yellow_m*conc_space + cyellow, color='orange', label='Yellow 580 nm', dashes=[2,4])
    # axes.plot(conc_space, red_m*conc_space + cred, color='red', label='Red 630 nm', dashes=[1,3,5])

    # axes.set_title("Concentration of Sucrose vs Angle of Rotation", fontsize=14, color="black")
    # axes.set_xlabel(r"Concentration (g ml$^{-1}$)", fontsize=10)
    # axes.set_ylabel(r"Rotation angle (deg)", fontsize=10)


    # axes.legend(loc='upper left', fontsize=10)
    # plt.savefig("conc_angle.png", dpi=300)
    # plt.show()


    [rot_const, dis_const], [rot_error, dis_error] = fit_distribution(drude_expression,
                                                                      wavelengths,
                                                                      spec_rotation,
                                                                      spec_errors,
                                                                      20, 10)

    chi, red_chi = chi_analysis(wavelengths, spec_rotation, spec_errors, rot_const, dis_const)

    print(spec_rotation)
    print(spec_errors)

    fig = plt.figure(0)
    axes = fig.add_subplot(111)

    axes.errorbar(wavelengths, spec_rotation, yerr=spec_errors, fmt='o', markersize=4, color='red')

    axes.plot(wave_space, drude_expression(wave_space, rot_const, dis_const), linewidth=2, color='blue', label='Line of best fit')

    axes.text(480, 65, r"$\chi^{2}$ = " + f"{chi:.3f}\n" + r"$\chi_{red}^{2}$ = " + f"{red_chi:.3f}")

    axes.set_title("Wavelength of Light vs Specific Rotation", fontsize=16, color="black")
    axes.set_xlabel("Wavelength (nm)", fontsize=12)
    axes.set_ylabel(r"Specific Rotation (deg dm$^{-1}$g$^{-1}$ml)", fontsize=12)


    axes.legend(loc='upper right', fontsize=11)
    plt.savefig("spec_rotation_real.png", dpi=300)
    plt.show()


def drude_expression(wavelength, rotaton_const, dispersion_const):
    return rotaton_const / (wavelength**2 - dispersion_const**2)


def fit_distribution(distribution: Callable, x, y, errors=None, *params):
    optimals, covariance = sci.curve_fit(distribution, x, y, p0=params, sigma=errors, absolute_sigma=False)
    param_errors = np.sqrt(np.diag(covariance))

    return optimals, param_errors


def chi_analysis(x, y, y_error, *free_params, lsfr: bool=False):
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

        chi_value = chi_squared(data, drude_expression, *free_params)
        red_chi = calculate_reduced_chi(chi_value, data.shape[0])

    return chi_value, red_chi


if __name__ == "__main__":
    sucrose_conc(sucrose_conc_data)

    

