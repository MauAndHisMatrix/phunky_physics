"""
TITLE: PHYS20161 Assignment 2: Nuclear Decay

This script takes in experimental data collected from two detectors
operating in alternating 10 minute intervals. They measured the activity
levels of the radioactive isotope 79_Rb over a certain period of time.

The script uses this data to calculate the following:
    a) The decay constants of 79_Rb and 79_Sr.
    b) The half lives of 79_Rb and 79_Sr.
    c) The reduced chi-squared value when the data is fitted to the equation
        for 79_Rb's activity.

A plot of the data containing a line of best fit is also produced.

Last updated: 14/12/22
Author: w77334ma
"""

# Standard Library
import sys
from typing import Union, Callable
from pathlib import Path

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sympy as sym
from scipy.optimize import fmin

# Constants and filepaths
SAMPLE_79_SR = 1e-6 # mol
AVOGADRO_CONST = 6.02214076e23 # mol^-1

DATA_FILES = [
    "Nuclear_data_1.csv",
    "Nuclear_data_2.csv"
]


def round_to_sig(number: float, sig_fig: int=3) -> float:
    """
    This function rounds the inputted number to the required number of
    significant figures.

    Parameters:
        number
        sig_fig: The desired no. of sig. figures. 3 is the default.

    Returns:
        The rounded number
    """
    one_sig_fig = - int(np.floor(np.log10(abs(number))))

    return round(number, one_sig_fig - 1 + sig_fig)


def query_yes_no(query: str) -> bool:
    """
    This function takes any string query and outputs it to the terminal
    as a 'yes or no' question, giving the user the opportunity to decide
    for or against the query.

    Parameters:
        query: The question to be posited to the user.

    Returns:
        A boolean indicating whether or not the query has been accepted.
    """
    return input(query + " [y/n]: ").lower() in ["y", "yes"]


def enter_filenames() -> list[str]:
    """
    This function takes in dataset filenames from the terminal if the user
    has specified that this is the preferred option, as opposed to setting
    a global variable.

    Parameters:
        None

    Returns:
        data_filenames
    """
    no_of_datasets = int(input("How many datasets are you importing? "))

    data_filenames = [str(input(f"Enter dataset {i + 1} filename: "))
                      for i in range(no_of_datasets)]

    return data_filenames


def import_data(data_filenames: Union[list[str], list[Path]]) -> np.ndarray:
    """
    This function imports all the datasets listed in the data_filenames
    argument. Once they've been validated, they are then concatenated
    together. Only datasets located in the same folder as the script that
    calls this function can be found.

    Parameters:
        data_filenames: List containing the filenames of all the datasets.

    Returns:
        sorted_import: A 2D numpy array containing all the datasets, linked
                        together as one.
    """
    if not data_filenames:
        print("No file paths given. Terminating script.")
        sys.exit(0)

    imported_data = np.empty((0, 3))
    for i, dataset in enumerate(data_filenames):
        try:
            temp_dataset = np.genfromtxt(dataset, delimiter=',',
                                         skip_header=1)
            imported_data = np.vstack((imported_data, temp_dataset))
        except ValueError:
            print(f"Error found in dataset {i + 1}. Potential data loss."
                  f"\nRows must have three columns: "\
                    "Time(hrs) , Activity (TBq) , Activity error (TBq)"\
                    "\nTerminating script.")
            sys.exit(0)
        except FileNotFoundError:
            print(f"Dataset {i + 1} was not found at the given file path.")
            if query_yes_no("Would you like to continue without it?"):
                continue
            else:
                sys.exit(0)

    sorted_import = imported_data[imported_data[:, 0].argsort()]

    return sorted_import


def clean_data(data: np.ndarray) -> np.ndarray:
    """
    This function takes in the 2D data array and removes any negatives
    and NaNs (Not-a-Number).

    Parameters:
        data

    Returns:
        cleaned_data
    """
    nan_filter = np.unique(np.where(np.isnan(data))[0])
    negative_zero_filter = np.unique(np.where(data <= 0)[0])
    combined_filter = np.hstack((nan_filter, negative_zero_filter))

    cleaned_data = np.delete(data, combined_filter, axis=0)

    return cleaned_data


def convert_units(data: np.ndarray,
                  x_multiplier: float,
                  y_and_y_unc_multiplier: float) -> np.ndarray:
    """
    This function allows the user to convert the units in a 2D data array
    containing the standard x, y, y_error columns. The units are converted
    by the inputted multipliers.

    Parameters:
        data
        x_multiplier
        y_and_y_unc_multiplier

    Returns:
        data
    """
    data[:, 0] *= x_multiplier
    data[:, [1, 2]] *= y_and_y_unc_multiplier

    return data


def remove_outliers(data: np.ndarray,
                    fitting_function: Callable,
                    initial_free_params: list[float]) -> np.ndarray:
    """
    This function removes outliers in the data by comparing the y value
    of a data point to an initial line of best fit value at the same x value.
    If the difference between them is more than the multiplier times the error
    bar on the data point, it is judged to be an outlier, and is then removed.
    This is because 0.3% of data points are expected to be outside three error
    bars of the line of best fit, and 0.3% of data points is usually less than
    one. As soon as one or more data points are expected to be outside three
    error bars, the multiplier becomes four, which works for any
    reasonably large dataset containing roughly N >= 333 data points.

    Parameters:
        data
        fitting_function
        initial_free_params: The initial free parameters on the fitting
                            function calculated using a chi-squared
                            minimisation of the function.

    Returns:
        outliers_removed: The data with the outliers removed.
    """
    initial_func = fitting_function(data[:, 0],
                                    initial_free_params[0],
                                    initial_free_params[1])

    multiplier = 3. if (data.shape[0] * 0.003 < 1) else 4.
    stddev_filter = np.unique(np.where(np.abs(
                initial_func - data[:, 1]) > multiplier * 2.*data[:, 2])[0])

    outliers_removed = np.delete(data, stddev_filter, axis=0)

    return outliers_removed


def activity_79_rb_function(time: Union[float, np.ndarray],
                        decay_const_79_rb: float,
                        decay_const_79_sr: float,
                        sym_form: bool=False) -> Union[float, np.ndarray]:
    """
    This function calculates the activity of 79_Rb based on time and the decay
    constants of 79_Rb and 79_Sr.

    Parameters:
        time
        decay_const_79_rb
        decay_const_79_sr
        sym_form: This boolean allows the user to specify whether or not the
                activity is calculated in symbolic form, which is needed for
                error calculations.

    Returns:
        Activity of 79_Rb
    """
    initial_79_sr_number = SAMPLE_79_SR * AVOGADRO_CONST
    fraction_of_decays = decay_const_79_sr / \
                (decay_const_79_rb - decay_const_79_sr)
    if sym_form:
        exponentials = sym.exp(- decay_const_79_sr * time) - \
                        sym.exp(- decay_const_79_rb * time)
    else:
        exponentials = np.exp(- decay_const_79_sr * time) - \
                        np.exp(- decay_const_79_rb * time)

    number_of_79_rb_nuclei = \
        initial_79_sr_number * fraction_of_decays * exponentials

    return decay_const_79_rb * number_of_79_rb_nuclei


def calculate_half_life(decay_constant: float) -> float:
    """
    This function calculates the half-life of a radioactive isotope given
    its decay constant.

    Parameters:
        decay_constant

    Returns:
        half life
    """
    return np.log(2) / decay_constant


def calculate_decay_const_from_half_life(half_life: float) -> float:
    """
    This function calculates the decay constant of a radioactive isotope given
    its half life.

    Parameters:
        half life

    Returns:
        decay constant
    """
    return np.log(2) / half_life


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


def calculate_half_lives_and_errors(decay_constants: list,
                               decay_const_errors: list) -> tuple[np.ndarray]:
    """
    This function calculates the half lives and half life errors of 79_Rb
    and 79_Sr using their decay constants and the decay constant errors.

    Parameters:
        decay_constants
        decay_const_errors

    Returns:
        A tuple containing two 1D arrays that contain the half life data
        for 79_Rb and 79_Sr respectively.
    """
    rb_half_life = calculate_half_life(decay_constants[0])
    rb_half_life_error = calculate_variable_error(calculate_half_life,
                                                  [("lamda_rb",
                                                    decay_constants[0],
                                                    decay_const_errors[0])])
    sr_half_life = calculate_half_life(decay_constants[1])
    sr_half_life_error = calculate_variable_error(calculate_half_life,
                                                  [("lamda_sr",
                                                    decay_constants[1],
                                                    decay_const_errors[1])])

    return np.array([rb_half_life, rb_half_life_error]), \
            np.array([sr_half_life, sr_half_life_error])


def calculate_activity_error(time: float,
                             decay_constants: list,
                             decay_const_errors: list) -> float:
    """
    This function is a wrapper of calculate_variable_error. It calculates the
    error on an activity value, which has been calculated using the
    activity_79_rb_function. The sym_form keyword argument is included to
    inform the error function that the activity equation must be processed
    using Sympy's exponential implementations in order for the differentiation
    to work.

    Parameters:
        time
        decay_constants
        decay_const_errors

    Returns:
        activity_error
    """
    contributing_variables = [("t", time, 0),
                              ("lamda_rb", decay_constants[0],
                                decay_const_errors[0]),
                              ("lamda_sr", decay_constants[1],
                                decay_const_errors[1])]
    activity_error = calculate_variable_error(activity_79_rb_function,
                                              contributing_variables,
                                              sym_form=True)

    return activity_error


def calculate_chi_squared(data: np.ndarray,
                          fitting_function: Callable,
                          *free_params) -> float:
    """
    This function calculates the chi-squared value for a function being
    fitted to some data, using the specified free parameters. A for loop
    has been used instead of a numpy summation because the latter method
    leads to complications during contour plotting.

    Parameters:
        data
        fitting_function
        free_params: The free parameters present in the fitting function.

    Returns:
        chi_sq_value
    """
    chi_sq_value = 0.
    for data_point in data:
        prediction = fitting_function(data_point[0], *free_params)
        chi_sq_value += ((prediction - data_point[1]) / data_point[2])**2

    return chi_sq_value


def reduce_chi_squared(chi_squared: float,
                       sample_size: int,
                       no_of_free_params: int=1) -> float:
    """
    This function calculates the reduced chi-squared value based on the
    chi-squared value, sample size, and the number of free parameters.

    Parameters:
        chi_squared
        sample_size
        no_of_free_params

    Returns:
        reduced chi-squared
    """
    deg_of_freedom = sample_size - no_of_free_params

    return chi_squared / deg_of_freedom


def minimise_chi_squared(data: np.ndarray,
                         fitting_function: Callable,
                         initial_params: list[float]) -> tuple:
    """
    This function is a wrapper of scipy.optimize.fmin. It takes in a dataset
    and a fitting function, then calculates the optimum free parameters by
    varying them from starting values until the minimum chi-squared value is
    found.

    Parameters:
        data
        fitting_function

    Returns:
        A tuple containing the optimum free parameters and the minimised
        chi-squared result.
    """
    results = \
        fmin(lambda free_params: calculate_chi_squared(data,
                                            fitting_function,
                                            free_params[0], free_params[1]),
                                    initial_params,
                                    full_output=True, disp=0)
    return results[0], results[1]


def calculate_param_errors(chi_1_coods: np.ndarray,
                           no_of_free_params: int=2) -> list:
    """
    This function calculates the errors on the free parameters. It takes in a
    2D array of the coordinates of the chi-squared + 1 contour, which
    represent combinations of free parameters within 1 s.d. of the optimum
    values. The columns are in the propagated order of the free parameters.
    The minimum and maximum values for each column are found, which represent
    free parameter - sigma, and free parameter + sigma, respectively. The
    difference between them is found, then divided by two, which yields sigma.

    Parameters:
        chi_1_coods: The chi-squared + 1 contour coordinates. Each column
                    represents a free parameter.
        no_of_free_params: The number of free parameters.

    Returns:
        errors
    """
    errors = [(np.max(chi_1_coods[:, i]) - np.min(chi_1_coods[:, i])) / 2
              for i in range(no_of_free_params)]

    return errors


def plot_data(data: np.ndarray,
              decay_constants: float,
              xmas_decorations: bool=False) -> None:
    """
    This function plots a scatter graph of the data, including a line of best
    fit. If requested, Christmas decorations can be added using the
    xmas_decorations flag.

    Parameters:
        data
        decay_constants
        xmas_decorations: Boolean that gives the user the option to add
                          Christmas decorations to the plot.

    Returns:
        None
    """
    fig = plt.figure(0)
    axes = fig.add_subplot(111)

    best_fit_y_values = activity_79_rb_function(data[:, 0],
                                                decay_constants[0],
                                                decay_constants[1])

    colour_space = ["black", "blue"]
    if xmas_decorations:
        max_activity = max(best_fit_y_values)
        max_activity_index = np.where(best_fit_y_values == max_activity)
        max_time = data[:, 0][max_activity_index]
        image = plt.imread("santa_hat.png")
        im_box = OffsetImage(image, zoom=0.03, alpha=0.75)
        annotation_box = AnnotationBbox(im_box, (max_time, max_activity),
                                        frameon=False)
        axes.add_artist(annotation_box)
        colour_space = ["green", "red"]

    axes.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='o',
                  markersize=4, color=colour_space[0],
                  label="Experimental data")

    axes.plot(data[:, 0], best_fit_y_values, color=colour_space[1],
              linewidth=2, label="Line of best fit")

    axes.set_title("Time elapsed vs Activity levels", fontsize=16,
                                                      color=colour_space[0])
    axes.set_xlabel("Time elapsed (s)", fontsize=12)
    axes.set_ylabel(r"Activity levels of $^{79}$Rb (Bq)", fontsize=12)

    axes.legend()
    plt.savefig("time_vs_activity_levels.png", dpi=300)
    plt.show()


def plot_contours(data: np.ndarray,
                  fitting_function: Callable,
                  free_params: list,
                  min_chi_squared: float,
                  free_param_names: list[str],
                  x_lim: tuple=None) -> np.ndarray:
    """
    This function plots a contour graph that depicts the minimisation process
    of the chi-squared value, with each axis representing a range of values
    that each free parqameter could take. The coordinates of the
    chi-squared + 1 contour are returned for use in calculating the errors on
    the free parameters.

    Parameters:
        data
        fitting_function
        free_params
        min_chi_squared
        free_param_names: List of the free parameter names used in the plot
                          annotations.
        x_lim: Allows user to set manual x limits on the contour plot.
                Form: (min, max)

    Returns:
        chi_plus_1_contour_coordinates
    """
    param_1_range = \
        np.linspace(0.93 * free_params[0], 1.07 * free_params[0], num=100)
    param_2_range = \
        np.linspace(0.93 * free_params[1], 1.07 * free_params[1], num=100)

    param_1_mesh, param_2_mesh = np.meshgrid(param_1_range, param_2_range)

    # The first list contains the specific chi squared band increments
    # for the standard deviation multiples.
    std_deviation_bands = np.array([1, 2.3, 5.99] + [12.07, 21.84]) \
                          + min_chi_squared

    fig = plt.figure(1)
    axes = fig.add_subplot(111)

    wider_band_contour = axes.contour(param_1_mesh, param_2_mesh,
                                      calculate_chi_squared(data,
                                                  fitting_function,
                                                  param_1_mesh, param_2_mesh),
                                      levels=std_deviation_bands[1:])
    chi_plus_1_contour = axes.contour(param_1_mesh, param_2_mesh,
                                      calculate_chi_squared(data,
                                                  fitting_function,
                                                  param_1_mesh, param_2_mesh),
                                      levels=[std_deviation_bands[0]],
                                      linestyles="dashed")

    axes.scatter(free_params[0], free_params[1], color='r',
               label=r"$\chi^{2}_{min.}$ = " + f"{min_chi_squared:.3f}")

    axes.clabel(wider_band_contour, inline=1, fontsize=10)
    axes.clabel(chi_plus_1_contour, inline=1, fontsize=10,
              fmt=r"$\chi^{2}_{min.}+ 1$")

    # x limits don't change if x_lim = None
    axes.set_xlim(x_lim)

    axes.ticklabel_format(scilimits=(-3, 4))

    axes.set_xlabel(f'Free parameter {free_param_names[0]}')
    axes.set_ylabel(f'Free parameter {free_param_names[1]}')
    axes.set_title(r'Contour plot that varies both the' \
        f' free parameters of\n {free_param_names[0]} and '\
        f'{free_param_names[1]} to minimise the chi-squared value')
    axes.legend(loc='upper left')

    plt.savefig("contour_minimising_chi_squared.png", dpi=300)
    plt.show()

    chi_plus_1_contour_coordinates = \
        chi_plus_1_contour.collections[0].get_paths()[0].vertices

    return chi_plus_1_contour_coordinates


def graph_eq(lamda):
    return lamda * np.exp(- lamda * )


def find_initial_params(data):
    N_0 = AVOGADRO_CONST * SAMPLE_79_SR
    equation_1 = data[:, 1] / N_0

    equation_2 = 


def main():
    """
    Main function that executes all the calculations for the specific task
    of analysing the decay of Rb 79.
    """
    if query_yes_no("Would you like to enter the data filenames?"):
        data = import_data(enter_filenames())
    else:
        data = import_data(DATA_FILES)
    data = clean_data(data)
    data = convert_units(data, x_multiplier=3600, y_and_y_unc_multiplier=1e12)

    print(data[data[:, 1].argsort()][-5:, :])
    # sys.exit(0)

    inital_params = [0.0005, 0.005]
    minimised_initial_consts, _ = minimise_chi_squared(data,
                                                      activity_79_rb_function,
                                                      inital_params)
    print(f"{minimised_initial_consts=}")
    data = remove_outliers(data, activity_79_rb_function,
                           minimised_initial_consts)

    decay_constants, chi_squared_result = minimise_chi_squared(data,
                                                     activity_79_rb_function,
                                                     minimised_initial_consts)

    chi_plus_1_contour_coordinates = plot_contours(data,
                                                   activity_79_rb_function,
                                                   decay_constants,
                                                   chi_squared_result,
                                                [r'$^{79}$Rb', r'$^{79}$Sr'],
                                                x_lim=(4.9e-4, 5.3e-4))
    decay_const_errors = calculate_param_errors(
                                            chi_plus_1_contour_coordinates,
                                            len(decay_constants))

    rb_half_life_data, sr_half_life_data = calculate_half_lives_and_errors(
                                        decay_constants, decay_const_errors)
    rb_half_life_data_in_min = rb_half_life_data / 60
    sr_half_life_data_in_min = sr_half_life_data / 60

    reduced_chi = reduce_chi_squared(chi_squared_result, data.shape[0],
                                     no_of_free_params=2)

    xmas_bool = query_yes_no("Would you like Xmas decorations on the plot?")
    plot_data(data, decay_constants, xmas_decorations=xmas_bool)

    activity_at_t_90_min = activity_79_rb_function(90. * 60.,
                                                   decay_constants[0],
                                                   decay_constants[1])

    t_90_min_error = calculate_activity_error(90. * 60., decay_constants,
                                                       decay_const_errors)

    print(f"\n79_Rb decay constant: ({decay_constants[0]:.2e} +/- "\
                                    f"{decay_const_errors[0]:.0e}) s^-1")
    print(f"79_Sr decay constant: ({decay_constants[1]:.2e} +/- "\
                                    f"{decay_const_errors[1]:.1e}) s^-1")

    print(f"\n79_Rb half life: ({round_to_sig(rb_half_life_data_in_min[0])} "\
              f"+/- {round_to_sig(rb_half_life_data_in_min[1]):.1f}) minutes")
    print(f"79_Sr half life: ({round_to_sig(sr_half_life_data_in_min[0])} "\
              f"+/- {round_to_sig(sr_half_life_data_in_min[1]):.2f}) minutes")

    print(f"\nReduced chi_squared value: {reduced_chi:.2f}")

    print(f"\nActivity at t = 90 minutes: "\
                    f"{round_to_sig(activity_at_t_90_min * 1e-12)} +/- "\
                    f"{round_to_sig(t_90_min_error * 1e-12):.1f} TBq")


if __name__ == "__main__":
    main()
