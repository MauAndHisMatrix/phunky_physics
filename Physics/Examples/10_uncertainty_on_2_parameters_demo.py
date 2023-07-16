# -*- coding: utf-8 -*-
"""
PHYS20161 Week 11 example of fitting two parameters and finding their
uncertainties

Reads in data, fits two parameters, a & b, for the polynomial:
    f(x) = ax^3+bx^2+x-4.5

Does the fit using a 2D hill-climbing approach with varying step size.
Uses np.random to guess starting points.
Exits code if fit is unsuccesful, IPython prints messy warning as a result.

Contour plots the chi^2 to illustrate location of uncertainties and dependence

Lloyd Cawthorne 28/11/19

"""

import sys
import numpy as np
import matplotlib.pyplot as plt

STEP_START = 1
TOLERANCE = 0.001

# If the routine is unable to find a better value after MAX_FAILED_ATTEMPTS
# number of continuous reductions to the step size then exit the routine.
MAX_FAILED_ATTEMPTS = 5


def function(x_variable, a_coefficient, b_coefficient):
    """
    Returns
    ax^3+bx^2+x-4.5
    x_variable (float)
    a_coefficient (float)
    b_coefficient (float)
    """
    return (a_coefficient * x_variable**3
            + b_coefficient * x_variable**2 + x_variable - 4.5)


def mesh_arrays(x_array, y_array):
    """Returns two meshed arrays of size len(x_array)
    by len(y_array)
    x_array array[floats]
    y_array array[floats]
    """
    x_array_mesh = np.empty((0, len(x_array)))

    for dummy in range(len(y_array)):
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for dummy in range(len(x_array)):
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh


def chi_squared(a_parameter, b_parameter, input_data):
    """Returns chi squared for a pre defined function depenedent on one
    variable, x, with two parameters, a & b.
    data is a 2D array composed of rows of [x values, f(x) values and
    uncertainties]

    inputdata array([float, float, float])
    a_parameter (float)
    b_parameter (float)
    """
    chi_square = 0
    for entry in input_data:
        chi_square += (((function(entry[0], a_parameter, b_parameter)
                         - entry[1]) / entry[2])**2)

    return chi_square


def hill_climbing(input_data, step=STEP_START):
    """
    Runs Hill Climbing algorithm in 2D with a varying step size.
    Args:
        input_data array([float, float, float])
        step kwarg (float)
    Returns:
        np.array([minimum_chi_squared (float),
                  [a_fit, b_fit] ([float, float]),
                  counter (int),
                  success (bool)])
    """
    # Guess starting values
    a_fit = np.random.randint(10)
    b_fit = np.random.randint(10)
    minimum_chi_squared = chi_squared(a_fit, b_fit, input_data)
    difference = 1
    counter = 0

    # Count how many times we fail to find a better value after drecreasing the
    # step size. If it matches MAX_FAILED_ATTEMPTS, then exit loop and
    # set success = 0.
    timeout = 0
    success = 1

    # Look around current best fit to find better value

    while difference > TOLERANCE:
        counter += 1

        # Save current best values for comparison later
        a_test = a_fit
        b_test = b_fit

        for i in np.arange(-1, 2, 1):
            for j in np.arange(-1, 2, 1):

                test_chi_squared = chi_squared(a_fit + i * step, b_fit
                                               + j * step, input_data)

                if test_chi_squared < minimum_chi_squared:

                    timeout = 0

                    # If better solution found update parameters:

                    difference = np.abs(minimum_chi_squared - test_chi_squared)
                    minimum_chi_squared = test_chi_squared

                    a_fit += i * step
                    b_fit += j * step

        if a_fit == a_test and b_fit == b_test:
            # If we have failed to find better values then reduce the step size
            step = step * 0.1
            timeout += 1
            if timeout == MAX_FAILED_ATTEMPTS:
                success = 0
                print('Failed to reach desired accuracy with a step size of'
                      ' {:g}'.format(step))
                break
    return [minimum_chi_squared, [a_fit, b_fit], counter, success]


# Get data
data = np.genfromtxt(r'C:\Users\Marawan\Desktop\Python\Physics\Examples\polynomial_data.csv', delimiter=',')

# Plot raw data
raw_data_figure = plt.figure()
raw_data_plot = raw_data_figure.add_subplot(111)

raw_data_plot.set_title('Raw data')
raw_data_plot.set_xlabel('x')
raw_data_plot.set_ylabel('f(x)')


raw_data_plot.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='o')
plt.show()

# Find values for a and b using 2D Hill-Climbing

fit = hill_climbing(data)

if fit[-1] == 0:
    print('Terminating code.')
    sys.exit()

fitted_minimum_chi_squared = fit[0]
fitted_parameters = fit[1]

fitted_data_figure = plt.figure()
fitted_data_plot = fitted_data_figure.add_subplot(111)

fitted_data_plot.set_title(r'fitted data. a = {0:3.2f}, b = {1:3.2f}.'.
                           format(fitted_parameters[0], fitted_parameters[1])
                           + r' $\chi^2_{{\mathrm{{red.}}}} = $ {0:3.2f}'.
                           format(fitted_minimum_chi_squared / (len(data)
                                                                - 2)))
fitted_data_plot.set_xlabel('x')
fitted_data_plot.set_ylabel('f(x)')

fitted_data_plot.plot(data[:, 0], function(data[:, 0], fitted_parameters[0],
                                           fitted_parameters[1]))
fitted_data_plot.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='o')
plt.show()

# Contour plot of parameters
# Note: these are hardcoded, not best practice.
a_values = np.linspace(fitted_parameters[0] - 0.05,
                       fitted_parameters[0] + 0.05, 500)
b_values = np.linspace(fitted_parameters[1] - 0.3,
                       fitted_parameters[1] + 0.3, 500)

a_mesh, b_mesh = mesh_arrays(a_values, b_values)

parameters_contour_figure = plt.figure()

parameters_contour_plot = parameters_contour_figure.add_subplot(111)

parameters_contour_plot.set_title(r'$\chi^2$ contours against parameters.',
                                  fontsize=14)
parameters_contour_plot.set_xlabel('a', fontsize=14)
parameters_contour_plot.set_ylabel('b', fontsize=14)


# Place minimum as single point
parameters_contour_plot.scatter(fitted_parameters[0], fitted_parameters[1],
                                label='Minimum')


# chi^2 min + 1 contour, treated separately as we want it dashed.
parameters_contour_plot.contour(a_mesh, b_mesh,
                                chi_squared(a_mesh, b_mesh, data),
                                levels=[fitted_minimum_chi_squared + 1.00],
                                linestyles='dashed',
                                colors='k')

# Contours to be plotted
# Ideally these numbers would be defined elsewhere so they are easy to ammend
# without having to ammend several things.
chi_squared_levels = (fitted_minimum_chi_squared + 2.30,
                      fitted_minimum_chi_squared + 5.99,
                      fitted_minimum_chi_squared + 9.21)

contour_plot = parameters_contour_plot.contour(a_mesh, b_mesh,
                                               chi_squared(a_mesh,
                                                           b_mesh, data),
                                               levels=chi_squared_levels)
labels = ['Minimum', r'$\chi^2_{{\mathrm{{min.}}}}+1.00$',
          r'$\chi^2_{{\mathrm{{min.}}}}+2.30$',
          r'$\chi^2_{{\mathrm{{min.}}}}+5.99$',
          r'$\chi^2_{{\mathrm{{min.}}}}+9.21$']

parameters_contour_plot.clabel(contour_plot)

# Want plot legend outside of plot area, need to adjust size of plot so that
# it is visible

box = parameters_contour_plot.get_position()
parameters_contour_plot.set_position([box.x0, box.y0, box.width * 0.7,
                                      box.height])

# Add custom plot labels
for index, label in enumerate(labels):
    parameters_contour_plot.collections[index].set_label(label)
parameters_contour_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                               fontsize=14)

plt.savefig('contour_plot_3.png', dpi=300)

plt.show()
