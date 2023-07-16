# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8 chi square analysis example

Read in data and fit coefficient of polynomial and
 find errors based on chi square using brute force.

Lloyd Cawthorne 12/11/19
"""

import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'polynomial_data_1.csv'

# Function definitions


def is_number(number):
    """
    Checks if number is float. Returns bool
    number (float)
    """
    try:
        float(number)
        return True
    except ValueError:
        return False


def function(x_variable, coefficient_variable):
    """
    Returns x^3 + coefficient * x + 7

    x_variable (float)
    coefficient_variable (float)
    """

    return x_variable**3 + coefficient_variable * x_variable + 7


def chi_squared(coefficient_variable, x_values, data, uncertainties):
    """ Returns chi squared after comparing function and data for a given
    coefficient

     coefficient_variable (float)
     x_values array of floats
     function_data array of floats
     errors array of floats
     """

    prediction = function(x_values, coefficient_variable)

    return np.sum(((prediction - data) / uncertainties)**2)


# Read in data

x_data = np.array([])
function_data = np.array([])
uncertainty_data = np.array([])

input_file = open(FILE_NAME, 'r')

for line in input_file:

    split_up = line.split(',')

    valid = []

    for entry in split_up:
        valid.append(is_number(entry))

    if all(valid):
        x_data = np.append(x_data, float(split_up[0]))
        function_data = np.append(function_data, float(split_up[1]))
        uncertainty_data = np.append(uncertainty_data, float(split_up[2]))

# Plot raw data

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Raw data', fontsize=14)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('f(x)', fontsize=14)

ax.tick_params(labelsize=14)

ax.errorbar(x_data, function_data, yerr=uncertainty_data, fmt='o')
plt.show()

# Find best value for coefficient by trying everything

# Using Brute Force so testing all values.
# Hard to decide on range to search across using this approach.
coefficient_values = np.linspace(-54, -41, 100000)

chi_squares = np.array([])

for coefficient in coefficient_values:

    chi_squares = np.append(chi_squares,
                            chi_squared(coefficient, x_data, function_data,
                                        uncertainty_data))

# Plot chi^2
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Chi^2 against coeffienct values', fontsize=14)
ax.set_xlabel('Coefficient values', fontsize=14)
ax.set_ylabel('Chi^2', fontsize=14)

ax.tick_params(labelsize=14)

ax.plot(coefficient_values, chi_squares)

# Select best value
fitted_coefficient = coefficient_values[np.argmin(chi_squares)]
minimum_chi_squared = np.min(chi_squares)

ax.scatter(fitted_coefficient, minimum_chi_squared, s=100, label='minimum',
           c='k')
ax.legend(fontsize=14)
plt.show()

# Visually compare result with data

fig = plt.figure()
ax = fig.add_subplot(111)


ax.set_title('Data against fit', fontsize=14)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('f(x)', fontsize=14)

ax.tick_params(labelsize=14)
x_plotting_values = np.linspace(-10, 10, 100)

ax.errorbar(x_data, function_data, yerr=uncertainty_data, fmt='o',
            label='Data')
ax.plot(x_plotting_values, function(x_plotting_values, fitted_coefficient),
        label='Fit')
ax.legend(fontsize=14)
plt.show()

# Visually show 1 & 2 sigma values

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Chi^2 against coeffienct values', fontsize=14)
ax.set_xlabel('Coefficient values', fontsize=14)
ax.set_ylabel('Chi^2', fontsize=14)

ax.tick_params(labelsize=14)

ax.plot(coefficient_values, chi_squares)

# fitted_coefficient = coefficient_values[np.argmin(chi_squares)]
# minimum_chi_squared = chi_squares[np.argmin(chi_squares)]

ax.scatter(fitted_coefficient, minimum_chi_squared, s=100, label='minimum',
           c='k')

# 2 sigma line

ax.plot(coefficient_values, np.full(len(coefficient_values),
                                    minimum_chi_squared + 3.841), c='grey',
        dashes=[4, 2], label=r'2 \sigma')

ax.scatter(coefficient_values[np.argmin(
    np.abs(chi_squares - minimum_chi_squared - 3.84))],
           minimum_chi_squared + 3.84, c='b', s=100)
ax.scatter(coefficient_values[np.argmin(
    np.abs(chi_squares - minimum_chi_squared - 3.835))],
           minimum_chi_squared + 3.84, c='b', s=100)


# 1 sigma line
ax.plot(coefficient_values, np.full(len(coefficient_values),
                                    minimum_chi_squared + 1), c='grey',
        dashes=[1, 1], label=r'1 \sigma')

sigma_index = np.argmin(np.abs(chi_squares - minimum_chi_squared - 1))

sigma = np.abs(coefficient_values[sigma_index] - fitted_coefficient)

ax.scatter(coefficient_values[np.argmin(
    np.abs(chi_squares - minimum_chi_squared - 1))],
           minimum_chi_squared + 1, c='b', s=100)

# Small offset, 0.005, to get point other side
ax.scatter(coefficient_values[np.argmin(
    np.abs(chi_squares - minimum_chi_squared - 0.995))],
           minimum_chi_squared + 1, c='b', s=100)


ax.legend(fontsize=14, loc='upper right')
plt.show()

print('We find C = {0:4.2f} +/- {1:4.2f} with a reduced chi square of'
      ' {2:3.2f}.'.format(fitted_coefficient, sigma,
                          minimum_chi_squared / (len(x_data) - 1)))
