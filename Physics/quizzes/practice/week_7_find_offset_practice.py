# -*- coding: utf-8 -*-
"""
PHYS20161 Week 7 quiz: find offset PRACTICE

This code reads in data and finds the offset according to the polynomial
defined.

It does this by minimising the variance between the data and the model given
some initial parameters.

Lloyd Cawthorne 01/11/19

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Define fitting constants
STEP = 1.
TOLERANCE = 1000.
STARTING_OFFSET = 0.


def polynomial(x_variable, offset_variable):
    """
    Returns x_variable^5 - 10x_variable^3 + 15x_variable
                                          + offset_variable (float)

    x_variable (float)
    offset_variable (float)
    """
    return (x_variable**5 - 10 * x_variable**3 + 15 * x_variable
            + offset_variable)


def is_float(number):
    """Checks if input is valid

    Returns True if number is float.
    """

    try:
        float(number)
        return True

    except ValueError:
        return False


# Read and validate data

input_file = open('.\week_7_find_offset_data_practice.csv', 'r')

data = np.zeros((0, 2))

for line in input_file:

    split_up = line.split(',')

    if is_float(split_up[0]) and is_float(split_up[1]):
        temp = np.array([float(split_up[0]), float(split_up[1])])
        data = np.vstack((data, temp))

input_file.close()

# Fitting procedure
# offset, difference, and comparison_0,1,2 are all varied when running so we
# will keep write them in snake_case to distinguish from constants.

offset = STARTING_OFFSET
difference = np.sum((data[:, 1] - polynomial(data[:, 0], offset))**2)
# print(difference)
# sys.exit(0)

while difference > TOLERANCE:

    comparison_0 = np.sum((data[:, 1] - polynomial(data[:, 0], offset))**2)

    comparison_1 = np.sum((data[:, 1] - polynomial(data[:, 0], offset + STEP))**2)

    comparison_2 = np.sum((data[:, 1] - polynomial(data[:, 0], offset - STEP))**2)

    if comparison_0 > comparison_1:
        offset += STEP
        difference = comparison_0 - comparison_1

    elif comparison_0 > comparison_2:
        offset -= STEP
        difference = comparison_0 - comparison_2
        
    else:
        print('Failed to reach desired precision.')
        break

print('The fitted offset is {:.1f}.'.format(offset))

# Plot results

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Fitted function against data')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.scatter(data[:, 0], data[:, 1])
ax.plot(data[:, 0], polynomial(data[:, 0], offset), c='black')
plt.show()
