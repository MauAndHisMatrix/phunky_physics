# -*- coding: utf-8 -*-
"""
PHYS20161 Week 7 Read & Validate data Practice

Write code to read in and validate the data. The validation should exclude any
row with non-numerical entries or values for y less than 0 or greater than 2.
A plotting routine has been provided to assist you.

Find the average of the x values, <x>, and the average of the y
values.

Lloyd Cawthorne 19/09/21

"""

import numpy as np
import matplotlib.pyplot as plt

def create_plot(data):
    """
    Creates a plot of the data

    Parameters
    ----------
    data : numpy array, [float, float]
        Data should be in order independent variable, dependent variable

    Returns
    -------
    None.

    """

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_title('Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y(x)')
    ax.scatter(data[:, 0], data[:, 1], s=1)
    plt.show()
    return None

# Main code
def is_float(number):
    try:
        float(number)
        return True

    except ValueError:
        return False

def y_bounds(entry):
    return 0. < entry < 2.


input_file = open('.\week7_unvalidated_data_practice.csv', 'r')

data = np.zeros((0, 2))

for line in input_file:

    split_up = line.split(',')

    if is_float(split_up[0]) and is_float(split_up[1]):
        x = float(split_up[0])
        y = float(split_up[1])
        if y_bounds(y):
            temp = np.array([x, y])
            data = np.vstack((data, temp))

input_file.close()

avg_x = np.mean(data[:, 0])
avg_y = np.mean(data[:, 1])

print(f"x: {avg_x:.3f}\ny: {avg_y:.3f}")

# create_plot(data)