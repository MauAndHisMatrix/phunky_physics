# -*- coding: utf-8 -*-
"""
PHYS20161 Week 7 Read csv file example

Example of reading in array from csv file

Lloyd Cawthorne 14/20/20

"""

import numpy as np

data = np.empty((0, 2))

input_file = open('xy_data_example.csv', 'r')

for line in input_file:
    if line[0] != '%':
        split_up = line.split(',')

        temp = np.array([float(split_up[0])])
        temp = np.append(temp, float(split_up[1]))

        data = np.vstack((data, temp))

input_file.close()
