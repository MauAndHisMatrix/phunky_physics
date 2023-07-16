# -*- coding: utf-8 -*-
"""
PHYS20161 Week 11 example of genfromtxt

Reads in data file and demonstrates basic use of genfromtxt

Lloyd Cawthorne 28/11/19

"""

import numpy as np

# By default reads data as floats

data_float = np.genfromtxt('data_1.txt', delimiter=',')

print(data_float, '\n\n')

# Specify data to be integers

data_integer = np.genfromtxt('data_1.txt', delimiter=',', dtype='i4')

print(data_integer)
