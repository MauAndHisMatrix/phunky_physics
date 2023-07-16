# -*- coding: utf-8 -*-
"""
PHYS20161 Week 7 read data example

Reads in a file containing sin and cos for angles from 0 to 90 degrees and
finds the tangent

Lloyd Cawthorne 31/08/21

"""
import numpy as np

tans= np.array([])

input_file = open('trig_data.txt', 'r')

for line in input_file:

    if line[0] != '%':

        split_up = line.split()

        if float(split_up[1]) != 0:

            tans = np.append(tans, float(split_up[0]) / float(split_up[1]))


input_file.close()

