# -*- coding: utf-8 -*-
"""
PHYS20161 Blackbaord quiz

Week 7: Reading & Writing files PRACTICE

Lloyd Cawthorne 29/10/19

This program reads in the file 'data_practice.txt' (it must be saved in the
same directory[folder] as the .py file) performs a calculation, then outputs
into seperate file.

Correct five bugs in the code below then write the correct operation to match
with the given outputs.

"""

import numpy as np

FILE_NAME = '.\week_7_read_and_write_data_practice.txt'

# Open data file to be read
input_file = open(FILE_NAME, 'r')

# Empty array to store data. Data is in rows of three so we set this up to take
# arrays of 3 values.
raw_data = np.empty((0, 3))


# Read data line by line and add each pair as an entry to raw_data
for line in input_file:
    entries = line.split(',')
    temp = np.array([])
    temp = np.append(temp, float(entries[0]))
    temp = np.append(temp, float(entries[1]))
    temp = np.append(temp, float(entries[2]))

    # Stack the arrays so we have an array of three-element arrays
    raw_data = np.vstack((raw_data, temp))

input_file.close()
# Aside: The above line is vital. No matter how well Python copes if you
# forget it, you will lose marks if it is absent.

# Empty array for resultant calculations
results = np.array([])

# Perform calculation for each row, you need to code here
results = np.sqrt(np.sum(raw_data**2, axis=1))

# Open output file to write
output_file = open('.\output.txt', 'w')

# Print to file
for x in results:
    print(f"{x:.2f}", file=output_file)

# Close output file
output_file.close()
