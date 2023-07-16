# -*- coding: utf-8 -*-
"""
PHYS20161 Week 5 Example adding a column.

Want an array with 5 rows and 3 columns with entries from 0 to 14.

Lloyd Cawthorne 20/10/19
"""
import numpy as np

FIRST_AND_SECOND_VALUES = np.arange(10)

[FIRST_VALUES, SECOND_VALUES] = np.hsplit(FIRST_AND_SECOND_VALUES, 2)

# Or FIRST_VALUES = FIRST_AND_SECOND_VALUES[:5]
#    SECOND_VALUES = FIRST_AND_SECOND_VALUES[5:]

FIRST_AND_SECOND_COLUMNS = np.dstack((FIRST_VALUES, SECOND_VALUES))[0]

# array([[0, 5],
#       [1, 6],
#       [2, 7],
#       [3, 8],
#       [4, 9]])

THIRD_VALUES = np.arange(10, 15)

# Reshape c so that it is 5 rows of 1 element

THIRD_COLUMN = THIRD_VALUES.reshape(5, 1)

# array([[10],
#       [11],
#       [12],
#       [13],
#       [14]])

# Now we can apply hstack

FINAL_ARRAY = np.hstack((FIRST_AND_SECOND_COLUMNS, THIRD_COLUMN))

# This leaves us with
# array([[ 0,  5, 10],
#       [ 1,  6, 11],
#       [ 2,  7, 12],
#       [ 3,  8, 13],
#       [ 4,  9, 14]])
