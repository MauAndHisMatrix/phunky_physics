# -*- coding: utf-8 -*-
"""
PHYS20161 Week 5 Examples of joining numpy arrays

Lloyd Cawthorne 20/10/19
"""
import numpy as np

FIRST_ARRAY = np.arange(5)

SECOND_ARRAY = np.arange(5, 10)

# Can create an array with append

THIRD_APPEND = np.append(FIRST_ARRAY, SECOND_ARRAY)

# Can join them by using concatenate to return
# this is better if we have more than two arrays we want to merge

THIRD_CONCATENATE = np.concatenate((FIRST_ARRAY, SECOND_ARRAY))

# Both methods above would need to be resized if we wanted to
# have a 2D array of the form [a, b]

THIRD_APPEND.resize(2, 5)

# Can 'stack' the arrays instead

# On top of one another
THIRD_VSTACK = np.vstack((FIRST_ARRAY, SECOND_ARRAY))

# Next to each other
THIRD_HSTACK = np.hstack((FIRST_ARRAY, SECOND_ARRAY))

# Along a third axis; [[a[0],b[0]], [a[1],b[1]],...]
THIRD_DSTACK = np.dstack((FIRST_ARRAY, SECOND_ARRAY))[0]

# Note: dstack alone returns an extra dimension:
# [[[a[0],b[0]], [a[1],b[1]],...]] <-- Extra brackets
# So, returning the first element gives something more useful.
