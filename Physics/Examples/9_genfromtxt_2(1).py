# -*- coding: utf-8 -*-
"""
PHYS20161 Week 9 further example of genfromtxt

Reads in data and showcases more options in genfromtxt

Lloyd Cawthorne 28/11/19

"""

import numpy as np

data = np.genfromtxt('data_2.csv', comments='%', delimiter=',',
                     skip_header=1)
print(data)
