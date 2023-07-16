# -*- coding: utf-8 -*-
"""
PHYS20161 Week 7 writing file example 2

Writes the sin of angles 0 to 90 degree in degree steps.

Lloyd Cawthorne 14/10/20

"""

import numpy as np

angles = np.arange(91)

sins = np.sin(np.deg2rad(angles))

sin_file = open('sin_file.txt', 'w')

for index, angle in enumerate(angles):


    print("sin({0:2.0f}) = {1:4.3f}".
          format(angle, sins[index]), file=sin_file)

sin_file.close()
