# -*- coding: utf-8 -*-
"""
PHYS20161 Week 7 read data example

Writes data for sin and cos for angles from 0 to 90 degrees.

Lloyd Cawthorne 04/11/19

"""
import numpy as np

angles = np.arange(0, 91)

sins = np.sin(np.deg2rad(angles))
coss = np.cos(np.deg2rad(angles))

output_file = open('trig_data_2.txt', 'w')

print('% sin  cos', file=output_file)

# enumerate returns the index and value of the array each iteration, we've used
# 'dummy' here to ignore the latter.

for index, dummy in enumerate(angles):
    print('{0:4.3f}   {1:4.3f}'.format(sins[index], coss[index]),
          file=output_file)

output_file.close()
