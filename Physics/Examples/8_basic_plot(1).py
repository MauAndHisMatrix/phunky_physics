# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8 Basic example of plot

Lloyd Cawthorne 08/11/19

Plots y = x^2 from x=0 to 99

"""

import numpy as np
import matplotlib.pyplot as plt

x_values = np.arange(100)
y_values = x_values**2

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('Title string goes here', fontsize=16, color='red')

ax.set_xlabel('x label string goes here', fontsize=10, color='r')
ax.set_ylabel('y label string goes here', fontsize=12, color='blue')

ax.tick_params(axis='x', labelsize=20, labelcolor='green')

ax.set_yticks(np.arange(500, 9500, 500))
ax.tick_params(axis='y', labelsize=15, labelcolor='#741919')

ax.plot(x_values, y_values, linewidth=4, color='purple')

plt.show()
