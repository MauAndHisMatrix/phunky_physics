# -*- coding: utf-8 -*-
"""
PHYS20161 Week 9 plt.figure() example

Showcase a variety of options

Lloyd Cawthorne 14/11/19

"""

import numpy as np
import matplotlib.pyplot as plt

X_VALUES = np.linspace(-np.pi, np.pi, 50)

plt.figure(figsize=(4.8, 6.4),
           facecolor='cyan',
           linewidth=3,
           edgecolor='y')

plt.plot(X_VALUES, np.sin(X_VALUES), label='sin(x)', color='red',
         linewidth=5)

plt.legend()
plt.savefig('sinplot.png',
            facecolor='cyan',
            linewidth=3,
            edgecolor='y')

plt.show()
