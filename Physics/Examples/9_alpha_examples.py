# -*- coding: utf-8 -*-
"""
PHYS20162 Week 9 example of plotting opacity

Demonstrates plotting with various opacities.

Lloyd Cawthorne 17/11/2019

"""

import numpy as np
import matplotlib.pyplot as plt

X_VALUES = np.arange(50)

LINE_VALUES = 2 * X_VALUES

SHIFTED_LINE_VALUES = LINE_VALUES + 10

# Generate 50 random integers between 0 and 100
SCATTER_VALUES = np.random.randint(0, 100, 50)

plt.figure()
plt.title('Examples of alpha', fontsize=16)

# set marker size, s, proportional to value
plt.scatter(X_VALUES, SCATTER_VALUES, s=2 * SCATTER_VALUES, alpha=0.7,
            label='Scatter')
plt.plot(X_VALUES, LINE_VALUES, c='k', label='No alpha', linewidth=4)

plt.plot(X_VALUES, SHIFTED_LINE_VALUES, c='k',
         alpha=0.5, label='Alpha = 0.5',
         linewidth=4)

plt.legend(fontsize=16)

plt.savefig('alpha.png', dpi=300, transparent=True)

plt.show()
