# -*- coding: utf-8 -*-
"""
PHYS20161 Week 8: Newton Raphson example

Simple example of implementing Newton Raphson to find root of x^2 - 2

Demonstrates convergence of solution.

"""

import numpy as np
import matplotlib.pyplot as plt

# Initial parameters

X_START = 4.
TOLERANCE = 0.001

# Function definitions


def function(x_variable):
    """Returns x^2 - 2
    x_variable (float)
    """
    return x_variable**2 - 2.


def derivative(x_variable):
    """Returns  2x, derivative of x^2 - 2
    x_variable (float)
    """
    return 2. * x_variable


def next_x(previous_x):
    """
    Returns next value for x according to algorithm

    previous_x (float)

    """

    return previous_x - function(previous_x) / derivative(previous_x)


# Main code

# Main algorithm

# set up parameters
difference = 1
counter = 0

differences = np.array([])
roots = np.array([])

x_root = X_START
roots = np.append(roots, x_root)
# Repeatedly find x_n until the tolerance threshold is met.

while difference > TOLERANCE:

    counter += 1

    x_test = x_root
    x_root = next_x(x_root)

    difference = abs(x_test - x_root)
    differences = np.append(differences, difference)
    roots = np.append(roots, x_root)

# Solution plot
x_values = np.linspace(0, 5, 100)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.plot(x_values, function(x_values))
ax.plot(x_values, 0 * x_values, c='grey', dashes=[4, 4])
ax.scatter(roots[0], function(roots[0]), c='grey', label='x_0')
ax.scatter(roots[1], function(roots[1]), c='b', label='x_1')
ax.scatter(roots[2], function(roots[2]), c='g', label='x_2')
ax.scatter(x_root, function(x_root), c='k', label='x_root')

ax.plot(roots[:2], np.full(2, function(roots[0])), c='red',
        dashes=[4, 4], alpha=0.6)
ax.annotate("", xy=(roots[1], function(roots[1])),
            xytext=(roots[1], function(roots[0])),
            arrowprops=dict(arrowstyle="<->", color='red'))

ax.annotate("difference", xy=(roots[1],
                              (function(roots[1]) + function(roots[0])) / 2),
            xytext=(roots[1]-0.9,
                    (function(roots[1]) + function(roots[0]) / 2)-2))

ax.plot(roots[1:3], np.full(2, function(roots[1])), c='red',
        dashes=[4, 4], alpha=0.6)
ax.annotate("", xy=(roots[2], function(roots[2])),
            xytext=(roots[2], function(roots[1])),
            arrowprops=dict(arrowstyle="<->", color='red'))

ax.annotate("difference", xy=(roots[2],
                              (function(roots[2]) + function(roots[1])) / 2),
            xytext=(roots[2]-0.9,
                    (function(roots[2]) + function(roots[1]) / 2)-0.5))

ax.set_xlim(0, 5)
ax.set_ylim(-5, 20)
ax.legend()
plt.savefig('nr_differences.png', dpi=300)
plt.show()

print('2^0.5 = {:6.5f}'.format(x_root))
print('This took {:d} iterations'.format(counter))

# Convergence plot

iterations = np.arange(1, counter + 1)

fig = plt.figure(111)
ax = fig.add_subplot(111)

ax.set_title('Convergence of solution.')
ax.set_xlabel('Iteration')
ax.set_ylabel('differences')
ax.plot(np.arange(counter + 2), np.full(counter + 2, TOLERANCE),
        c='grey', dashes=[4, 4], label='Tolerance')
ax.scatter(iterations, differences, c='k', label='iterations')

ax.set_xlim(0, iterations[-1] + 1)
ax.set_ylim(-0.1, 2)
ax.legend()
plt.savefig('nr_convergence.png', dpi=300)
plt.show()
