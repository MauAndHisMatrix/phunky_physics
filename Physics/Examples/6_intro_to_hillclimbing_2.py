# -*- coding: utf-8 -*-
"""
PHYS20161 Week 6 Example maximum of parabola.

Demonstrates and visualises hill-climbing approach for finding maximum of
a parabola.

Lloyd Cawthorne 15/10/20

"""

import numpy as np
import matplotlib.pyplot as plt

# Fitting parameters

X_START = -1.0
STEP = 0.001
TOLERANCE = 0.00001
MAX_COUNTER = 5000

# Functions


def parabola(x_variable):
    """Returns -2x^4 - 4x^2 + 5x - 10
    x_variable (float)
    """
    return -2 * x_variable**4 - 4. * x_variable**2 + 5. * x_variable + 10.


# Main code

x_values = np.linspace(-2., 2., 100)

# Set initial parameters
x_maximum = X_START
parabola_maximum = parabola(X_START)

# Display plot with startig point
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x_maximum, parabola_maximum, label='Starting_point', color='k')

# Set counter and difference
counter = 0
difference = 1

# Hillclimbing algorithm
while difference > TOLERANCE:

    counter += 1

    # Look either side of point
    parabola_test_plus = parabola(x_maximum + STEP)
    parabola_test_minus = parabola(x_maximum - STEP)

    # Set new values if a better point is found
    if parabola_test_plus >= parabola_maximum:

        difference = parabola_test_plus - parabola_maximum
        parabola_maximum = parabola_test_plus
        x_maximum += STEP

    elif parabola_test_minus >= parabola_maximum:

        difference = parabola_test_minus - parabola_maximum
        parabola_maximum = parabola_test_minus
        x_maximum -= STEP

    # No solution found
    else:
        print('Procedure failed. Last found difference is {0:f}.'.
              format(difference))
        break
    # Procedure timed out
    if counter == MAX_COUNTER:
        print('Procedure failed. Timed out after {0:d} iterations.'.
              format(counter))
        break

# Plot result
ax.plot(x_values, parabola(x_values))
ax.set_xlabel('x values')
ax.set_ylabel('f(x)')
ax.scatter(x_maximum, parabola_maximum, label='Maximum', color='r')
ax.legend()
plt.show()

print('f({0:.3f}) = {1:.3f} is the maximum point'.
      format(x_maximum, parabola_maximum))
print('This took {0:d} iterations.'.format(counter))
