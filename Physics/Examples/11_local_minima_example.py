# -*- coding: utf-8 -*-
"""
PHYS20161 Wekk 11 example: Global minima

Shows example of function with two minima.
Different starting point leads to determines solution.

Lloyd Cawthorne 26/05/20
"""
import numpy as np
import matplotlib.pyplot as plt

#  Constants

X_START = 2.0
STEP_SIZE = 0.3
TOLERANCE = 0.001

NUMBER_OF_SEARCHES = 1
SEARCH_MIN = -3
SEARCH_MAX = 3

#  Function definitions


def function(x_variable):
    """Returns x^4 - 4x^2 + 2x, float
    Args:
        x_variable (float)
    """
    return x_variable**4 - 4 * x_variable**2 + 2 * x_variable


def hill_climbing(x_minimum=X_START, step=STEP_SIZE):
    """
    Runs 1D hill climbing algorithm with varying step size
    """
    difference = 1
    minimum = function(x_minimum)
    counter = 0

    while difference > TOLERANCE:
        counter += 1
        minimum_test_minus = function(x_minimum - step)
        minimum_test_plus = function(x_minimum + step)
        if minimum_test_minus < minimum:
            x_minimum -= step
            difference = minimum - minimum_test_minus
            minimum = function(x_minimum)
        elif function(x_minimum + step) < minimum:
            x_minimum += step
            difference = minimum - minimum_test_plus
            minimum = function(x_minimum)
        else:
            step = step * 0.1

    return x_minimum, minimum, counter


def plot_result(x_minimum, starting_values):
    """
    Plots function and result
    """
    x_values = np.linspace(-3, 3, 100)

    figure = plt.figure()

    plot = figure.add_subplot(111)
    plot.set_xlabel('x', fontsize=14)
    plot.set_ylabel('f(x)', fontsize=14)

    plot.scatter(starting_values, function(starting_values), label=r'$x_0$',
                 color='grey')
    plot.scatter(x_minimum, function(x_minimum),
                 label=r'$x_{{min}} = {0:.2f}$'.format(x_minimum), color='red')
    plot.plot(x_values, function(x_values))
    plot.legend(fontsize=14)
    plt.show()
    return None


def random_start():
    """Generates a random number between the search bounds.
    np.random.rand() returns a random number between 0 and 1.
    Returns float
    """
    return SEARCH_MIN + np.random.rand() * (SEARCH_MAX - SEARCH_MIN)


def main():
    """
    Main function, runs a 1D hill climbing random search, plots and prints
    results.
    """
    starting_values = np.array([X_START])
    x_minimum, minimum, count = hill_climbing()
    searches = 1
    while searches < NUMBER_OF_SEARCHES:
        searches += 1
        x_start = random_start()
        starting_values = np.append(starting_values, x_start)
        (x_minimum_test, minimum_test,
         count_temp) = hill_climbing(x_minimum=x_start)
        # update solution if better value found
        if minimum_test < minimum:
            minimum = minimum_test
            x_minimum = x_minimum_test
        count += count_temp

    plot_result(x_minimum, starting_values)
    print('f({0:.2f}) = {1:.2f} is the minimum.'.format(x_minimum, minimum))
    print('This took {:d} iterations.'.format(count))
    return 0


main()
