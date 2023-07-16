# -*- coding: utf-8 -*-
"""
PHYS20161 Week 12 Example: Schelling model

The Schelling model is an agent-based model where two species relocate
depending on their immediate neighbours. It is often used to model urban
environments such as gentrification.

Schelling, T. C. (1971). Dynamic models of segregation. Journal of mathematical
sociology, 1(2), 143-186.

In it we take one cell at random and examine the fraction of similar immediate
neighbours. We then take another cell at random. If the second cell is empty,
we find the the fraction of similar neighbour. If there is a greater fraction,
around the empty cell, we move the original.

We can also introduce a threshold where a cell is 'content' with its
surrounding and will not attempt to relocate.

E.g.

This central X has 1/6 similar neighbours
O   X   Y
Y   X   Y
O   Y   Y

An empty square has 6/7 similar neighbours -> a move is favourable.

X   X   X
Y   O   O
X   X   X

We code this with cyclic boundaries, left of cell[0] is cell[-1].

Lloyd Cawthorne 27/05/20

"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
FILLING = 0.8  # Fraction of grid filled
GRID_SIZE = 100  # GRID_SIZE x GRID_SIZE grid
THRESHOLD = 1  # A satisfaction above this threshold means no desire to move.
# Fractions defined wrt filling
RATIO_X_OVER_Y = 1  # number of X / number of Y
NUMBER_OF_SWEEPS = 10


def get_coordinate():
    """Returns a random coordinate as a tuple"""
    return tuple(np.random.randint(0, high=GRID_SIZE, size=(1, 2))[0])


def define_grid():
    """
    Fills grid according to constants in preamble.
    Returns GRID_SIZE by GRID_SIZE array of 1s (X), -1s (Y) and 0s (empty).
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype='i4')

    number_of_x = ((RATIO_X_OVER_Y / (RATIO_X_OVER_Y + 1.)) * FILLING
                   * GRID_SIZE**2)
    number_of_y = (1. / (RATIO_X_OVER_Y + 1.)) * FILLING * GRID_SIZE**2

    while number_of_x + number_of_y > 0:
        coordinate = get_coordinate()
        if grid[coordinate] == 0:
            if number_of_x > 0 and number_of_y > 0:
                # randomly pick one to fill
                if np.random.randint(0, high=2) == 0:
                    grid[coordinate] = 1
                    number_of_x -= 1
                else:
                    grid[coordinate] = -1
                    number_of_y -= 1
            elif number_of_x > 0:
                grid[coordinate] = 1
                number_of_x -= 1
            else:
                grid[coordinate] = -1
                number_of_y -= 1
    return grid


def plot_grid(grid, iterations):
    """Plots the grid.
    Args:
        grid: GRID_SIZE by GRID_SIZE array of 1s, 0s and -1s.
        iterations: int, number of swaps attempted. Used in title of plot
    """
    fig = plt.figure()
    axis = fig.add_subplot(111)

    axis.set_title('Grid after {0:d} iterations.'.format(iterations))
    colour_mesh = axis.pcolormesh(grid, cmap='bwr')
    fig.colorbar(colour_mesh, ax=axis)
    plt.show()
    return None


def satisfaction(grid, coordinate, cell):
    """Returns inspects surrounding nearest and next-nearest neighbours of cell
    Args:
        grid: GRID_SIZE by GRID_SIZE array of 1s, 0s and -1s
        corrdinate: tuple(int, int), where each entry is between 0 and
                    GRID_SIZE
        cell: int, contents of cell: -1, 0 or 1
    Returns:
        fraction of like neighbours, float
        """
    number_of_neighbours = 0
    neighbours_total = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue  # do nothing
            else:
                neighbour = grid[(coordinate[0] + i) % GRID_SIZE,
                                 (coordinate[1] + j) % GRID_SIZE]
                if neighbour != 0:
                    if neighbour == cell:
                        neighbours_total += 1
                    number_of_neighbours += 1
    if number_of_neighbours == 0:
        return 1.
    return neighbours_total / number_of_neighbours


def compare_two_cells(grid):
    """Selects a filled cell, inspects its neighbours and compars that to an
    empty cell
    Args:
        GRID_SIZE by GRID_SIZE array of 1s, 0s and -1s
    Returns:
        GRID_SIZE by GRID_SIZE array of 1s, 0s and -1s
        """
    cell_0 = 0
    while cell_0 == 0:
        coordinate_0 = get_coordinate()
        cell_0 = grid[coordinate_0]
    satisfaction_0 = satisfaction(grid, coordinate_0, cell_0)
    if satisfaction_0 > THRESHOLD:
        return grid

    cell_1 = 1
    while cell_1 != 0:
        coordinate_1 = get_coordinate()
        cell_1 = grid[coordinate_1]
    satisfaction_1 = satisfaction(grid, coordinate_1, cell_0)

    if satisfaction_1 >= satisfaction_0:

        grid[coordinate_0] = cell_1
        grid[coordinate_1] = cell_0
    return grid


def main():
    """Runs main algorithm:
        -Defines grid according to constants
        -For each sweep attempts size^2 swaps
        -Plots result after each sweep
    """
    grid = define_grid()
    plot_grid(grid, 0)
    counter = None
    for sweeps in range(NUMBER_OF_SWEEPS):
        for counter in range(GRID_SIZE**2):
            grid = compare_two_cells(grid)
        plot_grid(grid, (sweeps + 1) * (counter + 1))
    return 0


main()
