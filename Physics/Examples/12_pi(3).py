# -*- coding: utf-8 -*-
"""
PHYS20161 Week 12 Determine pi using Monte Carlo techniques

Fills a square of length one with points and finds the fraction that
have a distance of one or less from the origin, this will be pi/4.

To do this generate a random (x,y) coordinate for each point and count
them if the condition is met.

Lloyd Cawthorne 06/12/19
"""

import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_POINTS = 100


def distance_from_origin(coordinate):
    """
    Given a coordinate outputs distance from centre.
    coordinate np.array([float, float])
    """
    # equivalent to sqrt(x^2 + y^2)
    distance = np.hypot(coordinate[0], coordinate[1])

    return distance


POINTS_INSIDE = np.zeros((0, 2))
POINTS_OUTSIDE = np.zeros((0, 2))

for dummy in range(NUMBER_OF_POINTS):

    point = np.random.random_sample((1, 2))[0]

    if distance_from_origin(point) <= 1:

        POINTS_INSIDE = np.vstack((POINTS_INSIDE, point))

    else:
        POINTS_OUTSIDE = np.vstack((POINTS_OUTSIDE, point))


PI = 4 * len(POINTS_INSIDE) / NUMBER_OF_POINTS

FIGURE = plt.figure(figsize=(6, 6))
AXIS = FIGURE.add_subplot(111)

AXIS.scatter(POINTS_INSIDE[:, 0], POINTS_INSIDE[:, 1], color='r')
AXIS.scatter(POINTS_OUTSIDE[:, 0], POINTS_OUTSIDE[:, 1], color='b')

AXIS.set_title(r'n = {0:d}, $\pi$ = {1:.4f}'.format(NUMBER_OF_POINTS, PI))
AXIS.set_xlim(0, 1)
AXIS.set_ylim(0, 1)
plt.show()
