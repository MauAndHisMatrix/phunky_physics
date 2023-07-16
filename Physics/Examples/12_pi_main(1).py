# -*- coding: utf-8 -*-
"""
PHYS20161 Week 12 Determine pi using Monte Carlo techniques written with main()

Fills a square of length one with points and finds the fraction that
have a distance of one or less from the origin, this will be pi/4.

To do this generate a random (x,y) coordinate for each point and count
them if the condition is met.

Lloyd Cawthorne 28/05/20
"""

import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_POINTS = 10000


def distance_from_origin(coordinate):
    """
    Given a coordinate outputs distance from centre.
    coordinate np.array([float, float])
    """
    # equivalent to sqrt(x^2 + y^2)
    distance = np.hypot(coordinate[0], coordinate[1])

    return distance


def generate_and_separate_points():
    """
    Randomly generates points and sorts them into within the arc and outside.
    """
    points_inside = np.zeros((0, 2))
    points_outside = np.zeros((0, 2))

    for dummy in range(NUMBER_OF_POINTS):
        point = np.random.random_sample((1, 2))[0]
        if distance_from_origin(point) <= 1:
            points_inside = np.vstack((points_inside, point))
        else:
            points_outside = np.vstack((points_outside, point))
    return points_inside, points_outside


def plot(points_inside, points_outside, pi_value):
    """Plots grid given coordinates for points inside and outside and
    calculated value of pi.
    Args:
        points_inside: array of [float, float]
        points_outside: array of [float, float]
        pi: float
    Returns:
        None
    """
    fig = plt.figure(figsize=(6, 6))
    axis = fig.add_subplot(111)
    axis.scatter(points_inside[:, 0], points_inside[:, 1], color='r')
    axis.scatter(points_outside[:, 0], points_outside[:, 1], color='b')
    axis.set_title(r'n = {0:d}, $\pi$ = {1:.4f}'.format(NUMBER_OF_POINTS,
                                                        pi_value))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    plt.show()
    return None


def main():
    """Genearates and separates points, calculates pi, plots result.
    """
    points_inside, points_outside = generate_and_separate_points()
    pi_value = 4 * len(points_inside) / NUMBER_OF_POINTS
    plot(points_inside, points_outside, pi_value)
    return 0


main()
