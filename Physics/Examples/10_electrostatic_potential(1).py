# -*- coding: utf-8 -*-
"""
PHYS20161 Week 9 Example of contour plot to show potential

Code evaulates the contour lines for the electrostatic potential from
two charges. Utilises a variety of aspects introduce recently.

Lloyd Cawthorne 24/11/20
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc

CHARGE_1 = 1 * pc.elementary_charge
CHARGE_2 = -1 * pc.elementary_charge

# in pico metres (e-12)
LOCATION_1 = np.array([0, 1])
LOCATION_2 = np.array([0, -1])

ZOOM = 10

def mesh_arrays(x_array, y_array):
    """Returns two meshed arrays of size len(x_array)
    by len(y_array)
    x_array array[floats]
    y_array array[floats]
    """
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in y_array:  # PyLint accepts _ as an uncalled variable.
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for dummy_element in x_array:  # PyLint accepts dummy_anything as well.
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh


def modulus_difference(vector_1, vector_2):
    """
    Finds |vector_1 - vector_2|

    Parameters
    ----------
    vector_1 : np.array([float, float])
    vector_2 : np.array([float, float])

    Returns
    -------
    result : float
    """

    result = np.sqrt(vector_1[0]**2 + vector_1[1]**2 + vector_2[0]**2
                     + vector_2[1]**2 - 2 * (vector_1[0] * vector_2[0]
                                             + vector_1[1] * vector_2[1]))
    return result

def electric_potential(distance, charge_1, charge_2, location_1, location_2):
    """
    Finds the electostatic potential from two charges given their charge
    and position.


    Parameters
    ----------
    distance : np.array([float, float])
        Position of interst from origin
    charge_1 : float
    charge_2 : float
    location_1 : np.array([float, float])
    location_2 : np.array([float, float])

    Returns
    -------
    potential : float
    """
    constant = 1 / (4 * np.pi * pc.epsilon_0 * pc.pico)

    potential = constant * (charge_1 / (modulus_difference(distance,
                                                           location_1))
                            + charge_2 / (modulus_difference(distance,
                                                             location_2)))
    return potential


def plot_potential(data_1=(CHARGE_1, LOCATION_1),
                   data_2=(CHARGE_2, LOCATION_2),
                   zoom=ZOOM):
    """
    Plots electrostatic potential contours around the chrages.
    Parameters
    ----------
    data_1 : (float, np.array([float, float])), optional
        The default is (CHARGE_1, LOCATION_1).
    data_2 : (float, np.array([float, float])), optional
        The default is (CHARGE_2, LOCATION_2).
    zoom : float, optional
        The default is ZOOM.

    Returns
    -------
    None.

    """

    # Unpack arguments
    charge_1, location_1 = data_1
    charge_2, location_2 = data_2

    # Define plotting region in pm
    max_value = 5 * zoom * np.max(np.hstack((location_1, location_2)))
    min_value = 5 * zoom * np.min(np.hstack((location_1, location_2)))

    # Cartesian axis system with origin at the dipole (m)
    x_values_mesh = np.linspace(min_value, max_value, 100)
    y_values_mesh = x_values_mesh.copy()
    x_values_mesh, y_values_mesh = mesh_arrays(x_values_mesh, y_values_mesh)

    electrostatic_potential = electric_potential([x_values_mesh,
                                                  y_values_mesh],
                                                 charge_1, charge_2,
                                                 location_1, location_2)

    figure = plt.figure(figsize=(8, 8))
    axes = figure.add_subplot(111)

    # Draw contours at values of potential given by levels
    # These must be given in order.
    levels = np.array([10**power for power in np.linspace(0, 5, 20)])
    levels = sorted(list(-levels) + list(levels))
    # Monochrome plot of potential, negative values presented dashed.
    axes.contour(x_values_mesh, y_values_mesh, electrostatic_potential,
                 levels=levels, colors='k')
    # Display charges
    axes.scatter(location_1[0], location_1[1], color='r', s=100)
    axes.scatter(location_2[0], location_2[1], color='b', s=100)

    axes.set_title('Electrostatic potential from two charges.', fontsize=14)
    axes.set_xlabel('x (pm)', fontsize=14)
    axes.set_ylabel('y (pm)', fontsize=14)

    axes.tick_params(axis='both', labelsize=16)

    plt.show()

    return None

plot_potential()
