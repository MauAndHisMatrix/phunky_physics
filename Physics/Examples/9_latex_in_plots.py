# -*- coding: utf-8 -*-
"""
PHYS20161 Week 9 mathematical expressions example

Demonstrates using LaTeX commands in Python to write Greek symbols and
mathematical symbols.

Compare group and phase velocities of a non-relativistic electron

omega(k) = hbar k / 2 m

v_p = omega / k

v_g = d omega / d k

Lloyd Cawthorne 18/11/19

"""

import numpy as np
import matplotlib.pyplot as plt

HBARC = 0.197  # 10^-6 m eV
MASS_ELECTRON = 0.511 * 10**6  # eV/c^2


def group_velocity(wave_number):
    """Returns group velocity for plane wave in Schrodinger equation

    wave_number (float)
    """

    return HBARC * wave_number / (2 * MASS_ELECTRON)


def phase_velocity(wave_number):
    """Returns phase velocity for plane wave in Schrodinger equation

    k (float)
    """

    return HBARC * wave_number / (MASS_ELECTRON)


WAVE_NUMBER_VALUES = np.linspace(5000, 16000, 100)

plt.figure()

plt.title(r'$\frac{\omega}{k}$ vs. $\frac{d \omega}{d k}$',
          fontsize=16)
plt.xlabel(r'$k$ ($\mu$m$^{-1}$)', fontsize=16)
plt.ylabel(r'velocity $/ c$', fontsize=16)

plt.plot(WAVE_NUMBER_VALUES, group_velocity(WAVE_NUMBER_VALUES),
         label=r'$v_g(k) = \frac{\hbar k}{2 m_e}$', linewidth=3)
plt.plot(WAVE_NUMBER_VALUES, phase_velocity(WAVE_NUMBER_VALUES),
         label=r'$v_p(k) = \frac{\hbar k}{m_e}$', linewidth=3)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)
plt.savefig('electron velocities.png', dpi=300, bbox_inches="tight")

plt.show()
