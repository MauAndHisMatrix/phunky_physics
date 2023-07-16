"""
Ultrasonics Script 2
"""

# Standard library imports

# Third-party imports
# from IPython.display import display
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci
import lmfit as lm
# Uni / my modules
import marawan_LSFR
# from uni_made import marawan_LSFR_20
# from lab_functions import straight_line_interval
# from Physics.Assignments.marawan_final_assignment import round_to_sig


# Data Import
# calib_data = pd.read_csv(filepath_or_buffer='https://docs.google.com/spreadsheets/d/e/2PACX-1vTz-kEUZalR'\
#                 'IwckibKamFwbV0UTFx9TFM66YECyJB8ae_ZD_fr7l5YE2QmzcS-Y5rNUOtx1N91yL-u_/pub?gid=0&single=true&output=csv')

DATA = "gamma.csv"

def plot_data(data):

    fig = plt.figure(1)
    axes = fig.add_subplot(111)

    eff = data[:,2] / (75.44 * data[:, 1])

    error = 3 * np.sqrt(eff)

    axes.plot(data[:, 0], eff)
    axes.errorbar(data[:, 0], eff, yerr=error, color="b")

    axes.set_xlabel("Energy (keV)")
    axes.set_ylabel("Efficiency of detector")
    axes.set_title(r"Energy vs Detector Efficiency for $^{152}Eu$")

    # m, c = np.polyfit(data[:, 0], data[:, 3], deg=1)
    # axes.plot(data[:, 0], m * data[:, 0] + c)

    # plt.show()
    plt.savefig("gamma.png", dpi=300)

    # print(f"m = {m:.3f} * x + {c:.3f}")

def import_data(dataset):
    data = np.genfromtxt(dataset, delimiter=',', skip_header=1)

    return data[np.argsort(data[:, 0])]


if __name__ == "__main__":
    data = import_data(DATA)
    # print(data)

    plot_data(data)