"""
TITLE: PHYS20161 Assignment 2: Z Boson

This script takes in experimental data collected from two different detectors
measuring the cross section of electron-positron collisions at different
energies. It uses the data to determine the mass and width of the Z boson,
along with the following variables:
    a) The lifetime of the Z boson
    b) The reduced chi-squared value

The reduced chi-squared value is an indicator of how well the data fits the
Breit-Wigner expression, approximated as a Gaussian distribution, with the
curve centred at 'mass_z', and the FWHM equivalent to 'width_z'.

Last updated: 12/01/22
Author: Marawan Adam UID: 10458577
"""

# Standard Library
import sys
import glob
from typing import Union, Callable

# Third Party
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.optimize import fmin
from scipy.constants import hbar, eV

# Constants / Filepaths
PARTIAL_WIDTH_EE = 8.391e-2 # GeV

# General data descriptor. Script will find all relevant files
DATA = "z_boson_data"


def round_to_sig(num: float, sig_fig: int=3) -> float:
    """
    This function rounds the given value to the desired number of
    significant figures.

    Parameters:
        num
        sig_fig: The desired no. of sig. figures. 3 is the default.

    Returns:
        The rounded number
    """
    one_sig_fig = - int(np.floor(np.log10(abs(num))))

    return round(num, one_sig_fig - 1 + sig_fig)


def import_data(data_descriptor: str) -> np.ndarray:
    """
    This function locates all files with the common data descriptor and
    imports them. For example, all files beginning with 'z_boson_data' will
    be found and imported. This saves the user the hassle of having to type
    all the filenames out if there are many of them. The imported data is
    added to an array which is then sorted by the first (x) column.

    Parameters:
        data_descriptor: Common string in each filename

    Returns:
        sorted_imported_data
    """
    dataset_paths = glob.glob(f"**/{data_descriptor}*", recursive=True)
    if not dataset_paths:
        print("No files found for given descriptor. Terminating script.")
        sys.exit(0)

    imported_data = np.empty((0, 3))
    for i, dataset in enumerate(dataset_paths):
        try:
            data = np.genfromtxt(dataset, delimiter=',', skip_header=1)
            imported_data = np.vstack((imported_data, data))
        except ValueError:
            # Option to continue not given as target results would be
            # inaccurate given incomplete data.
            print(f"Errors found in dataset {i + 1}. Likely column-related."
                  f"\nRows must have 3 columns: x, y, y_error")
            sys.exit(0)

    sorted_imported_data = imported_data[np.argsort(imported_data[:, 0])]

    return sorted_imported_data


def clean_data(data: np.ndarray) -> np.ndarray:
    """
    This function cleans the data by filtering for the following:
        1) NaNs and negative numbers
        2) Data points containing y values that are over 5 s.d. away
            from the mean

    Parameters:
        data: Raw and potentially flawed data

    Returns:
        cleaned_data
    """
    nan_filter = np.unique(np.where(np.isnan(data))[0])
    negative_filter = np.unique(np.where(data <= 0)[0])

    rows_for_removal = np.hstack((nan_filter, negative_filter))
    cleaned_data = np.delete(data, rows_for_removal, axis=0)

    standard_deviation = np.std(cleaned_data[:, 1])
    mean = np.mean(cleaned_data[:, 1])
    stddev_filter = np.unique(np.where((np.abs(cleaned_data[:, 1] - mean))
                                         > (5 * standard_deviation))[0])
    cleaned_data = np.delete(cleaned_data, stddev_filter, axis=0)

    return cleaned_data


def breit_wigner_function(energies: Union[float, np.ndarray],
                          mass_z: float,
                          width_z: float) -> Union[float, np.ndarray]:
    """
    This function calculates the cross-section of a collision at a specific
    energy, using the Breit-Wigner expression.

    Parameters:
        energies: Either a single energy value, or an array of them.
        mass_z
        width_z

    Returns:
        converted_c_section
    """
    cross_section = ((12 * np.pi) / mass_z**2) * (energies**2 /
                    ((energies**2 - mass_z**2)**2 +
                mass_z**2 * width_z**2)) * PARTIAL_WIDTH_EE**2

    converted_c_section = cross_section * 0.3894e6

    return converted_c_section


def width_from_bw_expression(mass_z: float) -> float:
    """
    This function calculates an approximate value for the width of the
    Z boson that will be used in starting the minimisation process of the
    chi-squared value.
    It is a simplified form of the BW expression as 'E' and 'm' are the
    same value.

    Parameters:
        mass_z

    Returns:
        width
    """
    width_squared = (12 * np.pi * PARTIAL_WIDTH_EE**2) / mass_z**2
    width = np.sqrt(width_squared)

    return width


def energy_from_bw_expression(mass_z: float, width_z: float) -> np.ndarray:
    """
    This function calculates the energy values at the half-maximum stage of
    the BW curve. They are required for the FWHM plot.
    The function utilises the Quadratic Formula for equations in this form:
        ax^2 + bx + c = 0
    This form has been achieved by rearranging the BW expression, having
    equated the expression when at half maximum to half the expression when
    in the simplified maximum form.
    Since the sqrt elements of the calculation must be positive, the first
    energy value involves the subtraction of the sqrt of the discriminant,
    whereas the second energy value involves its addition.

    Parameters:
        mass_z
        width_z

    Returns
        A numpy array containing E1 and E2 at half-maximum
    """
    a_coeff = 1 / (2 * width_z**2)
    b_coeff = - (1 + mass_z**2 / width_z**2)
    c_coeff = 0.5 * mass_z**2 * (mass_z**2 / width_z**2 + 1)

    e2_squared = (- b_coeff + np.sqrt(b_coeff**2 - 4 * a_coeff * c_coeff))\
                        / (2 * a_coeff)
    e1_squared = (- b_coeff - np.sqrt(b_coeff**2 - 4 * a_coeff * c_coeff))\
                        / (2 * a_coeff)

    energy_1 = np.sqrt(e1_squared)
    energy_2 = np.sqrt(e2_squared)

    return np.array([energy_1, energy_2])


def calculate_lifetime(width_z: float) -> float:
    """
    This function calculates the lifetime of the Z boson using its
    width and the adjusted Planck's constant.

    Parameters:
        width_z

    Returns:
        lifetime
    """
    lifetime = hbar / (width_z * 1e9 * eV)

    return lifetime


def chi_squared(data: np.ndarray,
                fitting_expression: Callable,
                *free_parameters) -> float:
    """
    This function calculates the chi-squared value of data being fitted
    to a given function.

    Parameters:
        data
        fitting_expression: The function that the data is being fitted to.
        free_parameters: The free parameters required in the fitting
                        expression's calculations.
    Returns:
        chi_squared_value
    """
    chi_sq_value = 0.
    for data_point in data:
        prediction = fitting_expression(data_point[0], *free_parameters)
        chi_sq_value += ((prediction - data_point[1]) / data_point[2])**2

    return chi_sq_value


def minimise_chi_squared(data: np.ndarray) -> tuple:
    """
    This function minimises the chi-squared value by varying both the mass
    and the width, using the inbuilt 'fmin' function.
    Before the final minimisation occurs, outliers outside 3 s.d. of the
    mean are removed over three iterations.

    Parameters:
        data

    Returns:
        minimised_results[0]: The mass and width values that minimise
                                the chi-squared value. (array)
        minimised_results[1]: The minimised chi-squared value
        filtered_data: Data with outliers removed
    """
    # General starting values for mass and width.
    mass_z_start = data[data.shape[0] // 2, 0]
    width_z_start = width_from_bw_expression(mass_z_start)

    filtered_data = data
    for _ in range(3):
        outlier_mass, outlier_width = \
            fmin(lambda m_w: chi_squared(filtered_data,
                                         breit_wigner_function,
                                         m_w[0], m_w[1]),
                 (mass_z_start, width_z_start), full_output=False, disp=0)
        outlier_cross_sections = breit_wigner_function(
                                        filtered_data[:, 0],
                                        outlier_mass,
                                        outlier_width)
        outliers = np.unique(np.where(
                np.abs(outlier_cross_sections - filtered_data[:, 1])
                > 3 * filtered_data[:, 2]))

        filtered_data = np.delete(filtered_data, outliers, axis=0)

    minimised_results = \
        fmin(lambda m_w: chi_squared(filtered_data, breit_wigner_function,
                                     m_w[0], m_w[1]),
             (mass_z_start, width_z_start), full_output=True, disp=0)

    return minimised_results[0], minimised_results[1], filtered_data


def calculate_reduced_chi(chi_result: float,
                          sample_size: int,
                          free_parameters: int=1):
    """
    This function calculates the reduced chi-squared value of a
    result given the sample size and number of free parameters.

    Parameters:
        chi_result: Chi-squared value
        sample_size: Number of data points
        free_parameters: Number of free parameters in the fitting
                        expression. For example, there are 3 in the case
                        of the BW expression: mass, width and constant.
    """
    deg_of_freedom = sample_size - free_parameters

    return chi_result / deg_of_freedom


def calculate_errors(chi_plus_1_coordinates: np.ndarray,
                     *free_parameters) -> list:
    """
    This function calculates the errors on the free parameters in the
    fitting expression. It does so by subtracting the free params from their
    max values in the coordinates array. This gives us the max distance of
    each variable from the chi-squared + 1 band. All values within this band
    are within 1 s.d. of their minimised form.

    Parameters:
        chi_plus_1_coordinates: Free param coordinates that form the
                                minimum chi-squared + 1 band.
        free_parameters

    Returns:
        errors (generator expression)
    """
    errors = (max(chi_plus_1_coordinates[:, i]) - free_param
              for i, free_param in enumerate(free_parameters))

    return errors


def plot_data(data: np.ndarray, mass_z: float, width_z: float) -> None:
    """
    This function plots a scatter graph of the data, along with a line of
    best fit. It also pin-points the maximum using the value of the mass
    of the Z boson. The FWHM is also plotted.

    Parameters:
        data
        mass_z
        width_z
    """
    x_energies = data[:, 0]
    y_cross_sections = data[:, 1]
    y_errors = data[:, 2]


    fig = plt.figure(0)
    axes = fig.add_subplot(111)
    axes.errorbar(x_energies, y_cross_sections, yerr=y_errors,
                fmt='o', markersize=4, color='green')
    axes.plot(x_energies, breit_wigner_function(x_energies, mass_z, width_z),
            color='blue', linewidth=2, label="Line of best fit")

    # FWHM and Maximum
    e1_and_e2_fwhm = energy_from_bw_expression(mass_z, width_z)
    half_max_c_section = breit_wigner_function(e1_and_e2_fwhm,
                                               mass_z, width_z)

    axes.plot(e1_and_e2_fwhm, half_max_c_section, color='orange',
              linewidth=2,
              dashes=[6, 3], label=r"FWHM / $\Gamma_{z}$")
    axes.plot([mass_z, mass_z], [0, 2 * half_max_c_section[0]],
              color='brown',
              linewidth=2, dashes=[6, 3])

    axes.set_title("Centre of mass energy vs cross-section\
                 \nof each electron-positron collision",
                 fontsize=16, color="blue")
    axes.set_xlabel("Centre of mass energy (GeV)", fontsize=12)
    axes.set_ylabel("Cross section of reaction (nanobarns)", fontsize=12)

    # Following code puts 'm_z' on the x-axis.
    x_ticks = sorted(list(axes.get_xticks()) + [mass_z])
    axes.set_xticks(x_ticks)
    mass_index = x_ticks.index(mass_z)
    x_ticks[mass_index] = r"$m_{z}$"
    axes.set_xticklabels(x_ticks)
    axes.get_xticklabels()[mass_index].set_color("brown")

    axes.tick_params(axis='x', labelsize=12)
    axes.tick_params(axis='y', labelsize=12)

    axes.text(85, 1.4, r"$m_{z}$ = " + f"{round_to_sig(mass_z, sig_fig=4)} "
            + r"$GeV/c^{2}$" + "\n" +
            r"$\Gamma_{z}$ = " + f"{round_to_sig(width_z, sig_fig=4)} GeV")

    axes.set_ylim(bottom=0)
    axes.legend()
    plt.savefig("com_energy_vs_cross_section.png", dpi=300)
    plt.show()


def plot_contour(data: np.ndarray,
                 mass_z: float,
                 width_z: float,
                 min_chi_squared: float) -> np.ndarray:
    """
    This function plots a contour graph that visualises the minimisation
    of the chi-squared value, and the errors on the free parameters
    (mass, width). The coordinates of the free parameters at the min.
    chi-squared value + 1 band are obtained from the graph, which are then
    used to calculate their errors.

    Parameters:
        data
        mass_z
        width_z
        min_chi_squared

    Returns:
        chi_plus_1_coordinates: Coordinates of mass, width at the min.
                            chi-squared + 1 band.
    """
    mass_range = np.linspace(91.123, 91.235, num=50)
    width_range = np.linspace(2.455, 2.565, num=50)
    mass_mesh, width_mesh = np.meshgrid(mass_range, width_range)

    fig = plt.figure(1)
    axes = fig.add_subplot(111)

    contour_increments = [1, 2.3, 5.99, 9.21, 13.5]
    contour_levels = [x + min_chi_squared for x in contour_increments]

    full_contour = axes.contour(mass_mesh, width_mesh,
                              chi_squared(data, breit_wigner_function,
                                          mass_mesh, width_mesh),
                              levels=contour_levels[1:])
    chi_plus1_contour = axes.contour(mass_mesh, width_mesh,
                                   chi_squared(data, breit_wigner_function,
                                               mass_mesh, width_mesh),
                                   levels=[contour_levels[0]],
                                   linestyles='dashed')

    axes.scatter(mass_z, width_z, color='r',
               label=r"$\chi^{2}_{min.}$ = " + f"{min_chi_squared:.3f}")

    axes.clabel(full_contour, inline=1, fontsize=10)
    axes.clabel(chi_plus1_contour, inline=1, fontsize=10,
              fmt=r"$\chi^{2}_{min.}+ 1$")

    axes.set_xlabel(r'Mass of Z boson ($GeV/c^{2}$)')
    axes.set_ylabel('Width of reaction (GeV)')
    axes.set_title('Contour plot that varies both the \n\
        mass and the width to minimise the chi-squared value')
    axes.legend(loc='upper left')

    plt.savefig("contour_minimising_chi_squared.png", dpi=300)
    plt.show()

    chi_plus_1_coordinates = \
        chi_plus1_contour.collections[0].get_paths()[0].vertices

    return chi_plus_1_coordinates


def main():
    """
    Main function that executes all the calculations.
    """
    imported_data = import_data(DATA)
    cleaned_data = clean_data(imported_data)

    [mass_z, width_z], min_chi_squared, filtered_data = \
        minimise_chi_squared(cleaned_data)

    red_chi = calculate_reduced_chi(min_chi_squared,
                                    filtered_data.shape[0],
                                    free_parameters=3)

    lifetime = calculate_lifetime(width_z)

    plot_data(filtered_data, mass_z, width_z)
    chi_plus_1_coordinates = plot_contour(filtered_data,
                                          mass_z, width_z, min_chi_squared)

    mass_error, width_error = calculate_errors(chi_plus_1_coordinates,
                                               mass_z, width_z)
    lifetime_error = lifetime * (width_error / width_z)

    print(f"\nCalculated values of mass and width of Z boson are:\
                \nmass_z = {round_to_sig(mass_z, sig_fig=4)}" \
                f" +/- {mass_error:.1g} GeV/c^2\
                \nwidth_z = {round_to_sig(width_z, sig_fig=4)}" \
                f" +/- {width_error:.2g} GeV\n\
            \nZ boson's lifetime: {round_to_sig(lifetime, sig_fig=3):.3g}" \
            f" +/- {lifetime_error:.2g} s\n\
            \nReduced chi squared: {red_chi:.3f}\n")


if __name__ == "__main__":
    main()
