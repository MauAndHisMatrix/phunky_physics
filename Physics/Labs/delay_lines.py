"""
Delay Lines
"""

# Standard library imports

# Third-party imports
# from IPython.display import display
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci

# Uni / my modules
# from uni_made import marawan_LSFR_20
# from lab_functions import straight_line_interval
# from assignments.final_assignment import round_to_sig


# Data Import
sine_a_data = pd.read_csv(filepath_or_buffer='https://docs.google.com/spreadsheets/d/e/'\
                                        '2PACX-1vRU2jAM1FWbDyXk5NdsfKcJ1NFiMHZ_kfqd8BocTe53tgH'\
                                        'eJbLERJiBL32it9l--LSndN3kILJZqWK3/pub?gid=0&single=true&output=csv')

sine_b_data = pd.read_csv(filepath_or_buffer='https://docs.google.com/spreadsheets/d/e/2PACX-1vRU2jAM1F'\
    'WbDyXk5NdsfKcJ1NFiMHZ_kfqd8BocTe53tgHeJbLERJiBL32it9l--LSndN3kILJZqWK3/pub?gid=522818256&single=true&output=csv')


def sine_a(data: pd.DataFrame) -> float:
    sine_a_data = data
    freq = np.array(sine_a_data.loc[:10, 'freq(kHz)']).astype(float) * 1e3
    v_in_infinite = np.array(sine_a_data.loc[:54, 'v1(V)']).astype(float)
    v_in_0 = np.array(sine_a_data.loc[:54, 'v2']).astype(float)
    v_in_capacitor = np.array(sine_a_data.loc[:54, 'v3']).astype(float)
    errors = np.array([0.01] * freq.shape[0])

    nat_freq = 1e5

    phase_change = np.arccos(1 - (2 * (2 * np.pi * freq)**2) / nat_freq**2)

    impedance = 1000 / (1 - (2 * np.pi * freq)**2 / nat_freq**2)**0.5

    fig = plt.figure(0)
    axes = fig.add_subplot(111)

    # axes.text(18, 12, f"m = {round(gradient, 2)}\n"\
    #                     f"c = ({round(sound_speed, 2)} +/- {round(sound_error, 2)}) m/s")

    # axes.plot(freq, v_in_infinite, color='red', label=r"$Z_{L} = \infty$")
    # axes.plot(freq, v_in_0, color='blue', label=r"$Z_{L} = 0$")
    # axes.plot(freq, v_in_capacitor, color='green', label=r"$Z_{L} = 0.03 \mu F capacitor$")

    # m, _ = np.polyfit(freq, phase_change, 1)

    # axes.plot(freq, phase_change, label=f'm = {m:.2g}')

    axes.plot(freq, impedance)

    # axes.set_title("Phase change against frequency", fontsize=16, color="black")
    # axes.set_xlabel("Frequency (Hz)", fontsize=12)
    # axes.set_ylabel("Phase change", fontsize=12)
    # axes.set_title("Sine waves at different terminating impedances", fontsize=16, color="black")
    # axes.set_xlabel("Frequency (kHz)", fontsize=12)
    # axes.set_ylabel("V_in (V)", fontsize=12)
    axes.set_title("Characteristic impedance against frequency", fontsize=16, color="black")
    axes.set_xlabel("Frequency (Hz)", fontsize=12)
    axes.set_ylabel("Impedance", fontsize=12)
    

    axes.legend(loc='upper left', fontsize=10)
    # plt.savefig("sine_delay_lines.png", dpi=300)
    plt.savefig("sine_impedance.png", dpi=300)
    # plt.savefig("sine_phase.png", dpi=300)
    plt.show()


def sine_b(data: pd.DataFrame):
    sine_b_data = data
    freq = np.array(sine_b_data.loc[:60, 'freq(kHz)']).astype(float)
    v_10 = np.array(sine_b_data.loc[:60, 'v1(V)']).astype(float)
    v_50 = np.array(sine_b_data.loc[:60, 'v2(V)']).astype(float)
    v_100 = np.array(sine_b_data.loc[:60, 'v3(V)']).astype(float)
    errors = np.array([0.01] * freq.shape[0])


    fig = plt.figure(0)
    axes = fig.add_subplot(111)

    # axes.text(18, 12, f"m = {round(gradient, 2)}\n"\
    #                     f"c = ({round(sound_speed, 2)} +/- {round(sound_error, 2)}) m/s")

    axes.plot(freq, v_10, color='red', label=r"$Z_{L} = 10$")
    axes.plot(freq, v_50, color='blue', label=r"$Z_{L} = 50$")
    axes.plot(freq, v_100, color='green', label=r"$Z_{L} = 100$")

    # m, _ = np.polyfit(freq, phase_change, 1)

    # axes.plot(freq, phase_change, label=f'm = {m:.2g}')


    # axes.plot(freq, np.sin(freq)**2, linewidth=2, label='Sine of frequency')

    # axes.set_title("Phase change against frequency", fontsize=16, color="black")
    # axes.set_xlabel("Frequency (Hz)", fontsize=12)
    # axes.set_ylabel("Phase change", fontsize=12)
    axes.set_title("Sine waves at different terminating impedances", fontsize=16, color="black")
    axes.set_xlabel("Frequency (kHz)", fontsize=12)
    axes.set_ylabel("V_in (V)", fontsize=12)

    axes.legend(loc='upper left', fontsize=10)
    plt.savefig("sine_b.png", dpi=300)
    # plt.savefig("sine_phase.png", dpi=300)
    plt.show()


# def lorentzian(freq, amp, centre, fwhm):
#     return amp * (fwhm**2 / ((freq - centre)**2 + fwhm**2))

# def gaussian(freq, amp, centre, fwhm):
#     sigma = fwhm**2 / (4 * np.log(2))
#     return amp * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1./2.) * ((freq - centre) / sigma)**2)
    # return amp * np.exp((-1./2.) * ((freq - centre) / sigma)**2)


# def fit_distribution(distribution: Callable, x, y, amp: int, cen: float, fwhm: float, errors=None):
#     optimals, covariance = sci.curve_fit(distribution, x, y, p0=[amp, cen, fwhm], sigma=errors, absolute_sigma=False)
#     param_errors = np.sqrt(np.diag(covariance))

#     return optimals, param_errors


def plot_response_curve(data: pd.DataFrame, speed, speed_error) -> None:
    attenuation_data = data
    freq2 = np.array(attenuation_data.loc[:54, 'freq(khz)']).astype(float)
    amp_data = np.array(attenuation_data.loc[:54, 'amplitude(mV)'])
    errors_amp = np.array(attenuation_data.loc[:54, 'error'])

    fig2 = plt.figure(1)
    axes2 = fig2.add_subplot(111)

    axes2.errorbar(freq2, amp_data, yerr=errors_amp, fmt='o', markersize=3, color='green')
    
    [amp, centre, fwhm], param_errors = fit_distribution(lorentzian, freq2, amp_data, 145, 11.96, 0.04, errors_amp)

    print(f"Peak amplitude: {amp:.4f} +/- {param_errors[0]:.4f}")
    print(f"Central frequency: {centre:.4f} +/- {param_errors[1]:.4f}")
    print(f"FWHM: {fwhm:.4f} +/- {param_errors[2]:.4f}")

    quality = centre / fwhm
    q_error = quality * np.sqrt((param_errors[2] / fwhm)**2 + (param_errors[1] / centre)**2)

    alpha = (np.pi * centre * 1e3) / (quality * speed)
    alpha_error = alpha * np.sqrt((param_errors[1] / centre)**2 + (q_error / quality)**2 + (speed_error / speed)**2)

    axes2.plot(freq2, lorentzian(freq2, amp, centre, fwhm), linewidth=2, color='blue')
    fwhm_coods = [[centre - fwhm / 2, centre + fwhm / 2], [amp / 2**0.5, amp / 2**0.5]]
    axes2.plot(fwhm_coods[0], fwhm_coods[1], linewidth=1, color='orange', label=f'FWHM = {fwhm:.3f} kHz')
    axes2.plot([centre, centre], [57, amp], linewidth=2, color='red', label=r'$v_{0}$' + f' = {centre:.3f} kHz')

    axes2.text(11.962, 70, f"Q = ({round(quality, 2)} +/- {round(q_error, 2)}\n"\
                            f"Alpha = ({round(alpha, 2)} +/- {round(alpha_error, 4)}")

    axes2.set_title("Resonant Response Curve", fontsize=16, color="black")
    axes2.set_xlabel("Frequency (kHz)", fontsize=12)
    axes2.set_ylabel("Amplitude (mV)", fontsize=12)

    plt.legend()
    plt.savefig("response_ultrasonics.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    sine_a(sine_a_data)
    # sine_b(sine_b_data)
    # plot_response_curve(attenuation_data, speed, speed_error)

