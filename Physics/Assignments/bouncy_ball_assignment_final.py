"""
TITLE: PHYS20161 Assignment 1: Bouncy Ball

This script performs calculations based on the physics of a bouncing ball.
It calculates the following:
    a) The number of bounces above the minimum height.
    b) The time taken to complete the bounces.

To do so, the following parameters are provided by the user:
    - Minimum height
    - Initial height (ball dropped from)
    - Efficiency factor (The factor by which the ball's energy decreases
                         each bounce)

Last updated: 22/10/22
Author: Marawan Adam UID: 10458577
"""

# Standard Library
import sys
from typing import Union

# Third Party
import numpy as np
import matplotlib.pyplot as plt

# Constants
# All SOLAR_SYSTEM values are in m/s^2
SOLAR_SYSTEM = {"mercury": 3.70, "venus": 8.87, "earth": 9.81, "mars": 3.71,
                "jupiter": 24.92, "saturn": 10.44, "neptune": 8.87,
                "uranus": 11.75, "pluto": 0.49}


def validate_heights() -> float:
    """
    This function receives the user's inputs for the initial height of the
    ball drop, and the minimum height over which the bounces are counted.

    Returns
        initial_height
        minimum_height
    """
    for i in range(1, 4):
        print(f"\nAttempt {i} of heights input.\n")
        try:
            initial_height = float(input("Enter the initial height:"))
            min_height = float(input("Enter the minimum height:"))

            if initial_height <= 0 or min_height <= 0:
                raise ValueError

            if min_height >= initial_height:
                print("The minimum height must be lower than the"\
                      " initial height.")
                raise ValueError

        except ValueError:
            print("The heights must be numbers and above 0.")
            continue

        else:
            print("Heights validated.")
            return initial_height, min_height

    print("You have run out of attempts, the program will terminate.")
    sys.exit(0)


def validate_efficiency() -> float:
    """
    This function receives the user's input for the efficiency factor and
    validates it.

    Returns
        efficiency
    """
    for i in range(1, 4):
        print(f"\nAttempt {i} of efficiency input.\n")
        try:
            efficiency = float(input("Enter the Efficiency factor:"))

            if not 0 < efficiency < 1:
                print("\nThe efficiency must be exclusively between 0 and 1.")
                raise ValueError

            if efficiency > 0.999999:
                print("\nEfficiency is too close to 1, which will yield a"\
                    " processor-melting number of bounces!")
                raise ValueError

        except ValueError:
            print("The value(s) entered is invalid, try again.")
            continue

        else:
            print("Efficiency validated.")
            return efficiency

    print("You have run out of attempts, the program will terminate.")
    sys.exit(0)


def choose_planet() -> float:
    """
    This function asks the user to choose a planet on which the bouncing ball
    simulation will take place. The significant difference between the planets
    is their gravitational acceleration constant. If no planet is chosen,
    or the input is invalid, the default planet is Earth.

    Returns:
        grav_const: The gravitational acceleration of the chosen planet.
    """
    print(f"\nSolar System planets: {list(SOLAR_SYSTEM.keys())}")
    planet = input("\nEnter your planet of choice from the "\
                        "solar system. DEFAULT=earth:")
    grav_const = SOLAR_SYSTEM.get(planet, 9.81)

    if grav_const == 9.81:
        print("\nChosen planet: earth")
    else:
        print(f"\nChosen planet: {planet}")

    return grav_const


def user_input_parameters() -> tuple:
    """
    This function receives the user's inputs and performs validation
    checks on them by calling their respective validation funcitons. It also
    asks the user if they want the ball's path to be plotted. Any invalid
    entry defaults to a 'False' boolean.

    Returns:
        initial_height
        min_height
        efficiency: Efficiency factor.
        grav_const: Chosen planet dictates the gravitational constant to be
                    used in calculation.
        plot_path_bool: Boolean that determines whether or not the path is
                        plotted later.
    """
    print("You have 3 attempts at each part of the parameter input stage.")

    initial_height, min_height = validate_heights()

    efficiency = validate_efficiency()

    grav_const = choose_planet()

    plot_path_query = input("Would you like the bouncing ball's "
                            + "path plotted? (For no_of_bounces > 0) [y/n]")

    plot_path_bool = plot_path_query in ["y", "yes"]

    return initial_height, min_height, efficiency, grav_const, plot_path_bool


def number_of_bounces(initial_height: float, min_height: float,
                      efficiency: float) -> int:
    """
    This function calculates the number of times the ball bounces
    above the minimum height. It uses a mathematical rearrangement of the
    following equation:
                            m*g*h_min = m*g*h_initial*eta^n

    Where 'eta' is the efficiency factor and 'n' is the number of bounces.

    Parameters:
        initial_height: User-inputted initial height of ball.

        min_height: Minimum height ball must bounce over.

        efficiency: The Efficiency factor. The ball's energy
        decreases by this factor every bounce. Must be
        between 0 and 1.

    Returns:
        Number of bounces above min_height. The calculation returns a decimal
        that gets rounded down to the next integer. Unless, the value is an
        integer, in which case 1 less is returned, as the assignment requires
        bounces *over* the min height, not 'at' the height.
    """
    num = float((np.log(min_height / initial_height) / np.log(efficiency)))
    if num.is_integer():
        num -= 1
    # int() both converts 'n' to an integer and rounds down if it's a float.
    return int(num)


def first_drop_outputs(initial_height: float, grav_const: float) -> tuple:
    """
    This function calculates the time elapsed during the first drop and
    the speed of the ball as it hits the ground. These values are useful
    in setting up the iterative calculations for the bounces.

    Parameters:
        initial_height: Initial height of the ball before release.
        grav_const

    Returns:
        time_taken: Time elapsed during drop.
        final_speed: Speed of ball as it hits the ground.

    Both are calculated using kinematic equations.
    """
    time_taken = ((2 * initial_height) / grav_const)**0.5
    final_speed = (2 * grav_const * initial_height)**0.5

    return time_taken, final_speed


def bounce_time(initial_speed: float, grav_const: float) -> float:
    """
    This function calculates how long a bounce lasts given its initial
    speed off the ground. A kinematic equation is used to do so.

    Parameters:
        initial_speed: Initial speed of ball off surface.
        grav_const

    Returns:
        Time elapsed during bounce.
    """
    return (2 * initial_speed) / grav_const


def calculate_total_time(no_of_bounces: int,
                         first_drop_time: float,
                         first_drop_speed: float,
                         efficiency: float,
                         grav_const: float) -> tuple:
    """
    This function calculates the total time elapsed from the moment the
    ball is dropped, to the end of the last bounce that surpasses the
    minimum height. It calls the 'bounce_time' function when adding the
    individual bounce times.

    Parameters:
        no_of_bounces
        first_drop_time: Time elapsed during the first drop.
        first_drop_speed: Speed of the ball just BEFORE the first bounce.
        efficiency
        grav_const

    Returns:
        total_time
        individual_bounce_attributes: A list of tuples containing information
                                    about each individual bounce, which will
                                    be unpacked in the plot function.
    """
    # First element contains the information required to plot the first drop.
    individual_bounce_attributes = [(0, 0, first_drop_time, 0)]
    total_time = first_drop_time
    initial_speed = first_drop_speed * efficiency**0.5

    answer_boolean = False
    # 10 seems a reasonable limit for displaying individual times.
    if no_of_bounces <= 10:
        answer = input("\nWould you like the times of each bounce? [y/n]")
        answer_boolean = answer.lower() in ["y", "yes"]

    time_shift = float(0.)
    for i in range(no_of_bounces + 1):
        individual_time = bounce_time(initial_speed, grav_const)

        # Useful variable for the plot later on
        if i == 0:
            time_shift += initial_speed / (grav_const * efficiency**0.5)
        else:
            time_shift += (2 * initial_speed) / (grav_const * efficiency**0.5)

        individual_bounce_attributes.append((initial_speed, total_time,
                                             individual_time, time_shift))

        if i != no_of_bounces:
            total_time += individual_time

        initial_speed *= efficiency**0.5

        if answer_boolean:
            print(f"The time elapsed during bounce {i+1} is:\
                 {individual_time:.2f}s")

    return total_time, individual_bounce_attributes


def displacement_equation(initial_speed: float,
                          time: Union[float, np.ndarray],
                          acceleration_const: float,
                          shift: float,
                          intercept: float=0) -> Union[float, np.ndarray]:
    """
    This function is used in the plots of the individual bounces. It is a
    modified version of the following SUVAT equation:

                    s = ut + (1/2)at^2

    In this function, 't' is replaced with '(t - shift)', and an intercept
    variable is introduced. These modifications facilitate the accurate
    plotting of the bounces. There is a negative between the 't' and 't^2'
    terms because the acceleration acts in the negative direction (downwards).

    Parameters:
        initial_speed
        time: Time interval of the bounce.
        acceleration_const
        shift: The bounce curves are calculated as if they are centred on
                't=0', so the shift allows them to be plotted at their
                actual 't' values.
        intercept: Required for the plotting of the first drop.

    Returns:
        Displacement values
    """
    return (initial_speed) * (time - shift) \
            - (0.5 * acceleration_const * (time - shift)**2) \
            + intercept


def plot_ball_displacement(total_time: float,
                           bounce_attributes: list,
                           grav_const: float,
                           initial_height: float,
                           min_height: float) -> None:
    """
    This function uses previously calculated values to plot the displacement
    of the bouncing ball against time. The list 'bounce_attributes' contains
    tuples that carry important kinematic values for each bounce. The
    'displacement_equation' function is called to calculate the displacement
    at each stage.

    Parameters:
        total_time
        bounce_attributes: List of tuples containing kinematic attributes of
                           each bounce.
        grav_const
        initial_height
        min_height
    """
    fig = plt.figure(1)
    axes = fig.add_subplot(111)

    for i, (initial_speed, running_total_time,
            individual_time, time_shift) in enumerate(bounce_attributes):

        time_domain = np.linspace(running_total_time,
                                  running_total_time + individual_time,
                                  int(10 * individual_time))
        displacement_range = displacement_equation(initial_speed,
                                                   time_domain,
                                                   grav_const,
                                                   time_shift,
                                            initial_height if i == 0 else 0.)
        if i == len(bounce_attributes) - 1:
            axes.plot(time_domain, displacement_range, color='red',
                      label=f"Bounce n = {len(bounce_attributes) - 1}, " +
                                "first below " + r"$h_{min}$")
        else:
            axes.plot(time_domain, displacement_range, color='blue')


    time_domain = np.linspace(0, total_time + bounce_attributes[-1][2],
                            int(10 * (total_time + bounce_attributes[-1][2])))
    axes.plot(time_domain, [min_height] * time_domain.shape[0],
              dashes=[6, 3], color='black',
              label=r"$h_{min}$: " + f"{min_height:.2f} m")


    height_domain = np.linspace(0, initial_height, int(10 * initial_height))
    axes.plot([total_time] * height_domain.shape[0], height_domain,
              dashes=[1, 2], color='black',
              label=r"$t_{total}$: " + f"{total_time:.2f} s")


    axes.set_title("Height bounced against time")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Height bounced (m)")

    plt.margins(0)
    plt.legend()
    plt.savefig("ball_displacement_graph.png", dpi=300)
    plt.show()


def main():
    """
    Main function that executes all the calculations.
    """
    print("This script calculates\n\
           a) The number of bounces over the min. height\n\
           b) Time taken to complete the bounces\n\
           given a set of initial parameters.\n")

    initial_height, min_height,\
    efficiency, grav_const, plot_path_bool = user_input_parameters()

    no_of_bounces = number_of_bounces(initial_height, min_height, efficiency)

    first_drop_time, first_drop_speed = first_drop_outputs(initial_height,
                                                           grav_const)
    if no_of_bounces == 0:
        print(f"\nGiven the input parameters, the ball cannot bounce over "\
        f"the minimum height.\nThe drop time is: {first_drop_time:.2f}s")
        sys.exit(0)

    total_time, individual_bounce_attributes = calculate_total_time(
                                                        no_of_bounces,
                                                        first_drop_time,
                                                        first_drop_speed,
                                                        efficiency,
                                                        grav_const)

    if plot_path_bool:
        plot_ball_displacement(total_time, individual_bounce_attributes,
                               grav_const, initial_height, min_height)

    print(f"\nNumber of bounces above the minimum height is: {no_of_bounces}")
    print("Total time of all bounces above the min height is: "\
          f"{total_time:.2f}s\n")


if __name__ == "__main__":
    main()
