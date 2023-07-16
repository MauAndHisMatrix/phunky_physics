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

Last updated: 21/10/21
Author: Marawan Adam UID: 10458577
"""

# Standard Library
import sys

# Third Party
import numpy as np

# Constants
SOLAR_SYSTEM = {"mercury": 3.70, "venus": 8.87, "earth": 9.81, "mars": 3.71,
                "jupiter": 24.92, "saturn": 10.44, "neptune": 8.87,
                "uranus": 11.75, "pluto": 0.49}


def user_input_parameters() -> tuple:
    """
    This function receives the user's inputs and performs validation
    checks on them. Once the checks are successful, the input loop is broken.

    Returns:
        initial_height
        min_height
        efficiency: Efficiency factor.
        grav_const: Chosen planet dictates the gravitational constant to be
                    used in calculation.
    """
    while True:
        try:
            initial_height = float(input("Enter the initial height:"))

            min_height = float(input("Enter minimum height:"))

            if initial_height <= 0 or min_height <= 0:
                print("\nThe entered heights must be higher than 0.")
                raise ValueError

            if min_height >= initial_height:
                print("\nThe minimum height must be lower than the"\
                      " initial height.")
                raise ValueError

            efficiency = float(input("Enter the Efficiency factor:"))

            if not 0 < efficiency < 1:
                print("\nThe efficiency must be exclusively between 0 and 1.")
                raise ValueError
            if efficiency > 0.9999999:
                print("\nEfficiency is too close to 1, which will yield a"\
                    " processor-melting number of bounces!")
                raise ValueError

            print(f"\nSolar System planets: {list(SOLAR_SYSTEM.keys())}")
            planet = input("Enter your planet of choice from the "\
                               "solar system. DEFAULT=earth:")
            grav_const = SOLAR_SYSTEM.get(planet, 9.81)

            return initial_height, min_height, efficiency, grav_const

        except ValueError:
            print("The value(s) entered is invalid, try again.\n")


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


def total_time_calculation(no_of_bounces: int, first_drop_time: float,
    first_drop_speed: float, efficiency: float, grav_const: float) -> float:
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
    """
    total_time = first_drop_time
    initial_speed = first_drop_speed * efficiency**0.5

    answer_boolean = False
    # 10 seems a reasonable limit for displaying individual times.
    if no_of_bounces <= 10:
        answer = input("\nWould you like the times of each bounce? [y/n]")
        answer_boolean = answer.lower() in ["y", "yes"]

    for i in range(no_of_bounces):
        individual_time = bounce_time(initial_speed, grav_const)
        total_time += individual_time

        initial_speed *= efficiency**0.5

        if answer_boolean:
            print(f"The time elapsed during bounce {i+1} is:\
                 {individual_time:.2f}s")

    return total_time


def main():
    """
    Main function that executes all the calculations.
    """
    print("This script calculates\n\
           a) The number of bounces over the min. height\n\
           b) Time taken to complete the bounces\n\
           given a set of initial parameters.\n")

    initial_height, min_height,\
    efficiency, grav_const = user_input_parameters()

    no_of_bounces = number_of_bounces(initial_height, min_height, efficiency)

    first_drop_time, first_drop_speed = first_drop_outputs(initial_height,
                                                           grav_const)

    if no_of_bounces == 0:
        print(f"\nGiven the input parameters, the ball cannot bounce over"\
        f"the minimum height.\nThe drop time is: {first_drop_time:.2f}s")
        sys.exit(0)

    total_time = total_time_calculation(no_of_bounces, first_drop_time,
                                    first_drop_speed, efficiency, grav_const)

    print(f"\nNumber of bounces above the minimum height is: {no_of_bounces}")
    print("Total time of all bounces above the min height is: "\
          f"{total_time:.2f}s")


if __name__ == "__main__":
    main()
