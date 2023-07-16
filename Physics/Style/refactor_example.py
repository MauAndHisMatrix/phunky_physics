# -*- coding: utf-8 -*-
"""
PHYS20161 refactoring example

Illustrates to versions of the same function. In one case a refactor issue is
flagged due to unnecessary else after return.

Lloyd Cawthorne 13/02/20

"""


def function_1(number_1, number_2):
    """
    Prints messages based on which of the inputs is greater.
    This version generates a refactor warning.
    Args:
        number_1 (float)
        number_2 (float)
    """

    if number_1 > number_2:
        print('{0:.1f} is greater than {1:.1f}.'.format(number_1,
                                                        number_2))
        return None

    else:
        print('{0:.1f} is not greater than {1:.1f}.'.format(number_1,
                                                            number_2))
        return None


def function_2(number_1, number_2):
    """
    Prints messages based on which of the inputs is greater.
    This version does not generate a refactor warning.
    Args:
        number_1 (float)
        number_2 (float)
    """

    if number_1 > number_2:
        print('{0:.1f} is greater than {1:.1f}.'.format(number_1,
                                                        number_2))
        return None

    print('{0:.1f} is not greater than {1:.1f}.'.format(number_1,
                                                        number_2))
    return None
