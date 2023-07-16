# -*- coding: utf-8 -*-
"""
PHYS20161 Week 12 example of class

Basic example of a class that prints Hello user

Lloyd Cawthorne 09/12/19

"""


class HelloUser:
    """
    Simple class that saves user name and has basic method"""

    def __init__(self, user='user'):
        """ Initialises class with default name 'user'"""

        self.user = user

    def hello(self):
        """Prints Hello user!"""

        print('Hello {:s}!'.format(self.user))


EXAMPLE_0 = HelloUser()

EXAMPLE_0.hello()

EXAMPLE_1 = HelloUser('World')

EXAMPLE_1.hello()
