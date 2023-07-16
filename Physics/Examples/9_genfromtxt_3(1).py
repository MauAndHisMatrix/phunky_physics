# -*- coding: utf-8 -*-
"""
PHYS20161 Week 10 Lecture quiz

Basic usage of numpy genfromtxt.

Lloyd Cawthorne 01/12/20
"""

import numpy as np

FILE_NAME = 'hello_world.txt'


def read_and_print_file(file_name=FILE_NAME):
    """
    Reads in data file of chars, joins each line and prints.

    Parameters
    ----------
    file_name : string, optional
        DESCRIPTION. The default is FILE_NAME.

    Returns
    -------
    None.

    """


    file_input = np.genfromtxt(file_name, delimiter=',', dtype='str')

    array = []

    for line in file_input:
        temp_string=''
        index = 0
        while index < len(line):
            temp_string = temp_string + str(line[index])
            index += 1

        array.append(temp_string)

    for string in array:
        print('{:s}'.format(string))

read_and_print_file()
