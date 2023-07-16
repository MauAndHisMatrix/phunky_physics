# -*- coding: utf-8 -*-
"""
PHYS20161 Week 6 writing file example

Writes a 'hello world' file.

Lloyd Cawthorne 14/10/20

"""

my_file = open('my_file.txt', 'w')

my_file.write('Hello World! \n')

my_file.close()


print('This prints Hello World too!', file=my_file)