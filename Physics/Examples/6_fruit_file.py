# -*- coding: utf-8 -*-
"""
PHYS20161 Week 7 reading file example

Prints the contents of fruit_list.txt

Lloyd Cawthorne 30/08/21

"""

fruit_file = open('fruit_list.txt', 'r')

list_of_fruits = []

for line in fruit_file:
    line = line.strip('\n')
    print(line)
    list_of_fruits.append(line)

fruit_file.close()

list_of_fruits.sort(reverse=True)

output_file = open('fruit_list_upper_case.txt', 'w')

print('% List of fruits in upper case.', file=output_file)

for fruit in list_of_fruits:
    print(fruit.upper(), file=output_file)

output_file.close()
