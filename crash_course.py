#########################           PYTHON CRASH COURSE          #########################

import math
import string
from collections import defaultdict


# METHODS AND FUNCTIONS ##################################################################

# 1: Volume of sphere
def vol(rad):
    pi = 3.14
    return 4 * pi * (rad**3)


#2: Number in range
def range_check(num, low, high):
    if low <= num <= high:
        print("Number is within range")
    else:
        print("Number is not within range")

def range_2(num, low, high):
    return low <= num <= high


#3: How many upper case / lower case letters are in sentence
s = 'Wow, you look cool Jeet'
def up_low(string):
    check_list = [0, 0]
    for char in string:
        if char.islower():
            check_list[0] += 1
        elif char.isupper():
            check_list[1] += 1
    print(f"Original string: {string}\nNumber of lower case letters: {check_list[0]}\n\
            Number of upper case letters: {check_list[1]}")


# 4: Unique elements in a list
def unique_elements(input_list):
    return list(set(input_list))


# 5: Multiply all numbers in a list
def multiply(input_list):
    product = float(1.0)
    for x in input_list:
        product *= x
    return product


# 6: Check for palindrome string
def palindrome_check(string):
    return string == string[::-1]


# 7: Sentence pangram check (all letters of the alphabet)
example = "The quick brown fox jumps over the lazy dog"
def is_pangram(string, choice_alphabet = string.ascii_lowercase):
    string = string.lower()
    case = True
    while case == True:
        for letter in choice_alphabet:
            case = letter in string
        break
    return case

# Funky, more efficient solution
def is_pangram_2(string, choice_alphabet = string.ascii_lowercase):
    return set(choice_alphabet) <= set(string.lower())
    

# OBJECT-ORIENTATED PROGRAMMING (OOP) ####################################################

class LineAttributes:

    def __init__(self, coo1, coo2):
        self.coo1 = coo1
        self.coo2 = coo2

    def distance(self):
        x_dist = self.coo1[0] - self.coo2[0]
        y_dist = self.coo1[1] - self.coo2[1]
        dist_sqaured = x_dist**2 + y_dist**2
        return math.sqrt(dist_sqaured)

    def slope(self):
        top = self.coo1[1] - self.coo2[1]
        bottom = self.coo1[0] - self.coo2[0]
        return top/bottom

c1 = (3,2)
c2 = (8,10)

line = LineAttributes(c1, c2)

# print(line.distance(), line.slope())

class CylinderChars:

    # Class object attribute
    pi = 3.14 

    def __init__(self, height=2, radius=2):
        self.height = height
        self.radius = radius
        self.cross_section = 0

    def volume(self):
        self.cross_section = CylinderChars.pi * self.radius**2
        return self.height * self.cross_section

    def surface_area(self):
        if self.cross_section != 0:
            circumference = CylinderChars.pi * 2 * self.radius
            sheet_area = self.height * circumference
            return sheet_area + 2 * self.cross_section
        else:
            # raise an error (must learn about this)
            print("cross-sectional area not generated")


c = CylinderChars(2, 3)

# print(c.volume(), c.surface_area())


# OOP Challenge - Bank Account ###########################################################

class BankAccount:

    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def __str__(self):
        return f"Account owner: {self.owner}\nAccount balance: {self.balance}"

    def withdraw(self, amount):
        if amount > self.balance:
            print(f"Debts will rise, empires will fall\nBalance remains {self.balance}")
        else:
            self.balance -= amount
            print(f"Successful withdrawal\nNew balance is {self.balance}")
    
    def deposit(self, amount):
        self.balance += amount
        print(f"Deposit Successful\nNew balance is {self.balance}")


ba = BankAccount('Mazlan', 2000)

# Output information of Bank Account
# print(ba)

# Output Attributes
# n
# print(f"Owner is {ba.owner} and balance is {ba.balance}")

# Deposits and Withdrawals
# ba.deposit(30)
# ba.withdraw(3000)
# ba.withdraw(200)


# NEAT BITS AND BOBS #####################################################################

empty_dict = dict.fromkeys(['section_finder', 'section'])

# print(empty_dict)

# DECORATORS ##############################################################################

def decorator_func_args(a1, a2):
    print("setting up decorator")

    def actual_decorator(present):
        print("decorating func")
        
        def wrapper_func(*args):
            print(f"Some args: {a1} {a2}")
            print("I am wrapping the present in wrapper paper!")
            present(*args)
            print("More wrapping paper!")

        return wrapper_func

    return actual_decorator

# class decorator_class:
#     def __init__(self, present):
#         self.p = present

#     def __call__(self):
#         print("Decorating pressie")
#         self.p()
#         print("All wrapped")


# @decorator_func_args('man', 'woman')
def present_func(arg, args):
    name = tuple((arg, args))
    print(f"    A tuple: {name}")

# print("End of decoration")

# present_func('a', 'b')

# ADVANCED MODULES #########################################################################

d = defaultdict()
