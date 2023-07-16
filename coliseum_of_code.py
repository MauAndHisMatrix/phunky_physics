# New code ideas are tested here, pitted against each other like gladiators...

####################          THE COLISEUM OF CODE          ###################

import json
import re
from typing import Union, Type
from pathlib import Path
from collections import namedtuple
from random import randint
import argparse
from functools import total_ordering, reduce
import operator

import numpy as np

from oxeeval.tools.file_utils import save_json

test_list = [1,1,2,2,3,3,3,3,4,5,'boy','boy','girl']

# new_list = list(set(test_list))

# print(new_list)

# with open('requirements_test.txt', 'r') as r:
#     requirements = r.read().splitlines()

# print(requirements)

path = str("/home/madam/preposterous_python_projects/pr_dict_test.json")

test_dict = {'string_heavy': {'string_list': ["a", "b", "s", "f", "c", "b", "a", "b"], 'mix_list': [1, 2, 3.4, 5.7773, True, "yes"]}, 
                              'number_heavy': {'number_list': [1, 333, 45, 23,22,6,3.4]}}

# save_json(test_dict, path, False)

path_string = "/home/madam/prepostet/room.json"

path_string = path_string.strip(".json")

# print(path_string)

tree_list = [
            "<root['stage_settings']['sunrise_sunset_stage']['longitude_positive_east']>", 
            "<root['settings']['sunrise_stage']['longitude_east']>",
            "<root['stage_settings_w']['sunrise_stage_e']['longitude_positive_east_y']>"
            ]

mini_tree_list = ["<root['settings']['sunrise_stage']['longitude_east']>"]

new_tree = list(set(tree_list) - set(mini_tree_list))
# print(new_tree)


def extract_key(dirty_key):
    stripped_key = re.search(r".+\'(\w+)", dirty_key)
    return stripped_key.group(1)

clean_key = extract_key(tree_list[0])
new_list = [extract_key(dirty_key) for dirty_key in tree_list]

# print(tree_list)
# print(new_list)
# print(clean_key)

test_dict.update({
    'new_key': 'yes', 
    'another': 'no'
})

test_dict_keys = list(test_dict.keys())

# print(test_dict[0])
# print(test_dict[test_dict_keys[0]])

path_ = '/boy/girl/dude.json'
gen_path = '/boy/girl'

# new_path = path_.strip(gen_path).strip('.json')
# print(new_path)

# print(list(test_dict.keys()), type(test_dict.keys()))
# print(test_dict.values(), type(test_dict.values()))


Data = namedtuple('Data', ['room', 'room_data', 'section'])

r = Data(200, {'persons': 3, 'AD': True}, {'persons': 1})

# print(r)
def check(number):
    return number > 5

# if check(3):
#     print('object held')
# else:
#     print('not held')

# if obj:
#     print("yes")
# else:
#     print("no")

test_l = [6, 2, 3]
test_2 = test_l[::-1]
test_l.reverse()

# print(test_l)
# print(test_2)

def if_check(object_: int):
    if object_ in {2, 3}:
        print('it worked')
    else:
        print("It didn't work")

dict_test = dict.fromkeys(['ssdfs', '23r23'])
# for v in dict_test.values():
    # if v:
        # print('bad news')
    # elif not v:
        # print("good, doesn't exist")

#-------------------------------------------------------------------------------

class ContextManager:
    def __init__(self):
        self.dict = {'a': [],
                     'b': [],
                     'c': {},
                     'd': {}}
    
    def __enter__(self):
        return self.dict

    def __exit__(self, exc_type, exc_val, traceback):
        save_json(self.dict, './test_dict.json')

class DummyClass:
    def __init__(self):
        with ContextManager() as self.dict:
            self.run()
        print('cm closed (hopefully)')

    def run(self):
        print('running code')
        self.print_()
        self.dict['a'].append('this is working')
        self.dict['c']['i'] = 42

        # print('cm closed (hopefully)')

    def print_(self):
        print(self.dict.keys())

#----------------------------------------------------------------------------------

sites = ['ard', 'pet', 'bm']
dict_check = {}

def key_filler(keys: list, dic: dict):
    for key in keys:
        if key not in dic:
            dic[key] = {}
        else:
            print('winner winner chicken dinner')
    return dic

dict_2 = dict.fromkeys(sites)
# print(dict_2)


key_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
arg_list = {'man': True, 'woman': False}

def arg_check(n1, n2, n3, n4, another_arg, and_another):
    print('args are {} {} {} {}'.format(n1, n2, n3, n4))
    total = n1 + n2 + n3 + n4
    print(f'Total is {total}, other args are: {another_arg} and {and_another}')

def arg_generator():
    return randint(1, 9), randint(1, 9), randint(1, 9), randint(1, 9)

def comp_arg():
    return [randint(1, 9) for x in range(4)]

#----------------------------------------------------------------------------------

def roman_to_int(roman: str) -> int:
    numerals = {'I': 1,
                'V': 5,
                'X': 10,
                'L': 50,
                'C': 100,
                'D': 500,
                'M': 1000}
    number_list = [numerals[x] for x in list(roman)]
    total = int(0)
    for i, n in enumerate(number_list):
        if i+1 < len(number_list) and n < number_list[i+1]:
            total -= n
        else:
            total += n
    return total

#-----------------------------------------------------------------------------------

array = [1,1,1,1,1,1,1,0,0,1,1,0,1,1]

def count_passing_cars(array: list) -> int:
    running_total = 0
    total = 0
    for x in reversed(array):
        running_total += x
        if x == 0:
            total += running_total
            if total > 1e9:
                return -1
    return total

#-----------------------------------------------------------------------------------

def zip_check(*iterables):
    sentinel = object()
    iterators = [iter(ite) for ite in iterables]
    while iterators:
        output = []
        for iterator in iterators:
            element = next(iterator, sentinel)
            if element is sentinel:
                return
            output.append(element)
        yield tuple(output)

def gen_output(number):
    for x in range(1, 4):
        yield x * number

gen_object = gen_output(2)

#-------------------------------------------------------------------------------------

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ArgTester:
    def __init__(self, **kwargs):
        self.args = kwargs.get('args', self.parse_args())

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--site', type=str)
        parser.add_argument('-r', '--room', type=str)
        
        return parser.parse_args()

    def run(self):
        site = self.args.site
        room = self.args.room
        s = int(site[-2:]) if site[-2] != '0' else int(site[-1])
        r = int(room[-2:]) if room[-2] != '0' else int(room[-1])

        print(s, r, s * r)

args = {'site': 'site02', 'room': 'room10'}
args = Namespace(site=args['site'], room=args['room'])

# print(args.site)

at = ArgTester(args=args)
# at = ArgTester()
# at.run()

#-----------------------------------------------------------------------------------

class Parent:
    def __init__(self):
        print('Parent class')
        self.gender = 'male'

    def scold(self):
        print(f"I'm Monica and I'm scolding you!")

    def praise(self):
        print(f"I'm Ross and I'm praising you!")

class Son(Parent):
    def __init__(self):
        Parent.__init__(self)
        self.name = 'Ross'
        self.gender = 'apache'

    def gender_reveal(self):
        print(self.gender)

class Daughter(Parent):
    def __init__(self):
        self.name = 'Monica'

# ross = Son()
# ross.praise()
# print(ross.gender)
# print(ross.name)

#------------------------------------------------------------------------------------

coods = {x: y for x, y in zip('abcdefgh', '12345678')}

c1 = ['a', '1']
c2 = 'b'
c_list = [c2]

unsorted_list = [1, 23, 54, 43, 32, 67]
sorted_list = sorted(unsorted_list)

@total_ordering
class IntWrapper:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other

    def __ge__(self, other):
        return self.value >= other

intw = IntWrapper(3)

#-------------------------------------------------------------------------------------

import pandas as pd

time = pd.Period('2020-06-05', 'H')

#-------------------------------------------------------------------------------------

ranks = [str(n) for n in range(2, 11)] + list('JQKA')
suits = 'Spades Clubs Hearts Diamonds'.split()

Card = namedtuple('Card', ['rank', 'suit'])

deck = [Card(rank=r, suit=s) for s in suits for r in ranks]

# Assigns slice objects to each suit for readibility
ACES = slice(12, None, 13)
JACKS = slice(9, None, 13)
TWOS = slice(0, None, 13)

aces = deck[ACES]
jacks = deck[JACKS]
twos = deck[TWOS]

# BUILT-INs ---------------------------------------------------------------------------

built = [1, 2, -3, 4, -5, False, 0, 3]

# returns tuple -> (x//y, x%y)
result = divmod(10, 3)

# absolutes every value
abs_built = [abs(x) for x in built]

# returns TRUE if any of the elements in the iterable are true
bool1 = any(built)

# returns TRUE if all elements of the iterable are true
bool2 = all(built)

# returns a boolean based on whether or not the object is an instance of the built-in type
bool3 = isinstance(built, list)

# TRUE if object is a member of the 'parent' class
bool4 = issubclass(list, object)

# returns an iterator object. If 'sentinel' is given, the iterator will CALL object until the value is equivalent.
it = iter(built)

# calls the iterator's __next__() method. Returns the default if StopIteration is raised.
element = next(it, 'end of iterator')

# min and max return the smallest or largest items in the given iterable, or selection of args
target = min(built, default=0, key=lambda x: abs(x))

# returns equivalent of x**y (with extra option of modulo %)
big = pow(7, 2, 8)

#-------------------------------------------------------------------------------------

def return_two(n):
    return (1, 2*n), (1, 3*n)

first = return_two(2)

#-------------------------------------------------------------------------------------

def mul_with_operator(iterable):
    return reduce(operator.mul, iterable)

def mul_with_lambda(iterable):
    return reduce(lambda x, y: x*y, iterable)

rng = np.random.default_rng()
it = rng.integers(1, 7, size=6)

squashed = mul_with_operator(it)

#--------------------------------------------------------------------------------------


formatter = ','.join(['%s']*10)

#--------------------------------------------------------------------------------------

def is_square(n):
    i = 1
    while True:
        sq = i**2
        if sq > n:
            return False
        elif sq == n:
            return True
        else:
            i += 1

#----------------------------------------------------------------------------------------

sample = "2 4 7 8 10"

def iq_test(nums):
    ints = {int(x) for x in nums.split()}
    odds = {i for i, n in enumerate(ints) if n % 2 != 0}
    evens = [i for i, n in enumerate(ints) if n % 2 == 0]
    if len(odds) == 1:
        return odds[0] + 1
    else:
        return evens[0] + 1
        
def iq_test2(nums):
    ints = [int(x) for x in nums.split()]
    odds = [n for n in ints if n % 2 != 0]
    evens = list(set(ints) - set(odds))
    if len(odds) == 1:
        return ints.index(odds[0]) + 1
    else:
        return ints.index(evens[0]) + 1

#-------------------------------------------------------------------------------------------

arr = np.array()

#----------------------------------------------------------------------------------------


grid = np.array([1,2,3,4,5])