import json
from deepdiff import DeepDiff    # **** DO I CALL IT ARTEMIS OR OPEN_SEASON?! ****
import pprint
import copy
keys = []

with open('/home/madam/pyramid_code/oxeeval/oxeeval/pyramid_code/room1.json') as q:      # **EDIT PATHS WHEN CHANGING COMPUTER**
    a = json.load(q)

with open('/home/madam/pyramid_code/oxeeval/oxeeval/pyramid_code/room2.json') as q:
    b = json.load(q)                  # 'with' command opens a file, anything indented
                                      # applies while open, closes automatically.

global_settings = a['global_settings']    # Like number indices in a list,
                                          # you can select a key using its name

ddiff = DeepDiff(a, b, ignore_order=True, verbose_level=0, view='tree')
# dict_ddiff = ddiff.to_dict()

# pprint.pprint(dict_ddiff)
# print(ddiff)

jsons = [a, b]
empirical = list(ddiff['values_changed'])                # has to be a list if I'm to abuse it.
# for i in empirical:
#     print(i)
# pprint.pprint(a)
# pprint.pprint(b)
hunted = a                                             # Allows me to hunt down the dirty objects in a separate JSON.

def strip_shell(target_key):                           # **EXFOLIATION: Function gets rid of the junk around the chosen 'prey'**
    length = len(target_key)
    start = int(0)                                     # start and end needed initialising to avoid errors.
    end = int(0)
    for count in range (1, length + 1):
        if (target_key[-count] == "]"):                # iterates from end, because logically, last [] item will be the target.
            end = length - count - 1                     # end is one character more that start because of how string manipulation works (inclusive).
        if (target_key[-count] == "["):
            start = length - count + 2                 # -1 and +2 get rid of the apostrophes and the square brackets. 
            break                                      # breaks loop because at this point, a [] item has been found.
    prey = target_key[start:end]                       # classic, schoolboy string manipulation - always a time and place.
    return(prey)

for count in range (0, len(empirical)):                # takes all the DeepDiff outputs and exfoliates them
    empirical[count] = str(empirical[count])
    keys.append(strip_shell(empirical[count]))         # they get put in a list as a reference later.
                                                             
def hunt(habitat):                                    # **CANNOT del key from dict, check this out...SOLVED: make a copy and iterate through that.
    jungle = habitat.copy()                          # Copy lets me iterate and delete without returning a 'dict changed size error'
    for key, value in jungle.items():
        if isinstance(value, dict):                 # iterate part where dictionary is cycled through till a real value is found (just reapplies function).
            hunt(value)
        if key in keys:                             # checks to see if in target prey list
            del habitat[key]                       # Neat thing this, habitat keeps changing, turning into each sub dictionary, till we only 
    return(habitat)                                # have to call the one key from this one sub-dictionary, had been biggest problem thus far.

fallen_beast = hunt(hunted)
pprint.pprint(fallen_beast)

with open('/home/madam/pyramid_code/oxeeval/oxeeval/pyramid_code/match_made_in_heaven.json', 'w') as fp:
    json.dump(fallen_beast, fp, indent=4)


# example = {"overlord": {'new value': 2, 'old value': 4}, 'turbo': True}
# scent = 'new_value'

# hunt = dict_ddiff | grep(scent, verbose_level=0)
# pprint.pprint(hunt)
