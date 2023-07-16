'''Functional version of the settings abstractor. Makes testing much easier.
Room jsons are imported and the script abstracts the common settings to a 'site.json'.
There are 3 potential cases that contribute to a difference beetween two room jsons:
        
        Case 1: A setting's value is different.
        Case 2: The reference json has a setting the other json doesn't have.
        Case 3: The other json has a setting the reference json doesn't have.

The 'common json generator' function accounts for all three cases.
'''
import json
import argparse
import glob

from deepdiff import DeepDiff

from oxeeval.tools.file_utils import format_json_string


def abstract_settings(home_path: str, common_room_name: str, destination_path: str = None,
                      excluded_rooms: list = []) -> None:
    '''
    Parameters:
    ----------
    home path (str): Directory path that points to the directory containing the json files.
                        e.g '/home/madam/stash/al-varberg-careh2/ee-room-room08.json

    common room name (str): Room name shared by all the rooms. Prefixes their number ID.
                            e.g. 'ee-room-room'.

    destination path (str): OPTIONAL argument that contains path pointing to the desired
                            destination directory for the output json files.
                            DEFAULTS to the home path.

    excluded rooms (list): OPTIONAL argument containing any rooms the user wishes to be
                            excluded from the settings abstraction.
    '''
    if not destination_path:
        destination_path = home_path

    json_dictionary = construct_jsons_dict(f'{home_path}/{common_room_name}', excluded_rooms)

    site_json, common_keys = common_json_generator(json_dictionary)

    clean_and_write_jsons(json_dictionary, common_keys)

    save_json(site_json, destination_path + '/site.json')



def construct_jsons_dict(generic_json_path: str, excluded_rooms: list = []) -> dict:
    '''
    Parameters
    ----------
    generic json path (str): Generic path to all jsons in a directory. 
                             e.g. /home/madam/caludon/ee-room-room

    excluded rooms (list): A list of any rooms (in the form of their numbers) the user wants 
                           excluded. e.g. 12 23 45
    Returns
    -------
    jsons dict (dict): Dictionary containing all the json files as dict values, with their paths
                       as the keys.
    '''
    jsons_dict = {}
    all_paths = glob.glob(generic_json_path + '*')

    finalised_paths = [path for path in all_paths 
                       if path.strip(generic_json_path).strip('.json') not in excluded_rooms]

    for path in finalised_paths:
        open_json_file(path, jsons_dict)

    return jsons_dict


def common_json_generator(jsons_dict: dict) -> (dict, list):
    '''Every setting in the common json reference gets checked to see
    if they exist in the room jsons. If a setting does for the given room, 
    the loop continues because that setting is deemed 'common'.
    However, if the setting is not in the given room json, it is immediately
    'unique', regardless of other rooms, and therefore gets removed from the 
    common json. 

    The 'flatten dictionary' function is used to make comparisions much easier
    by dodging past the limitations caused by a nested dictionary.

    Returns
    -------
    common json reference (dict): A dictionary containing only common settings.

    flat common json (list): A 'flattened' form of the common json.
    '''
    common_json_reference = list(jsons_dict.values())[0]
    flat_common_json = flatten_dictionary(common_json_reference)

    for room_json in jsons_dict.values():
        flat_room = flatten_dictionary(room_json)
        
        for setting in flat_common_json.copy():
            
            if setting not in flat_room:
                flat_common_json.remove(setting)

    key_terminator(common_json_reference, flat_common_json, reverse=True)

    return common_json_reference, flat_common_json


def clean_and_write_jsons(jsons_dict: dict, common_keys: list) -> None:
    '''Function that cleans room jsons of any common settings, using the 
    flat common json as the 'target keys' list, then writes them to file.
    '''
    for path, room_json in jsons_dict.items():
        clean_json = key_terminator(room_json, common_keys)
        save_json(clean_json, path)
        


def flatten_dictionary(dictionary: dict) -> list:
    '''This function locates the settings at the end of a nested dictionary
    chain (non-dict values) and creates a list of tuples, each one containing 
    three elements:

            index 0: parent key of setting key
            index 1: setting key
            index 2: setting value

    These tuples are unique settings identifiers used in the dictionary comparisons.
    While it wouldn't break json protocol (the tuple would have to go as far back as
    the root key), it is practically impossible for our room jsons to contain 
    identical tuple combinations.
    '''
    flat_dict = []
    def recursion(dictionary: dict, parent_key: str = None):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                recursion(value, key)
            elif len(key) > 2:
                flat_dict.append(tuple((parent_key, key, value)))

    recursion(dictionary)
    return flat_dict


def key_terminator(target_json: dict, target_keys: list, reverse: bool = False) -> dict:
    '''Iterates over a nested dictionary and deletes any key that
    'target_keys' contains.
    Parameters
    ----------
    target json (dict): Dictionary you wish to erase of keys in target keys.
    
    target keys (list): List of tuples. Each tuple contains a unique parent key,
                        child key, value combination, that points to the setting
                        to be deleted.

    parent key (str): The parent key of a given child key during a specific point
                      in the loop.
    Returns
    -------
    target json (dict): The inputted dictionary, now clean of the target settings.
    '''
    def recursion(dictionary: dict, target_keys: list, parent_key: str = None):
        twin_json = dictionary.copy()
        for key, value in twin_json.items():
            if isinstance(value, dict):
                recursion(value, target_keys, key)
            elif reverse:
                if tuple((parent_key, key, value)) not in target_keys:
                    del dictionary[key]
            elif tuple((parent_key, key, value)) in target_keys:
                del dictionary[key]
        return dictionary
    
    cleaned_json = recursion(target_json, target_keys)
    return cleaned_json


def open_json_file(path: str, storage: dict) -> None:
    '''Generic function that opens a json and stores it in a dictionary.
    Its path becomes the key.
    '''
    with open(path) as file_p:
        storage[path] = json.load(file_p)


def main():
    '''If the abstractor is run through the command line, this function executes
    the required code to control the functionality.
    '''
    parser = argparse.ArgumentParser(description='Tool that abstracts common settings from\
                                     multiple jsons to a higher level, e.g. from room to location.')
    parser.add_argument('-p', '--parent', type=str, required=True,
                        help='Parent directory for all the jsons. e.g. /home/madam/Desktop')
    parser.add_argument('-d', '--destination', type=str,
                        help='Destination json for common settings. e.g. /home/madam/location_settings\
                             \n DEFAULT: inputted parent directory')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Generic room name for all room jsons in a directory. e.g. ee-room-cell')
    parser.add_argument('-ex', '--exclude', type=str, nargs='*', default=[],
                        help='Excludes any rooms deemed irrelevant to the settings inquisition.\
                              Only number required. e.g. 03 or 34')
    args = parser.parse_args()

    abstract_settings(args.parent, args.name, args.destination, args.exclude)


if __name__ == '__main__':
    main()
