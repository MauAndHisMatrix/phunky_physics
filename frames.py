#!/usr/bin/python3

''' Generates reference images for given site_name and rooms (optional). Rooms can
    be given as either room_names or room_ids, the latter requiring the '-i' flag. '''

# Standard Library
import sys
import os
import argparse
from typing import Union, List, Iterable
from pathlib import Path
from itertools import groupby
import logging

# Third Party
import cv2

# Oxehealth
from oxeeval.data_access.read_oxevid import PythonOxevidReader
from oxeeval.vault_access.oxe_database_connection import DB
from oxeeval.oxe_logging.logger import log_newline
from oxeeval.tools.file_utils import find_oxevid, query_yes_no
from oxeeval import OXEHEALTH_DRIVE_MOUNT, REF_IMAGE_STAGING

logger = logging.getLogger('oxeeval')


def get_room_ids(c, site_name: str, rooms: Union[str, List[str]] = None) -> list:
    """Parameters
       ----------
            site_name: e.g. mh-epuft-hadleigh
            rooms: (optional) either a single room_name, or a list of them.
    Returns
    -------
            room_ids
    """
    if rooms:
        rooms = tuple(rooms)
        param_formatter = ','.join(['%s']*len(rooms))
        c.execute(f"""SELECT r.room_id FROM rdb2.sites as s
                      LEFT JOIN rdb2.rooms as r ON s.site_id = r.site_id
                      WHERE s.site_name = %s AND room_name in ({param_formatter})
                      AND r.secure_flag = 0""", (site_name, *rooms))
    else:
        c.execute("""SELECT r.room_id FROM rdb2.sites as s
                     LEFT JOIN rdb2.rooms as r ON s.site_id = r.site_id
                     WHERE s.site_name = %s AND r.secure_flag = 0""", (site_name,))

    room_ids = [r[0] for r in c.fetchall()]
    
    return room_ids


def get_cameras_and_paths(c, room_ids: List[int]) -> List[tuple]:
    """Returns a list of tuples, each tuple containing information belonging
    to a specific clip.
    Parameters:
    ----------
        room_ids
    Returns:
    -------
        unique_cameras: A list of tuples, each containing the room_id, cam_id, path and start_ts
                        of a clip. These clips belong to the most recently-made camera, and are
                        the most recently made clips. They DO NOT contain PID. PID is filtered out.
    """

    param_formatter = ','.join(['%s']*len(room_ids))
    c.execute(f"""SELECT ca.room_id, c.camera_id, c.path, c.start_ts FROM rdb2.cameras as ca
                 LEFT JOIN rdb2.clips as c on ca.camera_id = c.camera_id
                 WHERE ca.room_id in ({param_formatter}) AND c.path not like '%private%' 
                                                         AND not c.clip_type_id = 5""", (*room_ids,))
    
    records = c.fetchall()

    # TODO Potentially worth dumping the db fetch into a dataframe and manipulating it then.
    # Or I could just robust the sql query.

    # Room records grouped by room_id, in preparation for filtering.
    grouped_recs = [[rec for rec in group] for room_id, group in groupby(records, lambda r: r[0])]

    newest_cameras = [[rec for rec in group if rec[1] == max(group, key=lambda r: r[1])[1]] for group in grouped_recs]
    new_cameras_latest_clip = [max(group, key=lambda r: r[3]) for group in newest_cameras]
    
    return new_cameras_latest_clip


def update_camera_in_db(c, camera: int, 
                           door_open_path: Union[str, Path], 
                           door_closed_path: Union[str, Path]) -> None:
    """
    Updates the given camera's reference image paths in the database.
    """
    op = door_open_path.strip(OXEHEALTH_DRIVE_MOUNT)
    cl = door_closed_path.strip(OXEHEALTH_DRIVE_MOUNT)

    c.execute("""UPDATE rdb2.cameras SET door_open_reference_image = %s,
                                         door_closed_reference_image = %s
                 WHERE camera_id = %s""", (op, cl, camera))


def parser_args():
    parser = argparse.ArgumentParser(description='''Generates reference images for given
                                    site_name and rooms (optional). Rooms can be given as 
                                    either room_names or room_ids, the latter requireing the '-i' flag.''')

    parser.add_argument('--site', '-s', type=str, required=True, help='''Site_name, e.g. hadleigh (can be abbreviated)''')
    parser.add_argument('--rooms', '-r', type=str, nargs='*', default=None, help='''List of room_names or room_ids''')
    parser.add_argument('--ids', '-i', action='store_true', help='''Simply write '-i' to activate. If done so,
                                                                    the script will read the rooms as room_ids.''')
    parser.add_argument('--checked', '-ch', action='store_true', help='''Once images in local dir have been checked,
                                                                         re-run with this flag, and images will write to
                                                                         the reference_image_staging_area''')
    parser.add_argument('--target_directory', '-td', type=str, help='''Before writing to the staging area, you must first write
                                                                 to a local dir, in order to check the images.''')
    return parser.parse_args()


def main():
    args = parser_args()
    
    if args.target_directory:
        target_path = args.target_directory
        if not os.path.isdir(target_path):
            raise NotADirectoryError('Specified dir does not exist.')
        
    elif args.checked:
        answer = query_yes_no('''You're about to write ref images to RS1 staging, would you like to continue?
                                 [yes/no]''')
        if answer == 'no':
            logger.critical('Aborting ref image generation.')
            sys.exit(0)
        target_path = REF_IMAGE_STAGING

    else:
        logger.critical('Arguments are invalid, you must specify a target dir or use the checked flag')
        sys.exit(0)
    
    oxevid_reader = PythonOxevidReader()

    with DB() as (db, _):
        if args.ids:
            room_ids = [int(r) for r in args.rooms]
        else:
            room_ids = get_room_ids(db, args.site, args.rooms)

        records = get_cameras_and_paths(db, room_ids)

        logger.info('<<Camera and path information for ALL rooms retrieved>>')
        log_newline(2)


        for i, (_, camera, rs1_path, _) in enumerate(records):
            if i != 0:
                log_newline(2)
            
            logger.info(f'Sourcing oxevid for camera_id: {camera}')

            clip_folder_path = os.path.join(OXEHEALTH_DRIVE_MOUNT, rs1_path.strip('/'))

            try:
                absolute_oxevid_path = find_oxevid(clip_folder_path)
            except FileNotFoundError:
                logger.error(f"Could not find oxevid for camera: {camera}")
                continue
            

            logger.info('Extracting and writing reference images')

            open_path = os.path.join(target_path, f'{camera}-door_open_reference_frame.png')
            closed_path = os.path.join(target_path, f'{camera}-door_shut_reference_frame.png')

            image, _ = oxevid_reader.GetFrame(absolute_oxevid_path, 1)

            cv2.imwrite(open_path, image)
            cv2.imwrite(closed_path, image)


            if args.checked:
                logger.info('Images written to rs1 reference frame staging area')

                update_camera_in_db(db, camera, open_path, closed_path)

                logger.info('RDB2 Camera table updated with location of ref images')

            else:
                DB.commit = False
                logger.info(f'Images written to {target_path}')
            

if __name__ == '__main__':
    main()