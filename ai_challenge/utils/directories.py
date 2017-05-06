import os
from ai_challenge import ROOT_DIR

def check_file(path, file):
    if not os.path.isfile(os.path.join(path, file)):
        raise ValueError('File {} not found in {}'.format(file, path))


__missions_dir = os.path.join(ROOT_DIR, 'tasks/missions')


def get_mission_path(mission_nm):
    # check_file(__missions_dir, mission_nm)
    return os.path.join(__missions_dir, mission_nm)
