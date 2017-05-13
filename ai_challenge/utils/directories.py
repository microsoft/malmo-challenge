import os
from ai_challenge import ROOT_DIR


def check_file(path, file):
    if not os.path.isfile(os.path.join(path, file)):
        raise ValueError('File {} not found in {}'.format(file, path))


def check_create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


__missions_dir = os.path.join(ROOT_DIR, 'tasks/missions')
__results_dir = os.path.join(ROOT_DIR, 'results')
__config_dir = os.path.join(ROOT_DIR, 'config')


def get_mission_path(mission_nm):
    check_file(__missions_dir, mission_nm)
    return os.path.join(__missions_dir, mission_nm)


def get_results_path():
    check_create_dir(__results_dir)
    return __results_dir


def get_config_dir():
    if not os.path.isdir(__config_dir):
        raise ValueError('Config directory {} not found'.format(__config_dir))
    return __config_dir
