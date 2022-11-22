# TODO: Read these from a config file instead of hardcoding in .py file
import os


def check_if_inside_cluster():
    """
        Check if current code is being run inside a cluster.
        Set this flag in your .bashrc for your Rivanna env
    """
    if os.environ.get('ISRIVANNA') == "1":
        return True
    return False


def get_log_save_path():
    """
        Path where results are stored
    """
    return "./log"


def get_models_save_path():
    """
        Path where models are stored
    """
    if check_if_inside_cluster():
        return "/project/uvasrg_paid/blackboxsok/models"
    return "/p/blackboxsok/models"


def get_dataset_dir_path():
    """
        Path where datasets are stored
    """
    if check_if_inside_cluster():
        return "/project/uvasrg_paid/blackboxsok/datasets"
    return "/p/blackboxsok/datasets"


def get_cache_dir_path():
    """
        Path where datasets are stored
    """
    if check_if_inside_cluster():
        return "/project/uvasrg_paid/blackboxsok/experiment/"
    return "/p/blackboxsok/experiment/"


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
