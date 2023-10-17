import os


DATA_PATH = os.environ.get("BLACKBOXSOK_DATA_PATH", None)
if DATA_PATH is None:
    raise ValueError("Please set the environment variable BLACKBOXSOK_DATA_PATH")
MODELS_PATH = os.environ.get("BLACKBOXSOK_MODELS_PATH", None)
if MODELS_PATH is None:
    raise ValueError("Please set the environment variable BLACKBOXSOK_MODELS_PATH")
CACHE_PATH = os.environ.get("BLACKBOXSOK_CACHE_PATH", None)
if CACHE_PATH is None:
    raise ValueError("Please set the environment variable BLACKBOXSOK_CACHE_PATH")


def get_log_save_path():
    """
        Path where results are stored
    """
    return "./log"


def get_models_save_path():
    """
        Path where models are stored
    """
    return MODELS_PATH


def get_dataset_dir_path():
    """
        Path where datasets are stored
    """
    return DATA_PATH


def get_cache_dir_path():
    """
        Path where datasets are stored
    """
    return CACHE_PATH


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
