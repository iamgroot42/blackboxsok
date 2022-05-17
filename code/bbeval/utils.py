# TODO: Read these from a config file instead of hardcoding in .py file


def get_log_save_path():
    """
        Path where results are stored
    """
    return "./log"


def get_models_save_path():
    """
        Path where models are stored
    """
    return "./models"


def get_datasets_path():
    """
        Path where datasets are available
    """
    return "./datasets"


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