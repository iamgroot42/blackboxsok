import torch.hub as hub
from bbeval.utils import get_models_save_path

hub.set_dir(get_models_save_path())