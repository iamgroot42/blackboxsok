import numpy as np
import torch as ch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


class Loss:
    def __init__(self, name):
        self.name = name

    def __call__(self, label, preds, is_targeted, **kwargs):
        raise NotImplementedError(f"Loss class {self.name} must implement __call__")


class MarginLoss:
    def __init__(self):
        super().__init__("margin_loss")
    
    def __call__(self, label, preds, is_targeted, **kwargs):
        # TODO: Implement
        pass


_LOSS_FUNCTION_MAPPING = {
    "margin": MarginLoss,
    "ce": CrossEntropyLoss,
    "bce": BCEWithLogitsLoss
}


def get_loss_fn(loss_name: str):
    wrapper = _LOSS_FUNCTION_MAPPING.get(loss_name, None)
    if not wrapper:
        raise NotImplementedError(f"Loss function {loss_name} not implemented")
    return wrapper
