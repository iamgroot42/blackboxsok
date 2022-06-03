import torch.nn as nn


class Loss:
    def __init__(self, name):
        self.name = name

    def __call__(self, label, preds, is_targeted, **kwargs):
        raise NotImplementedError(f"Loss class {self.name} must implement __call__")


class MarginLossWrapper(Loss):
    def __init__(self):
        super().__init__("margin_loss")
        self.loss_obj = nn.MultiLabelMarginLoss()
    
    def __call__(self, label, preds, is_targeted, **kwargs):
        if preds.shape != label.shape:
            # Convert labels to one-hot
            label_ = nn.functional.one_hot(label, preds.shape[1])
        else:
            label_ = label
        return self.loss_obj(preds, label_)


class CrossEntropyLossWrapper(Loss):
    def __init__(self):
        super().__init__("cross_entropy_loss")
        self.loss_obj = nn.CrossEntropyLoss()
    
    def __call__(self, label, preds, is_targeted, **kwargs):
        return self.loss_obj(preds, label)


class BCEWithLogitsLossWrapper(Loss):
    def __init__(self):
        super().__init__("bce_with_logits_loss")
        self.loss_obj = nn.BCEWithLogitsLoss()
    
    def __call__(self, label, preds, is_targeted, **kwargs):
        return self.loss_obj(preds, label)


_LOSS_FUNCTION_MAPPING = {
    "margin": MarginLossWrapper,
    "ce": CrossEntropyLossWrapper,
    "bce": BCEWithLogitsLossWrapper
}


def get_loss_fn(loss_name: str):
    wrapper = _LOSS_FUNCTION_MAPPING.get(loss_name, None)
    if not wrapper:
        raise NotImplementedError(f"Loss function {loss_name} not implemented")
    return wrapper()
