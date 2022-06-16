import torch.nn as nn


class Loss:
    def __init__(self, name, reduction='mean'):
        self.name = name
        self.reduction = reduction

    def __call__(self, preds, label, is_targeted, **kwargs):
        raise NotImplementedError(f"Loss class {self.name} must implement __call__")


class MarginLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("margin_loss", reduction)
        self.loss_obj = nn.MultiLabelMarginLoss(reduction=reduction)
    
    def __call__(self, preds, label, is_targeted=False, **kwargs):
        if preds.shape != label.shape:
            # Convert labels to one-hot
            label_ = nn.functional.one_hot(label, preds.shape[1])
        else:
            label_ = label
        if is_targeted:
            return -self.loss_obj(preds, label_)
        else:
            return self.loss_obj(preds, label_)

class CrossEntropyLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("cross_entropy_loss", reduction)
        self.loss_obj = nn.CrossEntropyLoss(reduction=reduction)
    
    def __call__(self, preds, label, is_targeted=False, **kwargs):
        if is_targeted:
            return self.loss_obj(preds, label)
        else:
            return -self.loss_obj(preds, label)

class BCEWithLogitsLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("bce_with_logits_loss", reduction)
        self.loss_obj = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def __call__(self, preds, label, is_targeted=False, **kwargs):
        if is_targeted:
            return self.loss_obj(preds, label)
        else:
            return -self.loss_obj(preds, label)

_LOSS_FUNCTION_MAPPING = {
    "margin": MarginLossWrapper,
    "ce": CrossEntropyLossWrapper,
    "bce": BCEWithLogitsLossWrapper
}


def get_loss_fn(loss_name: str, reduction: str='mean'):
    wrapper = _LOSS_FUNCTION_MAPPING.get(loss_name, None)
    if not wrapper:
        raise NotImplementedError(f"Loss function {loss_name} not implemented")
    return wrapper(reduction)
