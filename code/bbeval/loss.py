import torch.nn as nn
import torch as ch

class Loss:
    def __init__(self, name, reduction='mean'):
        self.name = name
        self.reduction = reduction

    def __call__(self, preds, label, is_targeted, **kwargs):
        raise NotImplementedError(f"Loss class {self.name} must implement __call__")


class MarginLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("margin_loss", reduction)
        # self.loss_obj = nn.MultiLabelMarginLoss(reduction=reduction)
    def loss_obj(self,preds,labels_one_hot):
        preds_correct_class = (preds * labels_one_hot).sum(1, keepdim=True)
        diff = preds_correct_class - preds  # difference between the correct class and all other classes
        labels_ = ch.argmax(labels_one_hot,dim=1)
        diff[ch.arange(diff.size()[0]),labels_] = ch.tensor(float("Inf"))
        # diff[labels_one_hot] = ch.tensor(float("Inf"))  # to exclude zeros coming from f_correct - f_correct
        margins,_ = diff.min(1)
        return margins

    def __call__(self, preds, labels, is_targeted=False, **kwargs):
        if preds.shape != labels.shape:
            # Convert labels to one-hot
            labels_ = nn.functional.one_hot(labels, preds.shape[1])
        else:
            labels_ = labels
        margins = self.loss_obj(preds,labels_)
        if is_targeted:
            return margins * (-1)
        else:
            return margins
        # if is_targeted:
        #     return -self.loss_obj(preds, label_)
        # else:
        #     return self.loss_obj(preds, label_)

class CrossEntropyLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("cross_entropy_loss", reduction)
        self.loss_obj = nn.CrossEntropyLoss(reduction=reduction)
    
    def __call__(self, preds, label, is_targeted=False, **kwargs):
        if is_targeted:
            return -self.loss_obj(preds, label)
        else:
            return self.loss_obj(preds, label)

# need to be edited
class LogitLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("logit_loss", reduction)

    def __call__(self, preds, label, is_targeted=False, **kwargs):
        real = preds.gather(1, label.unsqueeze(1)).squeeze(1)
        logit_dists = (1 * real)
        loss = logit_dists.sum()
        if is_targeted:
            return -loss
        else:
            return loss

class PoTripLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("po_trip_loss", reduction)

    def Poincare_dis(self, a, b):
        L2_a = ch.sum(ch.square(a), 1)
        L2_b = ch.sum(ch.square(b), 1)

        theta = 2 * ch.sum(ch.square(a - b), 1) / ((1 - L2_a) * (1 - L2_b))
        distance = ch.mean(ch.acosh(1.0 + theta))
        return distance

    def Cos_dis(self, a, b):
        a_b = ch.abs(ch.sum(ch.multiply(a, b), 1))
        L2_a = ch.sum(ch.square(a), 1)
        L2_b = ch.sum(ch.square(b), 1)
        distance = ch.mean(a_b / ch.sqrt(L2_a * L2_b))
        return distance

    def __call__(self, preds, label, is_targeted=False, **kwargs):
        labels_ = nn.functional.one_hot(label, preds.shape[1])
        loss_po = self.Poincare_dis((preds / ch.sum(ch.abs(preds), 1, keepdim=True)),ch.clamp((labels_ - 0.00001), 0.0, 1.0))
        loss_cos = ch.clamp(self.Cos_dis(labels_, preds) - self.Cos_dis(labels_, preds) + 0.007, 0.0, 2.1)
        loss=loss_po + 0.01 * loss_cos
        if is_targeted:
            return -loss
        else:
            return loss

class BCEWithLogitsLossWrapper(Loss):
    def __init__(self, reduction='mean'):
        super().__init__("bce_with_logits_loss", reduction)
        self.loss_obj = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def __call__(self, preds, label, is_targeted=False, **kwargs):
        labels_ = nn.functional.one_hot(label, preds.shape[1])
        print(labels_true_onehot==labels_)
        if is_targeted:
            return -self.loss_obj(preds, labels_.float())
        else:
            return self.loss_obj(preds, labels_.float())

_LOSS_FUNCTION_MAPPING = {
    "margin": MarginLossWrapper,
    "ce": CrossEntropyLossWrapper,
    "bce": BCEWithLogitsLossWrapper,
    "logit": LogitLossWrapper,
    "potrip":PoTripLossWrapper
}


def get_loss_fn(loss_name: str, reduction: str='mean'):
    wrapper = _LOSS_FUNCTION_MAPPING.get(loss_name, None)
    if not wrapper:
        raise NotImplementedError(f"Loss function {loss_name} not implemented")
    return wrapper(reduction)
