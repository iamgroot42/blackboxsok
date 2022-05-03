import torch as ch
import torch.nn as nn
import numpy as np

from bbeval.models.core import GenericModelWrapper
from bbeval.utils import AverageMeter
from code.bbeval.config.core import TrainConfig
from tqdm import tqdm


class PyTorchModelWrapper(GenericModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
    
    def set_train(self):
        self.model.train()
    
    def set_eval(self):
        self.model.eval()
    
    def forward(self, x):
        return self.model(x)
    
    def get_top_k_probabilities(self, x, k) -> np.ndarray:
        predictions = self.forward(x)
        top_k_probs = ch.topk(predictions, k, dim=1)[0]
        return top_k_probs.detach().numpy()

    def get_predicted_class(self, x) -> int:
        predictions = self.forward(x)
        return ch.argmax(predictions, dim=1).detach().numpy()[0]

    def train(self, train_loader, val_loader, **kwargs):
        # TODO: Implement

    def eval(self, loader, loss_function, acc_fn, **kwargs):
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        # Set model to eval model
        self.set_eval()
        # Do not use 'ch.no_grad()' because we may want to
        # keep the gradients in some attacks
        iterator = tqdm(loader)
        for x, y in iterator():
            y_predicted = self.forward(x)
            loss = loss_function(y_predicted, y)
            accuracy = acc_fn(y_predicted, y)
            loss_tracker.update(loss, x.size(0))
            acc_tracker.update(accuracy, x.size(0))
            iterator.set_description("Loss: {:.4f}, Accuracy: {:.4f}".format(
                loss_tracker.avg, acc_tracker.avg))
        return loss_tracker.avg, acc_tracker.avg
