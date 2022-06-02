from bbeval.config.core import ModelConfig
import torch as ch
import numpy as np

from bbeval.models.core import GenericModelWrapper
from bbeval.utils import AverageMeter
from bbeval.config import ModelConfig
from tqdm import tqdm


class PyTorchModelWrapper(GenericModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
    
    def set_train(self):
        self.model.train()
    
    def post_process_fn(self, tensor):
        return tensor
    
    def set_eval(self):
        self.model.eval()
    
    def cuda(self):
        self.model.cuda()
    
    def pre_process_fn(self, x):
        return x
    
    def zero_grad(self):
        self.model.zero_grad()
    
    def forward(self, x):
        x_ = self.pre_process_fn(x)
        return self.post_process_fn(self.model(x_))
    
    def get_top_k_probabilities(self, x, k) -> np.ndarray:
        predictions = self.forward(x)
        top_k_probs = ch.topk(predictions, k, dim=1)[0]
        return self.post_process_fn(top_k_probs)

    def predict(self, x) -> int:
        predictions = self.forward(x)
        # return self.post_process_fn(ch.argmax(predictions, dim=1)[0])
        return self.post_process_fn(ch.argmax(predictions, dim=1))

    def train(self, train_loader, val_loader, **kwargs):
        # TODO: Implement
        pass

    def eval(self, loader, loss_function, acc_fn, detach: bool = True, **kwargs):
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        # Set model to eval model
        self.set_eval()
        iterator = tqdm(loader)
        for x, y in iterator:
            x, y = x.cuda(), y.cuda()
            with ch.set_grad_enabled(not detach):
                y_logits = self.forward(x)
                loss = loss_function(y_logits, y)
                y_predicted = ch.argmax(y_logits, dim=1)
                accuracy = acc_fn(y_predicted, y)
            loss_tracker.update(loss, x.size(0))
            acc_tracker.update(accuracy, x.size(0))
            iterator.set_description("Loss: {:.4f}, Accuracy: {:.4f}".format(
                loss_tracker.avg, acc_tracker.avg))
        return loss_tracker.avg, acc_tracker.avg
