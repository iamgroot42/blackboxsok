from bbeval.config.core import ModelConfig
import torch as ch
import numpy as np

from bbeval.models.core import GenericModelWrapper
from bbeval.utils import AverageMeter
from bbeval.config import ModelConfig, TrainConfig
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
        predictions = ch.softmax(predictions, dim=1)
        if k == np.inf:
            return predictions
        top_k_probs = ch.topk(predictions, k, dim=1)[0]
        return top_k_probs

    def predict(self, x) -> int:
        predictions = self.predict_proba(x)
        # return self.post_process_fn(ch.argmax(predictions, dim=1)[0])
        return ch.argmax(predictions, dim=1)

    def _train_epoch(self, loader, optimizer, loss_function, acc_fn):
        """
            Train model for single epoch
        """
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        iterator = tqdm(loader)
        for x, y in iterator:
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            y_logits = self.forward(x)
            loss = loss_function(y_logits, y)
            y_predicted = ch.argmax(y_logits, dim=1)
            accuracy = acc_fn(y_predicted, y)
            loss.backward()
            optimizer.step()
            loss_tracker.update(loss, x.size(0))
            acc_tracker.update(accuracy, x.size(0))
            iterator.set_description("Loss: {:.4f}, Accuracy: {:.4f}".format(
                loss_tracker.avg, acc_tracker.avg))
        return loss_tracker.avg, acc_tracker.avg

    def train(self, train_loader, val_loader, loss_function, acc_fn, train_config: TrainConfig, **kwargs):
        """
            Train model for given data (via loader) 
        """
        optimizer = ch.optim.Adam(
            self.model,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay)
        # TODO: Implement 'verbose' parameter
        iterator = tqdm(range(1, train_config.epochs + 1))
        for e in iterator:
            # Train
            avg_train_loss, avg_train_acc = self._train_epoch(
                train_loader, optimizer, loss_function, acc_fn)
            # Eval
            avg_val_loss, avg_val_acc = self.eval(
                val_loader, loss_function, acc_fn)
            # Update
            iterator.set_description("Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(
                e, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))

    def eval(self, loader, loss_function, acc_fn, detach: bool = True, **kwargs):
        """
            Evaluate model for given data (via loader)
        """
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
