"""
    All model architectures can be wrapped inside this generic model wrapper. This helps ensure
    cross-attack and cross-platform compatibility.
"""
import numpy as np


class GenericModelWrapper:
    def __init__(self, model):
        self.model = model
    
    def set_train(self):
        raise NotImplementedError(
            "This method must be implemented by the child class")
    
    def set_eval(self):
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def forward(self, x):
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def get_top_k_probabilities(self, x, k) -> np.ndarray:
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def get_all_probabilities(self, x) -> np.ndarray:
        return self.get_top_k_probabilities(x, k=np.inf)

    def get_predicted_class(self, x) -> int:
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def train(self, train_loader, val_loader, **kwargs):
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def eval(self, loader, loss_function, acc_fn, **kwargs):
        raise NotImplementedError(
            "This method must be implemented by the child class")


# We can have two classes that inheret from this class- one for PyTorch, one for Tensorflow.
# The train() and eval() methods will be different for each, but the same for all models
# used by those packages.