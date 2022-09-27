"""
    All model architectures can be wrapped inside this generic model wrapper. This helps ensure
    cross-attack and cross-platform compatibility.
"""
from bbeval.config.core import TrainConfig
import numpy as np
import os
from bbeval.config import ModelConfig
from bbeval.utils import get_models_save_path


class GenericModelWrapper:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.save_dir = os.path.join(
            get_models_save_path(),
            self.config.dataset,
            self.config.name)
        # Make sure save-dir exists
        os.makedirs(self.save_dir, exist_ok=True)
        self.is_robust = False
        self.access_level = "all"
        if hasattr(model_config, 'access_level'):
            self.access_level = model_config.access_level
    
    def cuda(self):
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def set_train(self):
        raise NotImplementedError(
            "This method must be implemented by the child class")
    
    def set_eval(self):
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def forward(self, x, detach: bool = False, internal_call: bool = False):
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def get_top_k_probabilities(self, x, k: int, detach: bool = False, internal_call: bool = False) -> np.ndarray:
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def predict_proba(self, x, internal_call: bool = False) -> np.ndarray:
        if not internal_call and self.access_level not in ["all"]:
            raise ValueError(f"Tried accessing probs, but access level is {self.access_level}")
        return self.get_top_k_probabilities(x, k=np.inf, internal_call=True)

    def predict(self, x) -> int:
        raise NotImplementedError(
            "This method must be implemented by the child class")
    
    def predict_label(self, x) -> int:
        return self.predict(x)

    def train(self, train_loader, val_loader, train_config: TrainConfig, **kwargs):
        raise NotImplementedError(
            "This method must be implemented by the child class")

    def eval(self, loader, loss_function, acc_fn, **kwargs):
        raise NotImplementedError(
            "This method must be implemented by the child class")
    
    def zero_grad(self):
        raise NotImplementedError(
            "This method must be implemented by the child class")

# We can have two classes that inheret from this class- one for PyTorch, one for Tensorflow.
# The train() and eval() methods will be different for each, but the same for all models
# used by those packages.