"""
    All model architectures can be wrapped inside this generic model wrapper. This helps ensure
    cross-attack and cross-platform compatibility.
"""
import numpy as np
from joblib import dump
import torch as ch
import os
from bbeval.config import ModelConfig, TrainConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.datasets.utils import collect_data_from_loader


class SKLearnModelWrapper(GenericModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def cuda(self):
        pass

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def save(self, path: str):
        path_to_save = os.path.join(self.save_dir, path)
        # Save sklearn model
        dump(self.model, path_to_save)

    def forward(self, x, detach: bool = False, internal_call: bool = False):
        raise ValueError("forward() calls do not make sense for sklearn models")

    def get_top_k_probabilities(self, x, k: int, detach: bool = False, internal_call: bool = False) -> np.ndarray:

        predictions = ch.from_numpy(self.model.predict_proba(x))
        if k == np.inf:
            return predictions
        top_k_probs = ch.topk(predictions, k, dim=1)[0]
        return top_k_probs

    def predict_proba(self, x, internal_call: bool = False) -> np.ndarray:
        if not internal_call and self.access_level not in ["all"]:
            raise ValueError(
                f"Tried accessing probs, but access level is {self.access_level}")
        return self.get_top_k_probabilities(x, k=np.inf, internal_call=True)

    def predict(self, x) -> int:
        return ch.from_numpy(self.model.predict(x))

    def predict_label(self, x) -> int:
        return self.predict(x)

    def train(self, train_loader, val_loader, train_config: TrainConfig, **kwargs):
        # Need to first extract all data from dataloader
        train_x, train_y = collect_data_from_loader(train_loader)
        val_x, val_y = collect_data_from_loader(val_loader)

        # In the case of sklearn models, loaders are actually data
        self.model.fit(train_x, train_y)
        train_acc = self.eval(train_x, train_y)
        val_acc = self.eval(val_x, val_y)
        print("Train acc: %.2f" % (100 * train_acc))
        print("Val acc: %.2f" % (100 * val_acc))
        return train_acc, val_acc

    def eval(self, x, y, **kwargs):
        if kwargs.get("loss_function", None) is not None:
            raise ValueError("Sklearn models only product accuracy scores")
        return self.model.score(x, y)

    def zero_grad(self):
        pass
