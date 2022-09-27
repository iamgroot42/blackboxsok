"""
    All model architectures can be wrapped inside this generic model wrapper. This helps ensure
    cross-attack and cross-platform compatibility.
"""
import numpy as np
import torch as ch
from bbeval.config import ModelConfig
from bbeval.models.core import GenericModelWrapper


class SKLearnModelWrapper(GenericModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def cuda(self):
        pass

    def set_train(self):
        pass

    def set_eval(self):
        pass

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

    def train(self, train_loader, val_loader, **kwargs):
        # In the case of sklearn models, loaders are actually data
        self.model.fit(train_loader)
        train_acc = self.eval(train_loader, None, None)
        val_acc = self.eval(val_loader, None, None)
        print("Train acc: ", train_acc)
        print("Val acc: ", val_acc)
        return train_acc, val_acc

    def eval(self, loader, loss_function, acc_fn, **kwargs):
        if loss_function is not None:
            raise ValueError("Sklearn models only product accuracy scores")
        return self.model.score(loader[0], loader[1])

    def zero_grad(self):
        pass
