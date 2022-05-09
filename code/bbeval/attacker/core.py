import numpy as np
from bbeval.models.core import GenericModelWrapper


class Attacker:
    def __init__(self,
                 model: GenericModelWrapper,
                 query_budget: int = np.inf,
                 norm_type: float = np.inf,
                 targeted: bool = True,
                 loss_type: str = 'xent',
                 seed: int = None):
        self.model = model
        self.query_budget = query_budget
        self.norm_type = norm_type
        self.targeted = targeted
        self.loss_type = loss_type
        self.seed = seed

    def attack(self, x, y, eps: float, **kwargs):
        pass

# TODO: Figure out a good way to strucutre classes for partial/full auxiliary information