import numpy as np
from bbeval.models.core import GenericModelWrapper


class Attacker:
    def __init__(self,
                 model: GenericModelWrapper,
                 query_budget: int = np.inf):
        self.model = model
        self.query_budget = query_budget


# TODO: Figure out a good way to strucutre classes for partial/full auxiliary information