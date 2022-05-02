"""
    Victim class keeps track of the total number of queries made to it so far.
    Stops returning results when # of queries exceeds query budget
"""
import numpy as np
from bbeval.models.core import GenericModelWrapper


class VictimWrapper:
    def __init__(self,
                model: GenericModelWrapper,
                total_queries_allowed: int = np.inf):
        self.model = model
        self.queries_so_far = 0
        self.total_queries_allowed = total_queries_allowed
    
    def query_check(self):
        self.queries_so_far += 1
        if self.queries_so_far > self.total_queries_allowed:
            return False
        return True

    def get_top_k_probabilities(self, x, k) -> np.ndarray:
        if not self.query_check():
            return None
        return self.model.get_top_k_probabilities(x, k)

    def get_all_probabilities(self, x) -> np.ndarray:
        if not self.query_check():
            return None
        return self.model.get_all_probabilities(x)

    def get_predicted_class(self, x) -> int:
        if not self.query_check():
            return None
        return self.model.get_predicted_class(x)
