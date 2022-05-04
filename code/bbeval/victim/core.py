"""
    Victim class keeps track of the total number of queries made to it so far.
    Stops returning results when # of queries exceeds query budget
"""
import numpy as np


class VictimModelWrapper:
    def __init__(self,
                 model,
                 total_queries_allowed: int = np.inf):
        super().__init__(model)
        self.queries_so_far = 0
        self.total_queries_allowed = total_queries_allowed
    
    def post_process_fn(self, tensor):
        return tensor.detach().cpu().numpy()

    def query_check(self):
        self.queries_so_far += 1
        if self.queries_so_far > self.total_queries_allowed:
            return False
        return True

    def get_top_k_probabilities(self, x, k) -> np.ndarray:
        if not self.query_check():
            return None
        return super().get_top_k_probabilities(x, k)

    def get_all_probabilities(self, x) -> np.ndarray:
        if not self.query_check():
            return None
        return super().get_all_probabilities(x)

    def get_predicted_class(self, x) -> int:
        if not self.query_check():
            return None
        return super().get_predicted_class(x)
