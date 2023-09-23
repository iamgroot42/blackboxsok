from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import AttackerConfig
from bbeval.logger.core import Logger


# TODO: figure out if this way of defining aux_model is correct or not
class Attacker:
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        self.model = model
        self.aux_models = aux_models
        self.config = config
        # Extract relevant parameters from config
        self.name = self.config.name
        self.query_budget = self.config.query_budget
        self.norm_type = self.config.norm_type
        self.eps = self.config.eps
        self.targeted = self.config.targeted
        self.loss_type = self.config.loss_type
        self.seed = self.config.seed
        self.params = self.config.attack_params
        # Creat new logger (for debugging, etc.)
        self.logger = Logger(self.config, experiment_config)
    
    def _attack(self, x_orig, x_adv, y_label, y_target=None):
        raise NotImplementedError("Attack functionbality not implemented yet")

    def attack(self, x_orig, x_adv, y_label, y_target=None):
        # if x_adv is None, use x_orig
        if x_adv is None:
            x_adv = x_orig

        # Basic asserts and checks
        if len(x_orig) != len(y_label):
            raise ValueError("x_orig and y_label must have the same length")
        if self.targeted and y_target is None:
            raise ValueError("Target label must be provided for targeted attack")
        if len(x_adv) != len(x_orig):
            raise ValueError("x_adv and x_orig must have the same length")

        return self._attack(x_orig, x_adv, y_label, y_target)

    def optimization_loop_condition_satisfied(self, iter: int, time: float, n_iters: int):
        """Utility function to check whether optimization loop should continue or not"""
        if self.config.time_based_attack:
            return time < self.config.time_per_batch
        else:
            return iter < n_iters

    def save_results(self):
        # Save logger
        self.logger.save()

# TODO: Figure out a good way to structure classes for partial/full auxiliary information
