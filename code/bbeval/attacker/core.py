from bbeval.models.core import GenericModelWrapper
from bbeval.config import AttackerConfig
from bbeval.logger.core import Logger


# TODO: figure out if this way of defining aux_model is correct or not
class Attacker:
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: AttackerConfig):
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
        self.logger = Logger(self.config)

    def attack(self, x_orig, y_label, y_target=None, x_adv=None):
        # TODO: Have sanity checks here (matching shape, presence of target if targeted, etc)
        # Call a new self._attack() from here, and have child class
        # Implement that function
        raise NotImplementedError("Attack functionbality not implemented yet")

    def save_results(self):
        # Save logger
        self.logger.save()

# TODO: Figure out a good way to structure classes for partial/full auxiliary information