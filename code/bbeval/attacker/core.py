from bbeval.models.core import GenericModelWrapper
from bbeval.config import AttackerConfig
from bbeval.logger.core import Logger


class Attacker:
    def __init__(self,
                 model: GenericModelWrapper,
                 config: AttackerConfig):
        self.model = model
        self.config = config
        # Extract relevant parameters from config
        self.name = self.config.name
        self.query_budget = self.config.query_budget
        self.norm_type = self.config.norm_type
        self.targeted = self.config.targeted
        self.loss_type = self.config.loss_type
        self.seed = self.config.seed
        # Creat new logger (for debugging, etc.)
        self.logger = Logger(self.config)

    def attack(self, x, y, eps: float, **kwargs):
        pass

    def save_results(self):
        # Save logger
        self.logger.save()

# TODO: Figure out a good way to structure classes for partial/full auxiliary information