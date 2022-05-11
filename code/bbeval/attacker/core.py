from bbeval.logging.core import Logger, AttackResult
from bbeval.models.core import GenericModelWrapper
from bbeval.config import AttackerConfig


class Attacker:
    def __init__(self,
                 model: GenericModelWrapper,
                 config: AttackerConfig):
        self.model = model
        self.config = config
        # Extract relevant parameters from config
        self.query_budget = self.config.query_budget
        self.norm_type = self.config.norm_type
        self.targeted = self.config.targeted
        self.loss_type = self.config.loss_type
        self.seed = self.config.seed
        # TODO: Combine two loggers below into one
        # Creat new logger (for debugging, etc.)
        self.logger = Logger(self.config.experiment_name)
        # Create new logger for actual results (JSON readable)
        self.result_logger = AttackResult(self.config)

    def attack(self, x, y, eps: float, **kwargs):
        pass

# TODO: Figure out a good way to structure classes for partial/full auxiliary information