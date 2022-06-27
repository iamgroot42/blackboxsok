from bbeval.attacker.full_score.square import Square_Attack
from bbeval.attacker.top1_score.rays import RayS
from bbeval.attacker.transfer_methods.staircase import Staircase
from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper


ATTACK_WRAPPER_MAPPING = {
    "square_attack": Square_Attack,
    "rays": RayS,
    "staircase_transfer": Staircase
}

def get_attack_wrapper(model: GenericModelWrapper, aux_models: dict, attack_config: AttackerConfig, experiment_config: ExperimentConfig):
    """
        Create attack wrapper for given attakc-config
    """
    wrapper = ATTACK_WRAPPER_MAPPING.get(attack_config.name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_config.name} not implemented")
    return wrapper(model, aux_models, attack_config, experiment_config)
