from bbeval.attacker.full_score import Square_Attack
from bbeval.attacker.top1_score import RayS
# from bbeval.attacker.transfer import Transfer
from bbeval.config import AttackerConfig
from bbeval.models.core import GenericModelWrapper


ATTACK_WRAPPER_MAPPING = {
    "square_attack": Square_Attack,
    "rays": RayS,
    # "transfer": Transfer,
}


def get_attack_wrapper(model: GenericModelWrapper, attack_config: AttackerConfig):
    """
        Create attack wrapper for given attakc-config
    """
    wrapper = ATTACK_WRAPPER_MAPPING.get(attack_config.name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_config.name} not implemented")
    return wrapper(model, attack_config)
