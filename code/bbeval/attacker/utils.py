from bbeval.attacker.full_score.square import Square_Attack
from bbeval.attacker.top1_score.rays import RayS
from bbeval.attacker.transfer_methods.FGSM import FGSM
from bbeval.attacker.transfer_methods.IFGSM import IFGSM
from bbeval.attacker.transfer_methods.MIFGSM import MIFGSM
from bbeval.attacker.transfer_methods.NIFGSM import NIFGSM
from bbeval.attacker.transfer_methods.VMIFGSM import VMIFGSM
from bbeval.attacker.transfer_methods.VNIFGSM import VNIFGSM
from bbeval.attacker.transfer_methods.DIFGSM import DIFGSM
from bbeval.attacker.transfer_methods.MIDIFGSM import MIDIFGSM
from bbeval.attacker.transfer_methods.TIFGSM import TIFGSM
from bbeval.attacker.transfer_methods.MITIDIFGSM import MITIDIFGSM
from bbeval.attacker.transfer_methods.SIFGSM import SIFGSM
from bbeval.attacker.transfer_methods.MITIDISIFGSM import MITIDISIFGSM
from bbeval.attacker.transfer_methods.NITIDISIFGSM import NITIDISIFGSM
from bbeval.attacker.transfer_methods.VMITIDISIFGSM import VMITIDISIFGSM
from bbeval.attacker.transfer_methods.VNITIDISIFGSM import VNITIDISIFGSM
from bbeval.attacker.transfer_methods.SMIFGSM import SMIFGSM
from bbeval.attacker.transfer_methods.SMIMIFGSM import SMIMIFGSM
from bbeval.attacker.transfer_methods.SMIMITIDISIFGSM import SMIMITIDISIFGSM
from bbeval.attacker.transfer_methods.EMIFGSM import EMIFGSM
from bbeval.attacker.transfer_methods.EMITIDISIFGSM import EMITIDISIFGSM
from bbeval.attacker.transfer_methods.IFGSSM import IFGSSM
from bbeval.attacker.transfer_methods.PITIDIFGSSM import PITIDIFGSSM
from bbeval.attacker.transfer_methods.ADMIXFGSM import ADMIXFGSM
from bbeval.attacker.transfer_methods.MIADMIXTIDIFGSM import MIADMIXTIDIFGSM
from bbeval.attacker.transfer_methods.RAPFGSM import RAPFGSM
from bbeval.attacker.top1_score.BayesOpt import BayesOpt
from bbeval.attacker.topk_score.NES import NES
from bbeval.attacker.full_score.BayesOpt_full import BayesOpt_full
from bbeval.attacker.transfer_methods.MITIDIAIFGSSM import MITIDIAIFGSSM
from bbeval.attacker.transfer_methods.EMITIDIAIFGSM import EMITIDIAIFGSM
from bbeval.attacker.transfer_methods.SMIMITIDIAIFGSM import SMIMITIDIAIFGSM
from bbeval.attacker.transfer_methods.VMITIDIAIFGSM import VMITIDIAIFGSM
from bbeval.attacker.transfer_methods.VNITIDIAIFGSM import VNITIDIAIFGSM

from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper


ATTACK_WRAPPER_MAPPING = {
    "square_attack": Square_Attack,
    "rays": RayS,
    "FGSM_transfer": FGSM,
    "IFGSM_transfer": IFGSM,
    "MIFGSM_transfer": MIFGSM,
    "NIFGSM_transfer": NIFGSM,
    "VMIFGSM_transfer": VMIFGSM,
    "VNIFGSM_transfer": VNIFGSM,
    "DIFGSM_transfer": DIFGSM,
    "MIDIFGSM_transfer": MIDIFGSM,
    "TIFGSM_transfer": TIFGSM,
    "MITIDIFGSM_transfer": MITIDIFGSM,
    "SIFGSM_transfer": SIFGSM,
    "MITIDISIFGSM_transfer": MITIDISIFGSM,
    "NITIDISIFGSM_transfer": NITIDISIFGSM,
    "VMITIDISIFGSM_transfer": VMITIDISIFGSM,
    "VNITIDISIFGSM_transfer": VNITIDISIFGSM,
    "SMIFGSM_transfer": SMIFGSM,
    "SMIMIFGSM_transfer": SMIMIFGSM,
    "SMIMITIDISIFGSM_transfer": SMIMITIDISIFGSM,
    "EMIFGSM_transfer": EMIFGSM,
    "EMITIDISIFGSM_transfer": EMITIDISIFGSM,
    "IFGSSM_transfer": IFGSSM,
    "PITIDIFGSSM_transfer": PITIDIFGSSM,
    "ADMIXFGSM_transfer": ADMIXFGSM,
    "MIADMIXTIDIFGSM_transfer": MIADMIXTIDIFGSM,
    "EMITIDIAIFGSM_transfer": EMITIDIAIFGSM,
    "SMIMITIDIAIFGSM_transfer": SMIMITIDIAIFGSM,
    "MIADMIXTIDIFGSSM_transfer": MITIDIAIFGSSM,
    "VMITIDIAIFGSM_transfer": VMITIDIAIFGSM,
    "VNITIDIAIFGSM_transfer": VNITIDIAIFGSM,
    "RAPFGSM_transfer": RAPFGSM,
    "BayesOpt":BayesOpt,
    "NES": NES,
    "BayesOpt_full":BayesOpt_full
}

def get_attack_wrapper(model: GenericModelWrapper, aux_models: dict, attack_config: AttackerConfig, experiment_config: ExperimentConfig):
    """
        Create attack wrapper for given attakc-config
    """
    wrapper = ATTACK_WRAPPER_MAPPING.get(attack_config.name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_config.name} not implemented")
    return wrapper(model, aux_models, attack_config, experiment_config)
