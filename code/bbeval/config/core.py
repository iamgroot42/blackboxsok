from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np
from dacite import from_dict
from simple_parsing.helpers import Serializable, choice, field


@dataclass
class DatasetConfig(Serializable):
    """
        Parameters for datasets
    """
    name: str
    """Name of str = the dataset"""
    type: str = field(choice(["image", "malware"]))
    """What domain does the dataset belong to?"""
    augment: bool = False
    """Use data augmentation?"""
    root: str = "/p/blackboxsok/datasets"
    """Path to datasets"""
    

@dataclass
class TrainConfig(Serializable):
    """
        Configuration values for training models.
    """
    data_config: DatasetConfig
    """Configuration for dataset"""
    epochs: int
    """Number of epochs to train for"""
    learning_rate: float
    """Learning rate for optimizer"""
    batch_size: int
    """Batch size for training"""
    verbose: Optional[bool] = False
    """Whether to print out per-classifier stats"""
    weight_decay: Optional[float] = 0
    """L2 regularization weight"""
    get_best: Optional[bool] = True
    """Whether to get the best performing model (based on validation data)"""
    cpu: Optional[bool] = False
    """Whether to train on CPU or GPU"""


@dataclass
class ModelConfig(Serializable):
    """
        Configuration for model
    """
    name: str
    """Name(s) of model(s)"""
    dataset: str
    """Which dataset this model is for"""
    use_pretrained: Optional[bool] = False
    "Use pre-trained model from library?"
    misc_dict: Optional[dict] = None
    """Extra parameters that may be used by the model"""


@dataclass
class MalwareModelConfig(ModelConfig):
    embedding_size: Optional[int] = None
    """Embedding size (if used)"""
    max_input_size: Optional[int] = None
    """Maximum input size (if used)"""
    embedding_value: Optional[int] = 256
    """Embedding value (if used)"""
    shift_values: Optional[bool] = False
    """Shift values (if used)"""


@dataclass
class StairCaseConfig(Serializable):
    """
        Configuration for Staircase attack
    """
    n_iters: int
    """Number of iterations"""
    image_resizes: List[int]
    """List of image resizes to try"""
    image_width: int = 299
    """Width of image"""
    amplification: float = 1.5
    """Amplification factor"""
    prob: float = 0.7
    """TODO: Fill"""
    interpol_dim: int = 256
    """Interpolation dimension"""


@dataclass
class SquareAttackConfig(Serializable):
    """
        Configuration for Square attack
    """
    # these default values are for imagenet only
    n_iters: int = 10000 # 100,000 for targeted attack 
    """TODO: Check"""
    p_init = 0.05
    """TODO: Check"""


@dataclass
class SparseEvoConfig(Serializable):
    """
        Configuration for Sparse-EVO Attack
    """
    n_pix: int = 4
    """TODO: Check"""
    mu: float = 0.04
    """TODO: Check"""
    pop_size: int = 10
    """TODO: Check"""
    cr: float = 0.5
    """TODO: Check"""
    scale: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    """TODO: Check"""


@dataclass
class AttackerConfig(Serializable):
    """
        Configuration for the attacker
    """
    name: str
    """Which attack is this?"""
    experiment_name: str
    """Name for experiment"""
    dataset_config: DatasetConfig
    """Config file for the dataset this attack will use"""
    adv_model_config: ModelConfig
    """Model config for adversary's model"""
    eps: float
    """Perturbation budget (epsilon)"""

    aux_model_configs_dict: Optional[List[dict]] = None
    """Model configs for adversary's leveraged auxiliary models"""
    aux_model_configs: Optional[List] = None

    attack_params: Optional[dict] = None
    """Additional attack-specific parameters"""

    access_level: str = field(choice(["only label", "top-k", "all", "none"]))
    """What level of access does the attacker have?"""
    query_budget: Optional[int] = np.inf
    """Query budget"""
    norm_type: Optional[float] = np.inf
    """Norm type (for bounding perturbations)"""
    targeted: Optional[bool] = False
    """Is the attack targeted?"""
    loss_type: Optional[str] = "ce"
    """Loss type"""
    seed: Optional[int] = None
    """Seed for RNG"""

    def __post_init__(self):
        # Have to do this because SimpleParsing does not support list of dataclasses
        data_class = type(self.adv_model_config)
        if self.aux_model_configs_dict:
            self.aux_model_configs = [from_dict(data_class=data_class, data=aux_dict) for aux_dict in self.aux_model_configs_dict]
