from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np
from dacite import from_dict
from simple_parsing.helpers import Serializable, choice, field
from bbeval.utils import get_dataset_dir_path


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
    root: str = get_dataset_dir_path()
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
class TransferredAttackConfig(Serializable):
    """
        Configuration for Staircase attack
    """
    n_iters: int
    """Number of iterations"""
    image_resizes: List[int]
    """List of image resizes to try"""
    image_width: int = 299
    """Width of image"""
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
    p_init: float = 0.05
    """TODO: Check"""


@dataclass
class BayesOpt(Serializable):
    """
        Configuration for BayesOpt attack
    """
    n_iters: int
    """Number of iterations"""
    image_resizes: List[int]
    """List of image resizes to try"""
    image_width: int = 299
    """Width of image"""
    interpol_dim: int = 256
    """Interpolation dimension"""


@dataclass
class BayesOpt_full(Serializable):
    """
        Configuration for BayesOpt_full attack
    """
    n_iters: int
    """Number of iterations"""
    image_resizes: List[int]
    """List of image resizes to try"""
    image_width: int = 299
    """Width of image"""
    interpol_dim: int = 256
    """Interpolation dimension"""


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
class NESConfig(Serializable):
    """
        Configuration for Sparse-EVO Attack
    """
    max_queries: int


@dataclass
class MalwareAttackerConfig(Serializable):
    """
        Configuration for malware-based attacler
    """
    name: str
    """Which attack is this?"""
    adv_model_config: ModelConfig
    """Model config for adversary's model"""
    access_level: str = field(choice(["only label", "top-k", "all", "none"]))
    """What level of access does the attacker have?"""
    aux_model_configs_dict: Optional[List[dict]] = None
    """Model configs for adversary's leveraged auxiliary models"""
    aux_model_configs: Optional[List] = None
    attack_params: Optional[dict] = None
    """Additional attack-specific parameters"""
    query_budget: Optional[int] = np.inf
    """Query budget"""
    loss_type: Optional[str] = "ce"
    """Loss type"""
    seed: Optional[int] = None
    """Seed for RNG"""

    def __post_init__(self):
        # Have to do this because SimpleParsing does not support list of dataclasses
        data_class = type(self.adv_model_config)
        if self.aux_model_configs_dict:
            self.aux_model_configs = [data_class(**aux_dict) for aux_dict in self.aux_model_configs_dict]


@dataclass
class AttackerConfig(Serializable):
    """
        Configuration for the attacker
    """
    name: str
    """Which attack is this?"""
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
    target_label_selection_mode: Optional[str] = field(choice(["easiest", "hardest", "random", "user"]))
    """How to select target label (for targeted attack)?"""
    target_label: Optional[int] = None
    """Target label to use (if 'user' mode is used)"""
    loss_type: Optional[str] = "ce"
    """Loss type"""
    seed: Optional[int] = None
    """Seed for RNG"""

    time_based_attack: Optional[bool] = True
    """If true, attack (per batch) terminates upon crossing time limit, not iteration limit"""
    time_per_batch: Optional[int] = 1800
    """If time_based_attack is True, this is the time limit per batch (in seconds)"""
    track_local_metrics: Optional[bool] = True
    """Track local attack metrics (ASR and loss) while running attack?"""

    def __post_init__(self):
        # Have to do this because SimpleParsing does not support list of dataclasses
        data_class = type(self.adv_model_config)
        if self.aux_model_configs_dict:
            self.aux_model_configs = [data_class(**aux_dict) for aux_dict in self.aux_model_configs_dict]
            # self.aux_model_configs = [from_dict(data_class=data_class, data=aux_dict) for aux_dict in self.aux_model_configs_dict]


@dataclass
class ExperimentConfig(Serializable):
    """
        Configuration for an experiment
    """
    experiment_name: str
    """Name for experiment"""
    dataset_config: DatasetConfig
    """Config file for the dataset this attack will use"""
    attack_configs_dict: List[dict]
    """Config ficts (converted to dataclasses later) for each attack in order"""
    attack_configs: Optional[List] = None
    """Config ficts (converted to dataclasses later) for each attack in order"""
    batch_size: Optional[int] = 32
    """Batch size for executing attacks"""
    profiler: Optional[bool] = False
    """Run profiler to measure GPU runtime?"""

    def first_attack_config(self):
        return self.attack_configs[0]
    
    def second_attack_config(self):
        return self.attack_configs[1] if len(self.attack_configs) > 1 else None

    def __post_init__(self):
        # For now, we support only 2 attack configs
        if len(self.attack_configs_dict) > 2:
            raise ValueError("Only 2 attack configs supported")
        # Have to do this because SimpleParsing does not support list of dataclasses
        self.attack_configs = []
        if self.dataset_config.type == "malware":
            class_to_use = MalwareAttackerConfig
        else:
            class_to_use = AttackerConfig

        for config_dict in self.attack_configs_dict:
            print("Use", class_to_use, "with", config_dict)
            self.attack_configs.append(class_to_use.from_dict(config_dict))


@dataclass
class ModelTrainingConfig(Serializable):
    """
        Configuration for training a model.
    """
    model_config: ModelConfig
    """Config file for model"""
    train_config: TrainConfig
    """Config file for training"""
