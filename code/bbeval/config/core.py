from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
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
    """Name of model"""
    dataset: str
    """Which dataset this model is for"""
    use_pretrained: Optional[bool] = False
    "Use pre-trained model from library?"


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
    access_level: str = field(choice(["only label", "top-k", "all"]))
    """What level of access does the attacker have?"""
    query_budget: Optional[int] = np.inf
    """Query budget"""
    norm_type: Optional[float] = np.inf
    """Norm type (for bounding perturbations)"""
    targeted: Optional[bool] = True
    """Is the attack targeted?"""
    loss_type: Optional[str] = "ce"
    """Loss type"""
    seed: Optional[int] = None
    """Seed for RNG"""



@dataclass
class VictimConfig(Serializable):
    """
        Configuration for the victim
    """
    access_level: str = field(choice(["only label", "top-k", "all"]))
    """What level of access does the victim provide?"""
