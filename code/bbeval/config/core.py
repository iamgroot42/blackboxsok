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
    root: str = "./data"
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
    use_pretrained: Optional[bool] = False
    "Use pre-trained model from library?"


@dataclass
class AttackerConfig(Serializable):
    """
        Configuration for the attacker
    """
    experiment_name: str
    """Name for experiment"""
    access_level: str = field(choice(["only label", "top-k", "all"]))
    """What level of access does the attacker have?"""
    query_budget: int = np.inf
    """Query budget"""
    norm_type: float = np.inf
    """Norm type (for bounding perturbations)"""
    targeted: bool = True
    """Is the attack targeted?"""
    loss_type: str = "xent"
    """Loss type"""
    seed: int = None
    """Seed for RNG"""



@dataclass
class VictimConfig(Serializable):
    """
        Configuration for the victim
    """
    access_level: str = field(choice(["only label", "top-k", "all"]))
    """What level of access does the victim provide?"""
