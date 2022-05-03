from dataclasses import dataclass
from typing import Optional
from simple_parsing.helpers import Serializable, choice


@dataclass
class DatasetConfig(Serializable):
    """
        Parameters for datasets
    """
    name: str
    """Name of the dataset"""
    type = choice(["image", "malware"])
    """What domain does the dataset belong to?"""
    

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
    access_level: choice(["only label", "top-k", "all"])
    """What level of access does the attacker have?"""


@dataclass
class VictimConfig(Serializable):
    """
        Configuration for the victim
    """
    access_level: choice(["only label", "top-k", "all"])
    """What level of access does the victim provide?"""
