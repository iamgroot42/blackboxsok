from torchvision.models import inception_v3

from bbeval.models.pytorch_model import PyTorchModelWrapper
from bbeval.config import ModelConfig


class Inceptionv3(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        model = inception_v3(pretrained=model_config.use_pretrained)
        super().__init__(model)
