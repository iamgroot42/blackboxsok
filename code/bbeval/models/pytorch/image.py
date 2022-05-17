
from torchvision.models import inception_v3
from bbeval.config import ModelConfig
from bbeval.models.pytorch.wrapper import PyTorchModelWrapper


class Inceptionv3(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        self.use_pretrained = model_config.use_pretrained
        model = inception_v3(pretrained=self.use_pretrained)
        super().__init__(model)

    def forward(self, x):
        return self.post_process_fn(self.model(x))
