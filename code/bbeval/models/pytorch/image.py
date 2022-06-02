
from torchvision.models import inception_v3
from torchvision import transforms

from bbeval.config import ModelConfig
from bbeval.models.pytorch.wrapper import PyTorchModelWrapper


class Inceptionv3(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        self.use_pretrained = model_config.use_pretrained
        model = inception_v3(pretrained=self.use_pretrained)
        super().__init__(model)
    
    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]) 
        return transform_norm(x)

    def forward(self, x):
        return self.post_process_fn(self.model(x))
