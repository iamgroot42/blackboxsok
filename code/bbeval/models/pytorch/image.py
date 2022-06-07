
from robustbench.utils import load_model
from torchvision.models import inception_v3, resnet18, vgg16
from torchvision import transforms

from bbeval.config import ModelConfig
from bbeval.models.pytorch.wrapper import PyTorchModelWrapper

class Inceptionv3(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = inception_v3(pretrained=self.use_pretrained, aux_logits=False)
    
    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        return x
        # TODO: Fix below to remove shape-based errors
        transform_norm = transforms.Compose([
            transforms.Normalize(mean, std)
        ]) 
        return transform_norm(x)

    # def post_process_fn(self, tensor):
    #     return tensor

class ResNet18(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = resnet18(pretrained=self.use_pretrained)
    
    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        return x
        # TODO: Fix below to remove shape-based errors
        transform_norm = transforms.Compose([
            transforms.Normalize(mean, std)
        ]) 
        return transform_norm(x)
    
class VGG16(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = vgg16(pretrained=self.use_pretrained)
    
    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        return x
        # TODO: Fix below to remove shape-based errors
        transform_norm = transforms.Compose([
            transforms.Normalize(mean, std)
        ]) 
        return transform_norm(x)

class RobustBenchModel(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = True
        self.is_robust = True
        # TODO: Make threat model configurable
        self.model = load_model(model_name=self.config.name,
                                model_dir=self.save_dir,
                                dataset=self.config.dataset)

# _MODEL_MAPPING = {
#     "inceptionv3": Inceptionv3,
#     "resnet18": ResNet18,
#     "vgg16": VGG16
# }