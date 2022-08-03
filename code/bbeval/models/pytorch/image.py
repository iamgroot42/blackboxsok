
from robustbench.utils import load_model
from torchvision.models import inception_v3, resnet18, vgg16, resnet101, resnet50, vgg16_bn
from torchvision import transforms

from bbeval.config import ModelConfig
from bbeval.models.pytorch.wrapper import PyTorchModelWrapper
import timm

class Inceptionv3(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = inception_v3(pretrained=self.use_pretrained, aux_logits=True)
    
    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                        std=std)
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
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                        std=std)
        return transform_norm(x)

class ResNet50(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = resnet50(pretrained=self.use_pretrained)

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

class VGG16(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = vgg16(pretrained=self.use_pretrained)
    
    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                        std=std)
        return transform_norm(x)

class VGG16_bn(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = vgg16_bn(pretrained=self.use_pretrained)

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
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

class ResNet101(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = resnet101(pretrained=self.use_pretrained)

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

class Inceptionv4(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = timm.create_model('inception_v4', pretrained=self.use_pretrained)

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

class InceptionResNetv2(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = timm.create_model('inception_resnet_v2', pretrained=self.use_pretrained)

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

class EnsAdvInceptionResNetv2(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=self.use_pretrained)

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

class AdvInceptionv3(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = timm.create_model('adv_inception_v3', pretrained=self.use_pretrained)

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

class ImageNetRobust(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf')

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

class Cifar10Robust(PyTorchModelWrapper):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_pretrained = model_config.use_pretrained
        self.model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')

    def pre_process_fn(self, x):
        # imagenet inputs need to be normalized
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_norm = transforms.Normalize(mean=mean,
                                              std=std)
        return transform_norm(x)

# _MODEL_MAPPING = {
#     "inceptionv3": Inceptionv3,
#     "resnet18": ResNet18,
#     "vgg16": VGG16
# }