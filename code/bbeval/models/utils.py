from bbeval.models.pytorch import image, malware
from bbeval.models.sklearn import malware as malware_sklearn
from bbeval.config import ModelConfig

MODEL_WRAPPER_MAPPING = {
    "inceptionv3": image.Inceptionv3,    
    "resnet18": image.ResNet18,
    "resnet50": image.ResNet50,
    "vgg16": image.VGG16,
    "vgg16_bn": image.VGG16_bn,
    "vgg19": image.VGG19,
    "resnet101": image.ResNet101,
    "inceptionv4": image.Inceptionv4,
    "inceptionresnetv2": image.InceptionResNetv2,
    "ensinceptionresnetv2": image.EnsAdvInceptionResNetv2,
    "advinceptionv3": image.AdvInceptionv3,
    "imagenetrobust": image.ImageNetRobust,
    "cifar10robust": image.Cifar10Robust,
    "mlp3": malware.MLPClassifier,
    "rf": malware_sklearn.RFWrapper,
    "malconv": malware.SecmlMalConv,
    "gbt": malware.SecmlGBT,
    "densenet121": image.DenseNet121,
    "densenet201": image.DenseNet201
}


def get_model_wrapper(model_configs: ModelConfig):
    """
        Create model wrapper for given model-config
    """
    def _get_model(model_config):
        wrapper = MODEL_WRAPPER_MAPPING.get(model_config.name, None)
        if not wrapper:
            raise NotImplementedError(
                f"Model {model_config.name} not implemented")
        return wrapper(model_config)

    if isinstance(model_configs, list):
        wrappers = {model_config.name: _get_model(model_config) for model_config in model_configs}
        return wrappers
    else:
        return _get_model(model_configs)
