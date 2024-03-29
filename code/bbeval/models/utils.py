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
    "rf": malware_sklearn.RFWrapper,
    "densenet121": image.DenseNet121,
    "densenet201": image.DenseNet201,
    "malconv": malware.SecmlMalConv,
    "sorel_0": (malware.SecmlSOREL, 0),
    "sorel_1": (malware.SecmlSOREL, 1),
    "sorel_2": (malware.SecmlSOREL, 2),
    "sorel_3": (malware.SecmlSOREL, 3),
    "sorel_4": (malware.SecmlSOREL, 4),
    "gbt": malware.SecmlGBT,
    "sorelgbt_0": (malware.SecmlGBTSorel, 0),
    "sorelgbt_1": (malware.SecmlGBTSorel, 1),
    "sorelgbt_2": (malware.SecmlGBTSorel, 2),
    "sorelgbt_3": (malware.SecmlGBTSorel, 3),
    "sorelgbt_4": (malware.SecmlGBTSorel, 4),
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
        if isinstance(wrapper, tuple):
            return wrapper[0](model_config, seed=wrapper[1])
        return wrapper(model_config)

    if isinstance(model_configs, list):
        wrappers = {model_config.name: _get_model(model_config) for model_config in model_configs}
        return wrappers
    else:
        return _get_model(model_configs)
