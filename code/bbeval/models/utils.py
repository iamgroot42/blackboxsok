from bbeval.models.pytorch import image
from bbeval.config import ModelConfig

MODEL_WRAPPER_MAPPING = {
    "inceptionv3": image.Inceptionv3,    
    "resnet18": image.ResNet18,
    "vgg16": image.VGG16
}


def get_model_wrapper(model_config: ModelConfig):
    """
        Create model wrapper for given model-config
    """
    def _get_model(name):
        wrapper = MODEL_WRAPPER_MAPPING.get(name, None)
        if not wrapper:
            raise NotImplementedError(
                f"Model {model_config.name} not implemented")
        return wrapper(model_config)

    if isinstance(model_config.name, list):
        wrappers = {_name: _get_model(_name) for _name in model_config.name}
        return wrappers
    else:
        return _get_model(model_config.name)
