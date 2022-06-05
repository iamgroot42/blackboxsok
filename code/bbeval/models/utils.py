from bbeval.models.pytorch import image
from bbeval.config import ModelConfig
from bbeval.config import AuxModelConfig

MODEL_WRAPPER_MAPPING = {
    "inceptionv3": image.Inceptionv3,    
    "resnet18": image.ResNet18,
    "vgg16": image.VGG16
}


# def get_model_wrapper(model_config: ModelConfig):
def get_model_wrapper(model_config: None):
    """
        Create model wrapper for given model-config
    """
    # wrapper = MODEL_WRAPPER_MAPPING.get(model_config.name, None)
    # if not wrapper:
    #     raise NotImplementedError(
    #         f"Model {model_config.name} not implemented")
    # return wrapper(model_config)
    if isinstance(model_config,ModelConfig):
        wrapper = MODEL_WRAPPER_MAPPING.get(model_config.name, None)
        if not wrapper:
            raise NotImplementedError(
                f"Model {model_config.name} not implemented")
        return wrapper(model_config)
    elif isinstance(model_config,AuxModelConfig):
        # aux model names are given as lists
        wrappers = {}
        for _name in model_config.name:
            wrappers[_name] = MODEL_WRAPPER_MAPPING.get(_name, None)(model_config)
        return wrappers
    else:
        raise NotImplementedError("Unsupported model configuration")
