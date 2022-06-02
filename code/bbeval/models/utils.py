from bbeval.models.pytorch import image
from bbeval.config import ModelConfig

MODEL_WRAPPER_MAPPING = {
    "inceptionv3": image.Inceptionv3,    
}


def get_model_wrapper(model_config: ModelConfig):
    """
        Create model wrapper for given model-config
    """
    wrapper = MODEL_WRAPPER_MAPPING.get(model_config.name, None)
    if not wrapper:
        raise NotImplementedError(
            f"Model {model_config.name} not implemented")
    return wrapper(model_config)
