from .original.model import Ex2VecOriginal

MODEL_REGISTRY = {
    "original": Ex2VecOriginal,
}


def load_model(config, checkpoint=None):
    model = get_model(config)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_model(config):
    try:
        return MODEL_REGISTRY[config['model_type'].lower()](config)
    except KeyError:
        raise ValueError(f"Probably nknown model: {config['model_type']}. "
                         f"Available models: {list(MODEL_REGISTRY.keys())}. "
                         f"Other options is missing value in config file. Please see previous exceptions.")


def check_model(config):
    return config['model_type'].lower() in MODEL_REGISTRY


def get_available_models():
    return list(MODEL_REGISTRY.keys())
