from .original.model import Ex2VecOriginal

MODEL_REGISTRY = {
    "original": Ex2VecOriginal,
}


def get_model(config):
    try:
        return MODEL_REGISTRY[config['model_type'].lower()](config)
    except KeyError:
        raise ValueError(f"Unknown model: {config['model_type']}. Available models: {list(MODEL_REGISTRY.keys())}")


def check_model(config):
    return config['model_type'].lower() in MODEL_REGISTRY


def get_available_models():
    return list(MODEL_REGISTRY.keys())
