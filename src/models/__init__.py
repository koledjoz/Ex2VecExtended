from .original.model import Ex2VecOriginal

MODEL_REGISTRY = {
    "original": Ex2VecOriginal,
}


def get_model(name: str):
    try:
        return MODEL_REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}")


def check_model(name: str):
    return name.lower() in MODEL_REGISTRY
