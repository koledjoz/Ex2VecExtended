from .original.model import Ex2VecOriginal
from .extended.model import Ex2VecExtended
from .extendedDouble.model import Ex2VecExtendedDouble
from .BLproxy.model import BLProxy
from .knn.model import KNNModelBase, KNNModelBL

MODEL_REGISTRY = {
    "original": Ex2VecOriginal,
    "extended": Ex2VecExtended,
    "extendeddouble": Ex2VecExtendedDouble,
    "bl_proxy": BLProxy,
    "knnBase": KNNModelBase,
    "knnBL": KNNModelBL
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
        raise ValueError(f"Probably unknown model: {config['model_type']}. "
                         f"Available models: {list(MODEL_REGISTRY.keys())}. "
                         f"Other options is missing value in config file. Please see previous exceptions.")


def check_model(config):
    return config['model_type'].lower() in MODEL_REGISTRY


def get_available_models():
    return list(MODEL_REGISTRY.keys())


def get_model_type(model_type):
    return MODEL_REGISTRY[model_type]
