from .original.dataset import Ex2VecOriginalDatasetWrap, Ex2VecOriginalDatasetShared

DATASET_REGISTRY = {
    "original": Ex2VecOriginalDatasetWrap,
}

DATASET_INIT_REGISTRY = {
    "original": Ex2VecOriginalDatasetShared
}

GROUP_TO_NAME = {}

GLOBAL_SHARED_DATA = {}


def init_dataset(group_id, config):
    dataset_type = config['dataset_type']
    GROUP_TO_NAME[group_id] = dataset_type
    GLOBAL_SHARED_DATA[group_id] = DATASET_INIT_REGISTRY[dataset_type](config)


def get_dataset(group_id):
    if group_id not in GLOBAL_SHARED_DATA:
        raise KeyError(f'No dataset found in group {group_id} in the GLOBAL_SHARED_DATA. Please initialize a dataset '
                       f'for this group first.')
    return DATASET_REGISTRY[GROUP_TO_NAME[group_id]](GLOBAL_SHARED_DATA[group_id])


def check_dataset(config):
    return config['dataset_type'].lower() in DATASET_INIT_REGISTRY


def get_available_datasets():
    return list(DATASET_INIT_REGISTRY.keys())


def is_initialized(group_id):
    return group_id in GLOBAL_SHARED_DATA


def get_initialized():
    list(GLOBAL_SHARED_DATA.keys())
