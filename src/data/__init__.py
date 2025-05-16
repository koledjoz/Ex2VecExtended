from .original.dataset import Ex2VecOriginalDatasetWrap, Ex2VecOriginalDatasetShared

DATASET_REGISTRY = {
    "original": Ex2VecOriginalDatasetWrap,
}

DATASET_INIT_REGISTRY = {
    "original": Ex2VecOriginalDatasetShared
}

GROUP_TO_NAME = {}

GLOBAL_SHARED_DATA = {}


def init_dataset(dataset_name, group_id, config):
    GROUP_TO_NAME[group_id] = dataset_name
    GLOBAL_SHARED_DATA[group_id] = DATASET_INIT_REGISTRY[dataset_name](config)


def create_dataset(group_id):
    if group_id not in GLOBAL_SHARED_DATA:
        raise KeyError(f'No dataset found in group {group_id} in the GLOBAL_SHARED_DATA. Please initialize a dataset '
                       f'for this group first.')
    return DATASET_INIT_REGISTRY[GROUP_TO_NAME[group_id]](GLOBAL_SHARED_DATA[group_id])


def check_dataset(name: str):
    return name.lower() in DATASET_INIT_REGISTRY


def get_available():
    return list(DATASET_INIT_REGISTRY.keys())


def is_initialized(group_id):
    return group_id in GLOBAL_SHARED_DATA


def get_initialized():
    list(GLOBAL_SHARED_DATA.keys())
