import json
import torch


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, weights_only=True) if checkpoint_path is not None else None
