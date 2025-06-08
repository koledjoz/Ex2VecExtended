import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import json

from ..utils import sample_excluding


class Ex2VecExtendedDatasetShared:
    def __init__(self, config):
        self.disable_tqdm = not config['verbose']
        self.data_path = config['data_path']
        self.usage_dict_path = config['usage_dict_path']
        self.grouping_size = config['grouping_size']
        self.sample_negative = config['sample_negative']
        self.history_size = config['history_size']

        self.data = pd.read_parquet(self.data_path)

        with open(self.usage_dict_path) as file:
            self.use_dict = {int(key): set(value) for key, value in json.load(file).items()}

        self.max_user = self.data['user_id'].max()
        self.max_item = self.data['track_id'].max()

    def get_n_users(self):
        return self.max_user

    def get_n_items(self):
        return self.max_item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pred_user_id = self.data.iloc[idx]['user_id']
        pred_item = self.data.iloc[idx]['track_id']

        if pred_item not in self.use_dict[pred_user_id]:
            return None

        pred_items = np.append(np.array(sample_excluding(self.max_item, self.sample_negative, pred_item)), pred_item)
        true_vals = np.append(np.array([0.0 for _ in range(len(pred_items) - 1)]), 1.0)

        history = self.data.iloc[max(idx - self.history_size, 0):idx]
        history = history[history['user_id'] == pred_user_id]

        ts = self.data.iloc[idx]['ts']
        timedeltas = (ts - history['ts']).to_numpy()

        history_items = history['track_id'].to_numpy()
        weights = np.ones_like(history_items)

        timedeltas = np.pad(timedeltas, (0, self.history_size - len(timedeltas)), mode='constant', constant_values=0)
        history_items = np.pad(history_items, (0, self.history_size - len(history_items)), mode='constant',
                               constant_values=0)
        weights = np.pad(weights, (0, self.history_size - len(weights)), mode='constant', constant_values=0)

        return {
            'user_id': torch.tensor(pred_user_id),
            'predict_items': torch.tensor(pred_items),
            'real_values': torch.tensor(true_vals),
            'history_items': torch.tensor(history_items),
            'timedeltas': torch.tensor(timedeltas),
            'weights': torch.tensor(weights)
        }


class Ex2VecOriginalDatasetWrap(torch.utils.data.Dataset):
    def __init__(self, shared_data):
        self.shared_data = shared_data

    def get_n_users(self):
        return self.shared_data.get_n_users()

    def get_n_items(self):
        return self.shared_data.get_n_items()

    def __len__(self):
        return self.shared_data.__len__()

    def __getitem__(self, idx):
        return self.shared_data.__getitem__(idx)

