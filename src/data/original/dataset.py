import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import torch
import json

from src.data.utils import sample_excluding


class Ex2VecOriginalDatasetShared:
    def __init__(self, config):
        self.disable_tqdm = config['disable_tqdm']
        self.data_path = config['data_path']
        self.usage_dict_path = config['usage_dict_path']
        self.history_size = config['history_size']
        self.sample_negative = config['sample_negative']

        self.data = pd.read_parquet(self.data_path)

        self.max_user = self.data['user_id'].max()
        self.max_item = self.data['track_id'].max()

        self.data.set_index(['user_id', 'track_id'], inplace=True, drop=False)

        self.max_padding = config['max_padding']

        with open(config['usage_dict_path']) as file:
            self.use_dict = {int(key): set(value) for key, value in json.load(file).items()}

        with h5py.File(config['timedeltas_list_path'], 'r') as f:
            self.offsets = f['offsets'][:]
            self.timestamps_flat = f['timestamps_flat'][:]

            self.pos_dict = {tuple(x): i for i, x in enumerate(tqdm(f['user_item'], disable=self.disable_tqdm))}

            total_size = (self.max_user + 1) * (self.max_item + 1)
            self.pos_array = np.full(total_size, -1, dtype=np.int32)
            for i, (user, item) in enumerate(tqdm(f['user_item'], disable=config['disable_tqdm'])):
                flat_index = user * (self.max_item + 1) + item
                self.pos_array[flat_index] = i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data.iloc[idx]['user_id']
        pred_item = self.data.iloc[idx]['track_id']
        ts = self.data.iloc[idx]['ts']

        if pred_item not in self.use_dict[user]:
            return None

        if self.sample_negative != -1:
            true_vals = np.zeros(self.sample_negative + 1, dtype=np.float32)
            samples = np.empty(self.sample_negative + 1, dtype=np.int32)
            timedeltas = np.zeros((self.sample_negative + 1, self.max_padding), dtype=np.float32)
        else:
            true_vals = np.zeros(self.max_item)
            samples = np.empty(self.max_item, dtype=np.int32)
            timedeltas = np.zeros((self.max_item, self.max_padding), dtype=np.float32)
        true_vals[-1] = 1.0

        samples[:-1] = sample_excluding(self.max_item, self.sample_negative, pred_item)
        samples[-1] = pred_item

        # Vectorized flat index computation
        flat_indices = user * (self.max_item + 1) + samples
        idx_items = self.pos_array[flat_indices]

        valid_mask = idx_items != -1
        valid_indices = np.nonzero(valid_mask)[0]
        valid_pos = idx_items[valid_mask]

        starts = self.offsets[valid_pos, 0]
        lengths = self.offsets[valid_pos, 1]
        ends = starts + lengths

        for i, (start, end, length, sample_idx) in enumerate(zip(starts, ends, lengths, valid_indices)):
            timedeltas[sample_idx, :length] = ts - self.timestamps_flat[start:end]

        weights = timedeltas > 0

        return {
            'user_id': torch.tensor(user),
            'predict_items': torch.tensor(samples),
            'real_values': torch.tensor(true_vals),
            'timedeltas': torch.from_numpy(timedeltas),
            'weights': torch.from_numpy(weights.astype(np.float32))
        }


class Ex2VecOriginalDatasetWrap(torch.utils.data.Dataset):
    def __init__(self, shared_data):
        self.shared_data = shared_data

    def __len__(self):
        return self.shared_data.__len__()

    def __getitem__(self, idx):
        return self.shared_data.__getitem__(idx)
