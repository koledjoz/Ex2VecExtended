import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# from ..models.extended.model import Ex2VecExtended
# from ..models.original.model import Ex2VecOriginal

from ..utils import collate_skip_stack_fn


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, start_time, end_time, n_steps):
        self.dataset = dataset
        self.index_data = []
        t_space = np.linspace(start_time, end_time, n_steps)

        for u, items in self.dataset.use_dict.items():
            for i in items:
                for t in t_space:
                    self.index_data.append((u, i, t))

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, idx):
        u, i, t = self.index_data[idx]
        return self.dataset._get_data(u, i, t)


def make_batch(sample: dict):
    """
    Take a single dataset sample (a dict of tensors) and turn it
    into a batch of size 1, just like PyTorch DataLoader would.
    """
    batch = {}
    for key, tensor in sample.items():
        # Ensure it's a tensor and add batch dimension
        if isinstance(tensor, torch.Tensor):
            batch[key] = tensor.unsqueeze(0)
        else:
            raise TypeError(f"Expected tensor for key '{key}', got {type(tensor)}")
    return batch


def prepare_generate_curves(model, data, output_path, run_config, time_begin, time_end, n_steps):
    data_wrap = DatasetWrapper(data, time_begin, time_end, n_steps)

    dataloader = torch.utils.data.DataLoader(data_wrap, batch_size=run_config['batch_size'],
                                             num_workers=run_config['num_workers'], shuffle=run_config['shuffle'],
                                             collate_fn=collate_skip_stack_fn)

    verbose = run_config['verbose'] if 'verbose' in run_config else False

    return {

        "model": model.to(run_config['device']),
        "dataloader": dataloader,
        "device": run_config['device'],
        "verbose": verbose,
        "output_path": output_path
    }


def generate_curves(model, dataloader, device, verbose, output_path):
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(not verbose))

    data_list = []

    with torch.no_grad():
        for i, batch in pbar:
            if batch is None:
                continue
            output = model.forward_batch(batch, device)
            score = output[:, -1].cpu().numpy()[:, None]
            prob = torch.nn.Softmax(dim=1)(output)[:, -1].cpu().numpy()[:, None]
            user = batch['user_id'].cpu().numpy()[:, None]
            item = batch['predict_items'][:, -1].cpu().numpy()[:, None]
            time = batch['predict_ts'].cpu().numpy()[:, None]
            data_list.append(np.concatenate([user, item, time, prob, score], axis=1))

    df = pd.DataFrame(np.concatenate(data_list), columns=['user', 'item', 'ts', 'prob', 'score'])
    df = (
        df.sort_values("ts")  # ensure scores are in time order
        .groupby(["user", "item"])  # group by user & item
        .agg({
            "score": list,
            "prob": list
        })  # collect ordered scores
        .reset_index()
    )
    df.to_csv(output_path, index=False)
