# Generated from: extended_base.ipynb
# Converted at: 2026-02-24T22:27:48.575Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import sys

sys.path.append("/home/koledjoz/Ex2VecExtended")

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.models.optimized.extendedWithEmb import Ex2VecExtendedWithEmbFast

df = pd.read_parquet('../../../../sorted_data.parquet')


max_history = 3500
user_count = df['user_id'].max()
item_count = df['track_id'].max()

#time_history = np.zeros((user_count+1, max_history), dtype=int)
#item_history = np.zeros((user_count+1, max_history), dtype=int)

#for user in tqdm(df['user_id'].unique()):
#    tmp = df[df['user_id'] == user]
#    item_history[user, :len(tmp)] = tmp['track_id'].to_numpy()
#    time_history[user, :len(tmp)] = tmp['ts'].to_numpy()

pos = df.groupby('user_id').cumcount().to_numpy()
users = df['user_id'].to_numpy()
tracks = df['track_id'].to_numpy()
times = df['ts'].to_numpy()

time_history = np.zeros((user_count + 1, max_history), dtype=times.dtype)
item_history = np.zeros((user_count + 1, max_history), dtype=tracks.dtype)

mask = pos < max_history  # safeguard if some users exceed max_history

item_history[users[mask], pos[mask]] = tracks[mask]
time_history[users[mask], pos[mask]] = times[mask]



# lets remove some from the history based on our test thingies
import json

with open('../../../../split_data/test/test_dict.json', 'r') as f:
    data = json.load(f)


pairs = {(int(u), i) for u, items in data.items() for i in items}
df = df[~df[['user_id','track_id']].apply(tuple, axis=1).isin(pairs)]


for user, items in data.items():
    row = item_history[int(user)]
    row[np.isin(row, items)] = 0

from torch.utils.data import Dataset

class Ex2VecDataset(Dataset):
    def __init__(self, times, users, items):
        self.times = times
        self.users = users
        self.items = items

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return self.times[idx], self.users[idx], self.items[idx]

import torch
from torch.utils.data import DataLoader

data = Ex2VecDataset(df['ts'].to_numpy(), df['user_id'].to_numpy(), df['track_id'].to_numpy())

batch_size = 2**17

# train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

device = 'cuda'
config = {'n_users': user_count, 'n_items': item_count, 'latent_d': 64,
          'pretrained_embeddings_path': '/home/koledjoz/Ex2VecExtended/split_data/track_embeddings.parquet',
          'item_mapping': '/home/koledjoz/Ex2VecExtended/configs/models/item_mapping.json'}


model = Ex2VecExtendedWithEmbFast(config)
model.initialize_histories(torch.tensor(item_history), torch.tensor(time_history))
model.to(device)


model.to(device)

log_every = 100

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

use_amp = bool(device == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

global_step = 0
history = {"train_loss": [], "val_loss": []}

for epoch in range(100):
    model.train()
    epoch_loss = 0.0
    n = 0
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for step, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch[0].to(device), batch[1].to(device))
            loss = criterion(outputs[:, 1:], batch[2].to(device) - 1)

        # Backprop (AMP-aware)
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        bs = batch[2].shape[0] if torch.is_tensor(batch[2]) and batch[2].ndim > 0 else 1
        epoch_loss += float(loss.item()) * bs
        n += bs
        global_step += 1

        # if (step % log_every == 0):
        #     lr = optimizer.param_groups[0]["lr"]
        #     avg_loss = epoch_loss / max(n, 1)
        #     print(
        #         f"[epoch {epoch}/100] "
        #         f"step {step}/{len(train_dataloader)} "
        #         f"loss={avg_loss:.4f} lr={lr:.2e} "
        #     )
    train_loss = epoch_loss / max(n, 1)
    history["train_loss"].append(train_loss)


    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    ckpt["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(ckpt, './extendedDoubleEmb.pt')
    print(f"Saved final checkpoint to: original.pt")
