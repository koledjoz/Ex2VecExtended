# Generated from: extended_base.ipynb
# Converted at: 2026-02-24T22:27:48.575Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import sys

#import torch._dynamo
#torch._dynamo.config.capture_scalar_outputs = True

from torch.utils.tensorboard import SummaryWriter

sys.path.append("/home/koledjoz/Ex2VecExtended")

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.models.optimized.extendedMLDistLoss import Ex2VecExtendedMLP

df = pd.read_parquet('../../../../sorted_data.parquet')


max_history = 3500
user_count = df['user_id'].max()
item_count = df['track_id'].max()

time_history = np.zeros((user_count+1, max_history), dtype=int)
item_history = np.zeros((user_count+1, max_history), dtype=int)

for user in tqdm(df['user_id'].unique()):
    tmp = df[df['user_id'] == user]
    item_history[user, :len(tmp)] = tmp['track_id'].to_numpy()
    time_history[user, :len(tmp)] = tmp['ts'].to_numpy()

# lets remove some from the history based on our test thingies
import json

with open('../../../../split_data/test/test_dict.json', 'r') as f:
    data = json.load(f)





for user, items in data.items():
    np.put(item_history[int(user), :], items, [0,0])

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

batch_size = 2**16

# train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

device = 'cuda'
config = {'n_users': user_count, 'n_items': item_count, 'latent_d': 64,
          'pretrained_embeddings_path': '/home/koledjoz/Ex2VecExtended/split_data/track_embeddings.parquet',
          'item_mapping': '/home/koledjoz/Ex2VecExtended/configs/models/item_mapping.json',
          'mlp_dist_conf': {
            'emb_dim': 128,
            'hidden_dims':[256, 128],
            'block_size':512,
            'activation':torch.nn.ReLU,
            'dropout':0.1,
            'positive_output':True
          }}


model = Ex2VecExtendedMLP(config)
model.initialize_histories(torch.tensor(item_history), torch.tensor(time_history))
model.to(device)


model.to(device)

#model = torch.compile(model, mode="max-autotune")

log_every = 10

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

use_amp = bool(device == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

global_step = 0
history = {"train_loss": [], "val_loss": []}


run_name = "extended_mlp_loss"
writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

# Log config as text
writer.add_text("config", str(config), 0)

checkpoint_path = './extendedDoubleMLP_loss.pt'

for epoch in range(100):
    model.train()
    epoch_loss = 0.0
    n = 0
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{100}")
    for step, batch in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch[0].to(device), batch[1].to(device))
            main_loss = criterion(outputs, batch[2].to(device))

        id_pen = model.metric.identity_penalty(model.embedding_item_extension.weight, kind="l2", reduction="sum")

        lambda_id = 1e-2
        loss = main_loss + lambda_id * id_pen

        # Backprop (AMP-aware)
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        bs = batch[2].shape[0] if torch.is_tensor(batch[2]) and batch[2].ndim > 0 else 1
        # bs = targets.shape[0] if targets.ndim > 0 else 1
        loss_value = float(loss.item())

        epoch_loss += loss_value * bs
        n += bs
        global_step += 1

        lr = optimizer.param_groups[0]["lr"]
        avg_loss = epoch_loss / max(n, 1)

        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

        

        if global_step % log_every == 0:
            writer.add_scalar("train/loss_step", loss_value, global_step)
            writer.add_scalar("train/loss_running_avg", avg_loss, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/id_penalty", id_pen.item(), global_step)
            writer.add_scalar("train/id_loss", lambda_id * id_pen.item(), global_step)
            writer.add_scalar("traom/main_loss", main_loss, global_step)

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

    writer.add_scalar("train/loss_epoch", train_loss, epoch)

    print(f"Epoch {epoch + 1}/{100} - train_loss={train_loss:.6f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")



writer.close()