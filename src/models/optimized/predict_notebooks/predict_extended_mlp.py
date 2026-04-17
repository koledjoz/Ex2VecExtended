#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

sys.path.append("/home/koledjoz/Ex2VecExtended")


# In[2]:


import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.models.optimized.extendedMLDist import Ex2VecExtendedMLP


# In[3]:


df = pd.read_parquet('../../../../sorted_data.parquet')


# In[4]:


max_history = 3500
user_count = df['user_id'].max()
item_count = df['track_id'].max()


# In[5]:


pos = df.groupby('user_id').cumcount().to_numpy()
users = df['user_id'].to_numpy()
tracks = df['track_id'].to_numpy()
times = df['ts'].to_numpy()

time_history = np.zeros((user_count + 1, max_history), dtype=times.dtype)
item_history = np.zeros((user_count + 1, max_history), dtype=tracks.dtype)

mask = pos < max_history  # safeguard if some users exceed max_history

item_history[users[mask], pos[mask]] = tracks[mask]
time_history[users[mask], pos[mask]] = times[mask]


# In[6]:


checkpoint_path = "../train_notebooks/extendedDoubleMLP.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


# In[7]:


checkpoint['model_state_dict']['embedding_item_extension.weight']


# In[8]:


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

model_bare = Ex2VecExtendedMLP(config)


# In[9]:


model_bare.embedding_item_extension.weight


# In[10]:


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


# In[11]:


import json

with open('../../../../split_data/test/test_dict.json', 'r') as f:
    data = json.load(f)

filter_df = (
    pd.DataFrame(
        [(int(u), t) for u, tracks in data.items() for t in tracks],
        columns=['user_id', 'track_id']
    )
)

# Inner merge keeps only matching pairs
filtered_df = df.merge(filter_df, on=['user_id', 'track_id'], how='inner')


# In[12]:


import torch
from torch.utils.data import DataLoader

data = Ex2VecDataset(filtered_df['ts'].to_numpy(), filtered_df['user_id'].to_numpy(), filtered_df['track_id'].to_numpy())

batch_size = 2**10

# train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

device = 'cpu'
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
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.initialize_histories(torch.tensor(item_history).to(device), torch.tensor(time_history).to(device))
model.to(device)


# In[13]:


model


# In[14]:


top_k = 50
all_batches = []

prediction_cols = [f"pred_{i + 1}" for i in range(top_k)]

with torch.no_grad():
    model.eval()
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for step, batch in enumerate(tqdm(dataloader)):
        model_result = model(batch[0].to(device), batch[1].to(device)).cpu().numpy()

        user_id = batch[1].cpu().numpy()
        item_id = batch[2].cpu().numpy()
        ts = batch[0].cpu().numpy()

        idx = np.argsort(-model_result[:, 1:], axis=1)
        predict_items = idx[:, :top_k] + 1

        u = user_id.reshape(-1)[:, None]  # -> (B, 1)
        i = item_id.reshape(-1)[:, None]  # -> (B, 1)
        t = ts.reshape(-1)[:, None]  # -> (B, 1)
        c = predict_items

        data = np.concatenate([u, i, t, c], axis=1)

        batch_df = pd.DataFrame(data, columns=["userId", "trackId", "ts"] + prediction_cols)
        all_batches.append(batch_df)


    df_preds = pd.concat(all_batches, ignore_index=True)
    # df_preds.to_csv(output_path, index=False)


# In[15]:


df_preds.to_csv('./predictions/extendedMLP/extendedmlp_output.csv', index=False)


# In[ ]:




