import dask.dataframe as dd
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from math import gcd
import time

import json

import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import ReciprocalRank

from sklearn.metrics import ndcg_score



class Ex2Vec(torch.nn.Module):
    def __init__(self, config):
        super(Ex2Vec, self).__init__()
        self.config = config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.latend_d = config['latent_d']

        self.global_lamb = torch.nn.Parameter(torch.tensor(1.0))

        self.user_lamb = torch.nn.Embedding(self.n_users+1, 1)

        self.user_bias = torch.nn.Embedding(self.n_users+1, 1)
        self.item_bias = torch.nn.Embedding(self.n_items+1, 1)

        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(-0.065))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))

        self.cutoff = torch.nn.Parameter(torch.tensor(3.0))

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users+1, embedding_dim=self.latend_d
        )

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items+1, embedding_dim=self.latend_d
        )

        self.logistic = torch.nn.Sigmoid()

        self.smooth = torch.nn.Parameter(torch.tensor(1.0))

        self.force = torch.nn.Parameter(torch.tensor(1.0))


    def forward(self, user_index, pred_item_indices, history_item_indices, history_timedeltas, history_weights):
        user_emb = self.embedding_user(user_index).unsqueeze(1)

        pred_items_emb = self.embedding_item(pred_item_indices)

        dist_user_item = torch.norm(user_emb - pred_items_emb, dim=2)

        history_items_emb = self.embedding_item(history_item_indices)

        pred_items_emb = pred_items_emb.unsqueeze(1)
        history_items_emb = history_items_emb.unsqueeze(2)

        dist = torch.norm(pred_items_emb - history_items_emb, dim=-1)

        dist = self.logistic(self.smooth / (1 + dist) - self.force * self.smooth) / self.logistic(self.smooth - self.force * self.smooth)

        history_timedeltas = (history_timedeltas + self.cutoff) ** -0.5

        history_timedeltas = history_timedeltas * history_weights

        result = history_timedeltas.unsqueeze(2) * dist

        lamb = self.global_lamb + self.user_lamb(user_index)

        result = lamb.unsqueeze(2) * result
        
        result = torch.sum(result, axis=1)
        
        output = torch.maximum(torch.zeros_like(dist_user_item), dist_user_item - result)

        u_bias = self.user_bias(user_index)
        i_bias = self.item_bias(pred_item_indices).squeeze(-1)

        I = self.alpha * output  + self.beta * torch.pow(output, 2) + self.gamma + u_bias + i_bias
      
        return I



def sample_excluding(n, x, a):
    if x == -1:
        return [num for num in range(1,n+1) if num != a ]
    
    if x > n - 1:
        raise ValueError("Cannot sample more elements than available excluding 'a'")
    
    # Sample x numbers from 1 to n-1
    sampled = random.sample(range(1, n), x)

    # Map values >= a to skip 'a'
    return [num if num < a else num + 1 for num in sampled]
# ok, lets look at what we need right here and there


class DeezerDataset(Dataset):
    def __init__(self, data_path, usage_dict_path, grouping_size=1000, sample_negative=999, history_size=128, seed=None):
        print('Starting dataset initialization')

        self.grouping_size = grouping_size

        with open(usage_dict_path) as file:
            self.use_dict = {int(key) : set(value) for key, value in json.load(file).items()}
        
        self.data = pd.read_parquet(data_path)

        self.sample_negative = sample_negative
        self.history_size=128

        self.seed = seed

        self.max_user = self.data['user_id'].max()
        self.max_item = self.data['track_id'].max()

        print('Dataset initialized')


    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        # t0 = time.time()

        pred_user_id = self.data.iloc[idx]['user_id']
        pred_item = self.data.iloc[idx]['track_id']

        if pred_item not in self.use_dict[pred_user_id]:
            return None



        
        pred_items = np.append(np.array(sample_excluding(self.max_item, self.sample_negative, pred_item)), pred_item)
        true_vals = np.append(np.array([0.0 for _ in range(len(pred_items)-1)]), 1.0)

        history = self.data.iloc[max(idx-self.history_size, 0):idx]
        history = history[history['user_id'] == pred_user_id]

        ts = self.data.iloc[idx]['ts']
        timedeltas = (ts - history['ts']).to_numpy()

        history_items = history['track_id'].to_numpy()
        weights = np.ones_like(history_items)

        # timedeltas = np.pad(timedeltas, (0, self.history_size-len(timedeltas)))
        # history_items = np.pad(history_items, (0, self.history_size-len(history_items)))
        # weights = np.pad(weights, (0, self.history_size-len(weights)))

        timedeltas = np.pad(timedeltas, (0, self.history_size - len(timedeltas)), mode='constant', constant_values=0)
        history_items = np.pad(history_items, (0, self.history_size - len(history_items)), mode='constant', constant_values=0)
        weights = np.pad(weights, (0, self.history_size - len(weights)), mode='constant', constant_values=0)

        
        return {
            'user_id' : torch.tensor(pred_user_id),
            'predict_items' : torch.tensor(pred_items),
            'real_values' : torch.tensor(true_vals),
            'history_items' : torch.tensor(history_items),
            'timedeltas' : torch.tensor(timedeltas),
            'weights' : torch.tensor(weights)
        }


dataset_train = DeezerDataset('sorted_data.parquet', 'train_dict.json')
dataset_val = DeezerDataset('sorted_data.parquet', 'val_dict.json', sample_negative=-1)

def collate_fn(batch):
    # Remove None entries
    batch = [x for x in batch if x is not None]
    
    if not batch:
        return None  # Signal to skip this batch
    
    # Stack each field in the batch
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = torch.stack([sample[key] for sample in batch])
    
    return collated_batch

loader_train = DataLoader(dataset_train, batch_size=192, num_workers = 32, shuffle=True, collate_fn=collate_fn)
loader_val = DataLoader(dataset_val, batch_size=192, num_workers = 32, shuffle=True, collate_fn=collate_fn)

writer = SummaryWriter(log_dir="runs/exp1")


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

config = {
    'n_users' : dataset_train.max_user,
    'n_items' : dataset_train.max_item,
    'latent_d' : 64
}


start_epoch = 0
model = Ex2Vec(config).to(device) # change this if something works


epoch_count = 10


loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

running_loss = 0.
last_loss = 0.


losses = []

for epoch in range(epoch_count):
    print(f'Running epoch {epoch}')

    pbar_train = tqdm(enumerate(loader_train), total=len(loader_train))


    train_instances = 0
    
    for i, batch in pbar_train:
        model.train()
        if batch is None:
            pbar_train.update(1)
            continue
            
        optimizer.zero_grad()

        real = batch['real_values'].to(device)
        user_id = batch['user_id'].to(device)
        predict_items = batch['predict_items'].to(device)
        history_items = batch['history_items'].to(device)
        timedeltas = batch['timedeltas'].to(device)
        weights = batch['weights'].to(device)
        
        output = model(user_id, predict_items, history_items, timedeltas, weights)
        loss = loss_fn(output, real)

        loss.backward()

        optimizer.step()

        loss_item = loss.item()
        pbar_train.set_description(f'Batch loss: {loss_item}')
        pbar_train.update(1)
        losses.append(loss_item)
        running_loss += loss_item
        train_instances += real.shape[0]


        global_step = epoch * len(loader_train) + i
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], global_step)


    # # lets do the validation, so lets goooo
    pbar_val = tqdm(enumerate(loader_val), total=len(loader_val))

    val_loss = 0.0
    val_instances = 0
    model.eval()

    rr_metric = ReciprocalRank()

    ncdg_final = 0.0

    with torch.no_grad():
        for i, batch in pbar_val:
            if batch is None:
                pbar_val.update(1)
                continue
                
            real = batch['real_values'].to(device)
            user_id = batch['user_id'].to(device)
            predict_items = batch['predict_items'].to(device)
            history_items = batch['history_items'].to(device)
            timedeltas = batch['timedeltas'].to(device)
            weights = batch['weights'].to(device)

            output = model(user_id, predict_items, history_items, timedeltas, weights)

            rr_metric.update(output, torch.argmax(real, dim=1))

            ncdg_sum = ndcg_score(real.cpu(), output.cpu(), k=1000)
            
            loss = loss_fn(output, real)


            
            loss_item = loss.item()
            pbar_val.set_description(f'Batch loss: {loss_item:.04f}; Batch MRR: {rr_metric.compute().mean().cpu().numpy():.04f}; Batch NCDG: {ncdg_sum:.04f}')
            pbar_val.update(1)

            val_loss += loss_item * real.shape[0]
            val_instances += real.shape[0]
            ncdg_sum = ncdg_sum * real.shape[0]
            ncdg_final += ncdg_sum


    mrr = rr_metric.compute().mean().cpu().numpy()

    
    
    val_loss = val_loss / val_instances

    global_step = epoch * len(loader_train)
    writer.add_scalar("Loss/val", val_loss, global_step)
    writer.add_scalar("Loss/MRR", mrr, global_step)
    writer.add_scalar("Loss/NCDG", ncdg_final / val_instances, global_step)
        
    last_loss = running_loss / train_instances # loss per epoch
    print('   epoch {} loss: {}'.format(epoch, last_loss))
    running_loss = 0.
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'model_epoch_{epoch+start_epoch}.pt')

