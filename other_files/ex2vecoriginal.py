import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import ReciprocalRank

from sklearn.metrics import ndcg_score

import pandas as pd
from tqdm import tqdm

import random

import numpy as np

import json

import h5py


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

class Ex2VecDataset(Dataset):
    def __init__(self, data_path, usage_dict_path, timedeltas_list_path, history_size=3500, sample_negative=999, max_padding=256):
        self.data_path = data_path
        self.usage_dict_path = usage_dict_path
        self.history_size = history_size
        self.sample_negative = sample_negative

        self.data = pd.read_parquet(self.data_path)

        self.max_user = self.data['user_id'].max()
        self.max_item = self.data['track_id'].max()

        self.data.set_index(['user_id', 'track_id'], inplace=True, drop=False)

        self.max_padding=max_padding

        with open(usage_dict_path) as file:
            self.use_dict = {int(key) : set(value) for key, value in json.load(file).items()}

        with h5py.File(timedeltas_list_path, 'r') as f:
            self.offsets = f['offsets'][:]
            self.timestamps_flat = f['timestamps_flat'][:]

            self.pos_dict = {tuple(x) : i for i,x in enumerate(tqdm(f['user_item']))}
    
            total_size = (self.max_user + 1) * (self.max_item + 1)
            self.pos_array = np.full(total_size, -1, dtype=np.int32)
            for i, (user, item) in enumerate(tqdm(f['user_item'])):
                flat_index = user * (self.max_item + 1) + item
                self.pos_array[flat_index] = i

        

    def __len__(self):
        return len(self.data)

    # @profile
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
            'user_id' : torch.tensor(user),
            'predict_items' : torch.tensor(samples),
            'real_values' : torch.tensor(true_vals),
            'timedeltas' : torch.from_numpy(timedeltas),
            'weights' : torch.from_numpy(weights.astype(np.float32))
        }
        
# @profile
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

class Ex2VecOriginal(torch.nn.Module):
    def __init__(self, config):
        super(Ex2VecOriginal, self).__init__()
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

    
    def forward(self, user_id, item_id, timedeltas, weights):
        user_emb = self.embedding_user(user_id).unsqueeze(1)
        item_emb = self.embedding_item(item_id)

        u_bias = self.user_bias(user_id)
        i_bias = self.item_bias(item_id).squeeze(-1)

        # base_dist = torch.sqrt((user_emb - item_emb)**2).sum(dim=2)
        base_dist = torch.norm(user_emb - item_emb, dim=-1)

        lamb = self.global_lamb + self.user_lamb(user_id).unsqueeze(-1)


        # print('ORIG TIMEDELTAS', timedeltas)

        # print('PRE POW DELTAS', (timedeltas + self.cutoff) * weights)

        timedeltas = torch.pow(torch.clamp(timedeltas + self.cutoff, min=1e-6), -0.5)
        # print('TIMEDELTAS 1', timedeltas)
        # if torch.isnan(timedeltas).any():
        #     print(f'crashing {15324/0}')

        timedeltas = timedeltas * weights
        # print('TIMEDELTAS 2', timedeltas)

        # if torch.isnan(timedeltas).any():
        #     print(f'crashing {15324/0}')

        timedeltas = timedeltas * weights

        base_level = lamb * timedeltas

        # print('LAMB', lamb)
        # print('TIMEDELTAS', timedeltas)

        # if torch.isnan(base_level).any():
        #     print('NONE FOUND BASE LEVEL 1 ', base_level)
        #     print(f'crashing {15324/0}')

        # print('PRE SUM', base_level)

        base_level = torch.sum(base_level, axis=2)

        # if torch.isnan(base_level).any:
        #     print('NONE FOUND BASE LEVEL 2 ', base_level)
        #     print(f'crashing {15324/0}')

        output = torch.maximum(torch.zeros_like(base_dist), base_dist - base_level)

        # if torch.isnan(output).any():
        #     print('NONE FOUND OUTPUT', output)
        #     print(f'crashing {15324/0}')

        I = self.alpha * output  + self.beta * torch.pow(output, 2) + self.gamma + u_bias + i_bias
      
        return I

# @profile
def train_model():
    
    dataset_train = Ex2VecDataset('sorted_data.parquet', 'train_dict.json', 'interactions.h5')
    dataset_val = Ex2VecDataset('sorted_data.parquet', 'val_dict.json', 'interactions.h5', sample_negative=-1)
    
    loader_train = DataLoader(dataset_train, batch_size=1024, num_workers = 8, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=1024, num_workers = 8, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    writer = SummaryWriter(log_dir="runs/Ex2VecOriginal")
    
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'n_users' : dataset_train.max_user,
        'n_items' : dataset_train.max_item,
        'latent_d' : 64
    }
    
    
    # start_epoch = 0
    start_epoch = 30
    model = Ex2VecOriginal(config).to(device) # change this if something works
    
    
    
    epoch_count = 10
    
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    checkpoint = torch.load('orig_model_epoch_29.pt', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    running_loss = 0.
    last_loss = 0.
    
    
    losses = []
    
    for epoch in range(epoch_count):
        epoch = epoch + start_epoch
        print(f'Running epoch {epoch}')
    
        pbar_train = tqdm(enumerate(loader_train), total=len(loader_train))
    
    
        train_instances = 0
    
        model.train()
        
        for i, batch in pbar_train:
    
            # if i == 100:
            #     break
            
            if batch is None:
                pbar_train.update(1)
                continue
                
            optimizer.zero_grad()
    
            real = batch['real_values'].to(device)
            user_id = batch['user_id'].to(device)
            predict_items = batch['predict_items'].to(device)
            timedeltas = batch['timedeltas'].to(device)
            weights = batch['weights'].to(device)
    
            output = model(user_id, predict_items, timedeltas, weights)
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
    
    
    
        last_loss = running_loss / train_instances # loss per epoch
        print('   epoch {} loss: {}'.format(epoch, last_loss))
    
        
        # # lets do the validation, so lets goooo
        pbar_val = tqdm(enumerate(loader_val), total=len(loader_val))
    
        val_loss = 0.0
        val_instances = 0
        model.eval()
    
        # rr_metric = ReciprocalRank()
    
        with torch.no_grad():
            for i, batch in pbar_val:

                # if i == 100:
                #     break
                
                if batch is None:
                    pbar_val.update(1)
                    continue
                    
                real = batch['real_values'].to(device)
                user_id = batch['user_id'].to(device)
                predict_items = batch['predict_items'].to(device)
                timedeltas = batch['timedeltas'].to(device)
                weights = batch['weights'].to(device)
    
                output = model(user_id, predict_items, timedeltas, weights)
    
                # rr_metric.update(output, torch.argmax(real, dim=1))
    
                # ncdg_sum = ndcg_score(real.cpu(), output.cpu(), k=1000)
                
                loss = loss_fn(output, real)
    
    
                
                loss_item = loss.item()
                # pbar_val.set_description(f'Batch loss: {loss_item:.04f}; Batch MRR: {rr_metric.compute().mean().cpu().numpy():.04f}; Batch NCDG: {ncdg_sum:.04f}')
                pbar_val.set_description(f'Batch loss: {loss_item:.04f}')
                pbar_val.update(1)
    
                val_loss += loss_item
                val_instances += real.shape[0]
                # ncdg_sum = ncdg_sum * real.shape[0]
    
    
        # mrr = rr_metric.compute().mean().cpu().numpy()
    
        
        val_loss = val_loss / val_instances
    
        global_step = epoch * len(loader_train)
        writer.add_scalar("Loss/val", val_loss, global_step)
        # writer.add_scalar("Loss/MRR", mrr, global_step)
        # writer.add_scalar("Loss/NCDG", ncdg_sum / val_instances, global_step)
            
        last_loss = running_loss / train_instances # loss per epoch
        print('   epoch {} val loss: {}'.format(epoch, val_loss))
        running_loss = 0.
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'orig_model_epoch_{epoch}.pt')

if __name__ == '__main__':
    train_model()