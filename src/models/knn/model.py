import torch
import json
import pyarrow.parquet as pq
from ..base_model import BaseModel

def load_item_extension_from_parquet(path, id_col="track_id", emb_col="vector"):
    tbl = pq.read_table(path)

    ids = tbl[id_col].to_pylist()
    raw = tbl[emb_col].to_pylist()

    if any(r is None for r in raw):
        bad = [i for i, r in enumerate(raw) if r is None][:5]
        raise ValueError(f"Found None in embedding column at rows (first 5): {bad}")

    embs = []
    for row in raw:
        vec = [d["item"] for d in row["list"]]
        embs.append(vec)

    embs = torch.tensor(embs, dtype=torch.float32)

    return ids, embs

class KNNModelBase(BaseModel):
    def __init__(self, config):
        super(KNNModelBase, self).__init__()
        self.config = config
        self.n_items = config['n_items']


        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items + 1, embedding_dim=128
        )

        pretrained_path = config.get('pretrained_embeddings_path', None)
        ids, embs = load_item_extension_from_parquet(pretrained_path)

        with open(config['item_mapping'], 'r') as fp:
            mapping = json.load(fp)

        with torch.no_grad():
            for i, item_id in enumerate(ids):
                if str(item_id) not in mapping:
                    continue
                mapped_id = mapping[str(item_id)]
                self.embedding_item.weight[mapped_id] = embs[i]

        self.embedding_item.weight.requires_grad_(False)

    def forward(self, pred_item_indices, history_items_indices, history_weights):
        hist_item_emb = self.embedding_item(history_items_indices)
        hist_item_emb = hist_item_emb * history_weights
        sum_emb = hist_item_emb.sum(dim=1)  # (batch, d)
        count = hist_item_emb.sum(dim=1).clamp(min=1e-8)
        user_emb = sum_emb / count

        pred_item_emb = self.embedding_item(pred_item_indices)

        user_emb_expanded = user_emb.unsqueeze(1).expand_as(pred_item_emb)

        diff = user_emb_expanded - pred_item_emb
        dists = torch.norm(diff, dim=-1)
        scores = -dists

        return scores


    def forward_batch(self, batch, device):
        predict_items = batch['predict_items'].to(device)
        history_items = batch['history_items'].to(device)
        weights = batch['weights'].to(device)

        return self.forward(predict_items, history_items, weights)



class KNNModelBL(BaseModel):
    def __init__(self, config):
        super(KNNModelBL, self).__init__()
        self.config = config
        self.n_items = config['n_items']


        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items + 1, embedding_dim=128
        )

        pretrained_path = config.get('pretrained_embeddings_path', None)
        ids, embs = load_item_extension_from_parquet(pretrained_path)

        with open(config['item_mapping'], 'r') as fp:
            mapping = json.load(fp)

        with torch.no_grad():
            for i, item_id in enumerate(ids):
                if str(item_id) not in mapping:
                    continue
                mapped_id = mapping[str(item_id)]
                self.embedding_item.weight[mapped_id] = embs[i]

        self.embedding_item.weight.requires_grad_(False)

    def forward(self, pred_item_indices, history_items_indices, history_timedeltas, history_weights):
        timedeltas = torch.log(torch.pow(torch.clamp(history_timedeltas, min=1e-6), -0.5))
        timedeltas = timedeltas * history_weights


        hist_item_emb = self.embedding_item(history_items_indices)
        hist_item_emb = hist_item_emb * timedeltas
        sum_emb = hist_item_emb.sum(dim=1)  # (batch, d)
        count = timedeltas.sum(dim=1).clamp(min=1e-8)
        user_emb = sum_emb / count

        pred_item_emb = self.embedding_item(pred_item_indices)

        user_emb_expanded = user_emb.unsqueeze(1).expand_as(pred_item_emb)

        diff = user_emb_expanded - pred_item_emb
        dists = torch.norm(diff, dim=-1)
        scores = -dists

        return scores


    def forward_batch(self, batch, device):
        predict_items = batch['predict_items'].to(device)
        history_items = batch['history_items'].to(device)
        timedeltas = batch['timedeltas'].to(device)
        weights = batch['weights'].to(device)

        return self.forward(predict_items, history_items, timedeltas, weights)