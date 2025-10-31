import json
import torch
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


class Ex2VecExtendedDouble(BaseModel):
    def __init__(self, config):
        super(Ex2VecExtendedDouble, self).__init__()
        self.config = config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.latent_d = config['latent_d']
        self.global_lamb = torch.nn.Parameter(torch.tensor(1.0))

        self.user_lamb = torch.nn.Embedding(self.n_users + 1, 1)

        self.user_bias = torch.nn.Embedding(self.n_users + 1, 1)
        self.item_bias = torch.nn.Embedding(self.n_items + 1, 1)

        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(-0.065))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))

        self.cutoff = torch.nn.Parameter(torch.tensor(3.0))

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users + 1, embedding_dim=self.latent_d
        )

        self.embedding_item_base = torch.nn.Embedding(
            num_embeddings=self.n_items + 1, embedding_dim=self.latent_d
        )

        self.embedding_item_extension = torch.nn.Embedding(
            num_embeddings=self.n_items + 1, embedding_dim=128
        )

        print(f'Base item emb shape: {self.embedding_item_extension.weight.shape}')
        print(f'Extended item emb shape: {self.embedding_item_extension.weight.shape}')

        self.logistic = torch.nn.Sigmoid()

        self.smooth = torch.nn.Parameter(torch.tensor(1.0))

        self.force = torch.nn.Parameter(torch.tensor(1.0))

        # check if there is a pretrained weight passed
        pretrained_path = config.get('pretrained_embeddings_path', None)
        if pretrained_path:
            ids, embs = load_item_extension_from_parquet(pretrained_path)

            with open(config['item_mapping'], 'r') as fp:
                mapping = json.load(fp)

            with torch.no_grad():
                for i, item_id in enumerate(ids):
                    if str(item_id) not in mapping:
                        continue
                    mapped_id = mapping[str(item_id)]
                    self.embedding_item_extension.weight[mapped_id] = embs[i]

        if config.get("freeze_item_extension", False):
            self.embedding_item_extension.weight.requires_grad_(False)

    def forward(self, user_index, pred_item_indices, history_item_indices, history_timedeltas, history_weights):
        user_emb = self.embedding_user(user_index).unsqueeze(1)

        pred_items_emb_base = self.embedding_item_base(pred_item_indices)
        pred_items_emb_extended = self.embedding_item_extension(pred_item_indices)

        dist_user_item = torch.norm(user_emb - pred_items_emb_base, dim=2)

        history_items_emb_ext = self.embedding_item_extension(history_item_indices) # these are the extensions

        pred_items_emb_extended = pred_items_emb_extended.unsqueeze(1)
        history_items_emb_ext = history_items_emb_ext.unsqueeze(2)

        dist = torch.norm(pred_items_emb_extended - history_items_emb_ext, dim=-1)

        weight = self.logistic(self.smooth / (1 + dist) - self.force * self.smooth) / self.logistic(
            self.smooth - self.force * self.smooth)

        dist = dist * weight

        # dist = self.logistic(self.smooth / (1 + dist) - self.force * self.smooth) / self.logistic(
        #     self.smooth - self.force * self.smooth)

        history_timedeltas = (history_timedeltas + self.cutoff) ** -0.5

        history_timedeltas = history_timedeltas * history_weights

        result = history_timedeltas.unsqueeze(2) * dist

        lamb = self.global_lamb + self.user_lamb(user_index)

        result = lamb.unsqueeze(2) * result

        result = torch.sum(result, axis=1)

        output = torch.maximum(torch.zeros_like(dist_user_item), dist_user_item - result)

        u_bias = self.user_bias(user_index)
        i_bias = self.item_bias(pred_item_indices).squeeze(-1)

        I = self.alpha * output + self.beta * torch.pow(output, 2) + self.gamma + u_bias + i_bias

        return I

    def forward_batch(self, batch, device):
        user_id = batch['user_id'].to(device)
        predict_items = batch['predict_items'].to(device)
        history_items = batch['history_items'].to(device)
        timedeltas = batch['timedeltas'].to(device)
        weights = batch['weights'].to(device)

        return self.forward(user_id, predict_items, history_items, timedeltas, weights)
