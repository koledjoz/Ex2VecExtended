import json

import torch
from ..base_model import BaseModel
from ..extendedDouble.model import load_item_extension_from_parquet


class Ex2VecExtendedWithEmbFast(BaseModel):
    def __init__(self, config):
        super(Ex2VecExtendedWithEmbFast, self).__init__()
        self.history_times = None
        self.history_items = None
        self.config = config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.latent_d = config['latent_d']
        pretrained_path = config['pretrained_embeddings_path']

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

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items + 1, embedding_dim=self.latent_d
        )

        self.logistic = torch.nn.Sigmoid()

        self.smooth = torch.nn.Parameter(torch.tensor(1.0))

        self.force = torch.nn.Parameter(torch.tensor(1.0))

        ids, embs = load_item_extension_from_parquet(pretrained_path)

        self.embedding_item_extension = torch.nn.Embedding(
            num_embeddings=self.n_items + 1, embedding_dim=64 if pretrained_path is None else 128
        )

        with open(config['item_mapping'], 'r') as fp:
            mapping = json.load(fp)

        with torch.no_grad():
            for i, item_id in enumerate(ids):
                if str(item_id) not in mapping:
                    continue
                mapped_id = mapping[str(item_id)]
                self.embedding_item_extension.weight[mapped_id] = embs[i]

        # self.item_dist_matrix = torch.cdist(self.embedding_item_extension.weight, self.embedding_item_extension.weight).to('cuda')
        dist = torch.cdist(self.embedding_item_extension.weight,
                           self.embedding_item_extension.weight)
        self.register_buffer("item_dist_matrix", dist)

    def build_hist_len(self, pad_id=0):
        # Assumes padding is pad_id (0) and (ideally) padded at the end
        hist_len = (self.history_items != pad_id).sum(dim=1).to(torch.long)  # [n_users]
        self.register_buffer("hist_len", hist_len)

    def initialize_histories(self, history_items, history_times):
        self.history_items = history_items
        self.history_times = history_times
        self.build_hist_len()

    def forward(self, prediction_times, prediction_users):
        # ---- embeddings / distances ----
        user_emb = self.embedding_user(prediction_users)  # [B, d]
        item_emb = self.embedding_item.weight  # [I, d] where I = n_items+1

        # dist_matrix: [B, I]
        # dist_matrix = torch.linalg.vector_norm(item_emb[None, :, :] - user_emb[:, None, :], dim=-1)
        dist_matrix = torch.cdist(user_emb, item_emb)

        weight = self.logistic(self.smooth / (1 + self.item_dist_matrix) - self.force * self.smooth) / self.logistic(
            self.smooth - self.force * self.smooth)

        item_weight_matrix = (1 / (1 + self.item_dist_matrix) * weight).clamp(0.0, 1.0)

        lengths = self.hist_len[prediction_users]  # [B]
        lmax = int(lengths.max().item())

        # ---- history lookups ----
        hist_items = self.history_items[prediction_users, :lmax]  # [B, H]
        hist_times = self.history_times[prediction_users, :lmax]  # [B, H]

        # prediction_times -> [B, 1]
        t = prediction_times
        if t.dim() == 1:
            t = t[:, None]
        else:
            t = t.view(-1, 1)

        # ---- compute per-history contribution ----
        dt = t - hist_times  # [B, H]
        # only keep dt >= 0
        dt_pos = dt.clamp_min_(0)  # in-place clamp

        # rsqrt(x) == x**-0.5 but faster and more stable
        # contrib: [B, H], zeroed where dt < 0
        contrib = (dt_pos + self.cutoff.clamp_min(0.001)).rsqrt_()  # in-place rsqrt
        contrib = contrib * (dt >= 0).to(contrib.dtype)  # mask negatives to 0



        # ---- accumulate to per-item activation via scatter_add ----
        B, H = hist_items.shape
        I = self.n_items + 1

        bl_activation = contrib.new_zeros((B, I))  # [B, I]
        mask = hist_items != 0
        bl_activation.scatter_add_(dim=1, index=hist_items, src=contrib * mask.to(contrib.dtype))

        bl_activation = bl_activation @ item_weight_matrix

        # ---- lambda ----
        lamb = (self.global_lamb.clamp_min(0.001) +
                self.user_lamb(prediction_users).clamp_min(0.001))  # [B]

        # ---- base distance and output ----
        base_dist = (dist_matrix - lamb * bl_activation).clamp_min_(0)  # [B, I]

        # biases
        user_b = self.user_bias(prediction_users)  # [B, 1]
        item_b = self.item_bias.weight.view(1, -1)  # [1, I]

        output = (self.alpha * base_dist +
                  self.beta.clamp_min(0.001) * (base_dist * base_dist) +
                  self.gamma + user_b + item_b)  # [B, I]

        # If you truly need [B, 1, I] like your original broadcasting suggested:
        # output = output[:, None, :]

        return output

    def forward_batch(self, batch, device):
        pass
