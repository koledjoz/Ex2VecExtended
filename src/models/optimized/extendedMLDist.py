import json

import torch



import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.extendedDouble.model import load_item_extension_from_parquet


class FastSymmetricPairwiseMLP(nn.Module):
    """
    Fast symmetric pairwise scorer using features [|x-y|, x*y].

    Optimizations:
    - computes only upper-triangular blocks, mirrors to lower triangle
    - fuses first linear layer, so no explicit concat([absdiff, prod])
    - blockwise computation to control memory
    - works with autograd

    Output:
        scores: (N, N), symmetric
    """
    def __init__(
        self,
        emb_dim: int,
        hidden_dims=(256, 128),
        activation=nn.ReLU,
        dropout: float = 0.0,
        block_size: int = 256,
        positive_output: bool = False,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.block_size = block_size
        self.positive_output = positive_output

        hidden_dims = list(hidden_dims)
        assert len(hidden_dims) >= 1, "Need at least one hidden layer"

        # First layer is special: it consumes [|x-y|, x*y] without explicit concat
        self.first = nn.Linear(2 * emb_dim, hidden_dims[0])

        # Remaining MLP
        layers = []
        in_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)

        self.act = activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.out_act = nn.Softplus()

    def _first_layer_fused(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        """
        xi: (Bi, D)
        xj: (Bj, D)
        returns: (Bi, Bj, H)
        """
        D = self.emb_dim
        W = self.first.weight
        b = self.first.bias

        # Split first-layer weights into the two feature groups:
        # [|x-y|, x*y]
        W_abs = W[:, :D]
        W_prod = W[:, D:]

        # Pairwise features, blockwise
        delta = xi[:, None, :] - xj[None, :, :]  # MODIFIED
        absdiff = (xi[:, None, :] - xj[None, :, :]).abs()   # (Bi, Bj, D)
        prod = xi[:, None, :] * xj[None, :, :]              # (Bi, Bj, D)

        # Fused first linear layer without explicit concatenation
        h = F.linear(absdiff, W_abs) + F.linear(prod, W_prod, b)  # (Bi, Bj, H)

        h = self.act(h)
        if self.dropout is not None:
            h = self.dropout(h)

        return h, delta   # MODIFIED

    def _mlp_tail(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (..., H)
        returns: (..., 1)
        """
        if len(self.hidden) > 0:
            h = self.hidden(h)
        h = self.out(h)
        h = self.out_act(h)
        return h

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (N, D)
        returns: (N, N) symmetric score matrix
        """
        N, D = emb.shape
        assert D == self.emb_dim

        out = emb.new_empty((N, N))

        bs = self.block_size
        for i in range(0, N, bs):
            i2 = min(i + bs, N)
            xi = emb[i:i2]   # (Bi, D)

            for j in range(i, N, bs):
                j2 = min(j + bs, N)
                xj = emb[j:j2]   # (Bj, D)

                # First fused layer on block pair
                h, delta = self._first_layer_fused(xi, xj)   # MODIFIED

                # Tail MLP
                scale = self._mlp_tail(h).squeeze(-1)

                base = torch.linalg.norm(delta, dim=-1)  # MODIFIED: (Bi, Bj)

                scores = base * scale  # MODIFIED: exact zero diagonal

                out[i:i2, j:j2] = scores
                if i != j:
                    out[j:j2, i:i2] = scores.transpose(0, 1)

        return out



class Ex2VecExtendedMLP(BaseModel):
    def __init__(self, config):
        super(Ex2VecExtendedMLP, self).__init__()
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

        self.embedding_item_extension.requires_grad = False



        self.metric = FastSymmetricPairwiseMLP(**self.config['mlp_dist_conf'])

        self.metric = torch.compile(self.metric, dynamic=False)
        #     emb_dim=64 if pretrained_path is None else 128,
        #     hidden_dims=[256, 128],
        #     block_size=512,
        #     activation=nn.ReLU,
        #     dropout=0.1,  # keep 0 for max speed
        #     positive_output=True  # set True if you want nonnegative "distance"
        # )
            # # self.item_dist_matrix = torch.cdist(self.embedding_item_extension.weight, self.embedding_item_extension.weight).to('cuda')
            # dist = torch.cdist(self.embedding_item_extension.weight,
            #                    self.embedding_item_extension.weight)
        # self.register_buffer("item_dist_matrix", dist)

    def build_hist_len(self, pad_id=0):
        # Assumes padding is pad_id (0) and (ideally) padded at the end
        hist_len = (self.history_items != pad_id).sum(dim=1).to(torch.long)  # [n_users]
        self.register_buffer("hist_len", hist_len)

    def initialize_histories(self, history_items, history_times):
        self.register_buffer("history_items", history_items)
        self.register_buffer("history_times", history_times)
        self.build_hist_len()


    def forward(self, prediction_times, prediction_users):
        item_dist_matrix = self.metric(self.embedding_item_extension.weight)

        user_emb = self.embedding_user(prediction_users)  # [B, d]
        item_emb = self.embedding_item.weight  # [I, d] where I = n_items+1

        # dist_matrix: [B, I]
        # dist_matrix = torch.linalg.vector_norm(item_emb[None, :, :] - user_emb[:, None, :], dim=-1)
        dist_matrix = torch.cdist(user_emb, item_emb)

        weight = self.logistic(self.smooth / (1 + item_dist_matrix) - self.force * self.smooth) / self.logistic(
            self.smooth - self.force * self.smooth)

        item_weight_matrix = (1 / (1 + item_dist_matrix) * weight).clamp(0.0, 1.0)
        item_weight_matrix[0, :] = 0
        item_weight_matrix[:, 0] = 0


        lengths = self.hist_len[prediction_users]  # [B]
        lmax = int(lengths.max().item())

        # ---- history lookups ----
        hist_items = self.history_items[prediction_users, :lmax]  # [B, H]
        hist_times = self.history_times[prediction_users, :lmax].float()  # [B, H]

        # prediction_times -> [B, 1]
        t = prediction_times.float()
        if t.dim() == 1:
            t = t[:, None]
        else:
            t = t.view(-1, 1)

        # ---- compute per-history contribution ----
        dt = t - hist_times  # [B, H]
        # only keep dt >= 0
        dt_pos = dt.clamp_min(0)  # in-place clamp

        # rsqrt(x) == x**-0.5 but faster and more stable
        # contrib: [B, H], zeroed where dt < 0
        contrib = (dt_pos + self.cutoff.clamp_min(0.001)).rsqrt()  # in-place rsqrt
        contrib = contrib * (dt > 0).to(contrib.dtype)  # mask negatives to 0



        # ---- accumulate to per-item activation via scatter_add ----
        B, H = hist_items.shape
        I = self.n_items + 1

        bl_activation = contrib.new_zeros((B, I))  # [B, I]
        # mask = hist_items != 0
        bl_activation.scatter_add_(dim=1, index=hist_items, src=contrib)# * mask.to(contrib.dtype))

        bl_activation = bl_activation @ item_weight_matrix

        # ---- lambda ----
        lamb = (self.global_lamb +
                self.user_lamb(prediction_users)).clamp_min(0.001)  # [B]

        # ---- base distance and output ----
        base_dist = (dist_matrix - lamb * bl_activation).clamp_min(0)  # [B, I]

        # biases
        user_b = self.user_bias(prediction_users)  # [B, 1]
        item_b = self.item_bias.weight.view(1, -1)  # [1, I]

        output = (self.alpha * base_dist +
                  self.beta.clamp_max(-0.001) * (base_dist * base_dist) +
                  self.gamma + user_b + item_b)  # [B, I]

        # If you truly need [B, 1, I] like your original broadcasting suggested:
        # output = output[:, None, :]

        return output

    def forward_batch(self, batch, device):
        pass
