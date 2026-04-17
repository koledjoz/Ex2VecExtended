import torch
from ..base_model import BaseModel


class Ex2VecOriginalFast(BaseModel):
    def __init__(self, config):
        super(Ex2VecOriginalFast, self).__init__()
        self.history_times = None
        self.history_items = None
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

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items + 1, embedding_dim=self.latent_d
        )

        self.logistic = torch.nn.Sigmoid()

    def build_hist_len(self, pad_id=0):
        # Assumes padding is pad_id (0) and (ideally) padded at the end
        hist_len = (self.history_items != pad_id).sum(dim=1).to(torch.long)  # [n_users]
        self.register_buffer("hist_len", hist_len)

    def initialize_histories(self, history_items, history_times):
        self.history_items = history_items
        self.history_times = history_times
        self.build_hist_len()

    # def forward(self, prediction_times, prediction_users):
    #     dist_matrix = torch.norm(self.embedding_item.weight.unsqueeze(0) - self.embedding_user(prediction_users).unsqueeze(1),
    #                              dim=2)
    #
    #     items = torch.arange(self.n_items + 1)
    #
    #     mask_item = (self.history_items[prediction_users].unsqueeze(-1) == items.unsqueeze(0).unsqueeze(0))
    #
    #     negative = prediction_times.unsqueeze(0).unsqueeze(0) - self.history_times[prediction_users].unsqueeze(2)
    #     sign_mask = negative >= 0
    #     a = torch.clamp(negative, min=0)
    #
    #     # unmasked_result = ((item_mask.unsqueeze(0).unsqueeze(2) * a.unsqueeze(1) + self.cutoff) ** -0.5)
    #
    #     unmasked_result = ((sign_mask * a + self.cutoff) ** -0.5)
    #
    #     bl_activation = (
    #             mask_item.unsqueeze(-2)
    #             * unmasked_result.unsqueeze(-1)
    #             * sign_mask.unsqueeze(-1)
    #     ).sum(dim=1)
    #
    #     lamb = (self.global_lamb.clamp_min(0.001) + self.user_lamb(prediction_users).clamp_min(0.001))
    #
    #     base_dist = (dist_matrix[:, None, :] - lamb.unsqueeze(-1) * bl_activation).clamp_min(0)
    #
    #     output = self.alpha * base_dist + self.beta.clamp_min(0.001) * torch.pow(base_dist, 2) + \
    #              self.gamma + self.user_bias(prediction_users)[:, :, None] + self.item_bias.weight.view(1, 1, self.n_items + 1)
    #
    #     return output

    def forward(self, prediction_times, prediction_users):
        # ---- embeddings / distances ----
        user_emb = self.embedding_user(prediction_users)  # [B, d]
        item_emb = self.embedding_item.weight  # [I, d] where I = n_items+1

        # dist_matrix: [B, I]
        # dist_matrix = torch.linalg.vector_norm(item_emb[None, :, :] - user_emb[:, None, :], dim=-1)
        dist_matrix = torch.cdist(user_emb, item_emb)

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
        dt_pos = dt.clamp_min(0)  # in-place clamp

        # rsqrt(x) == x**-0.5 but faster and more stable
        # contrib: [B, H], zeroed where dt < 0
        contrib = (dt_pos + self.cutoff).rsqrt()  # in-place rsqrt
        
        contrib = contrib * (dt > 0).to(contrib.dtype)  # mask negatives to 0

        # print(contrib)
        # for x, y in zip(contrib[0], dt[0]):
        #     print(x, ':', y)
        
        # ---- accumulate to per-item activation via scatter_add ----
        B, H = hist_items.shape
        I = self.n_items + 1

        # print('Hist items shape:', hist_items.shape)
        # print('Contrib shape:', contrib.shape)

        bl_activation = contrib.new_zeros((B, I))  # [B, I]
        bl_activation.scatter_add_(dim=1, index=hist_items, src=contrib)

        #print('='*100)
        #print('Activation shape:', bl_activation.shape)
        #print('Contrib shape', contrib.shape)
        #print('timedeltas shape:', dt.shape)
        #for x, y in zip(contrib[0], dt[0]):
        #    print(x, ':', y)

        #print('='*100)
        #print('ACTIVATIONS FOR ITEMS')
        #print('='*100)
        #for x, y in zip(bl_activation[0], range(self.n_items+1)):
        #    print(x, ':', y)

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
