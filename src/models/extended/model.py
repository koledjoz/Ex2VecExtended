import torch
from ..base_model import BaseModel

class Ex2VecExtended(BaseModel):
    def __init__(self, config):
        super(Ex2VecExtended, self).__init__()
        self.config = config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.latent_d = config['latent_d']
        # model.global_lamb, model.alpha, model.beta, model.gamma, model.cutoff, model.smooth, model.force
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

    def forward(self, user_index, pred_item_indices, history_item_indices, history_timedeltas, history_weights):
        user_emb = self.embedding_user(user_index).unsqueeze(1)

        pred_items_emb = self.embedding_item(pred_item_indices)

        dist_user_item = torch.norm(user_emb - pred_items_emb, dim=2)

        history_items_emb = self.embedding_item(history_item_indices)

        pred_items_emb = pred_items_emb.unsqueeze(1)
        history_items_emb = history_items_emb.unsqueeze(2)

        dist = torch.norm(pred_items_emb - history_items_emb, dim=-1)

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