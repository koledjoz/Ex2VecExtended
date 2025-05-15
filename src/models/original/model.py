import torch


class Ex2VecOriginal(torch.nn.Module):
    def __init__(self, config):
        super(Ex2VecOriginal, self).__init__()
        self.config = config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.latend_d = config['latent_d']

        self.global_lamb = torch.nn.Parameter(torch.tensor(1.0))

        self.user_lamb = torch.nn.Embedding(self.n_user s +1, 1)

        self.user_bias = torch.nn.Embedding(self.n_user s +1, 1)
        self.item_bias = torch.nn.Embedding(self.n_item s +1, 1)

        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(-0.065))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))

        self.cutoff = torch.nn.Parameter(torch.tensor(3.0))

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users +1, embedding_dim=self.latend_d
        )

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items +1, embedding_dim=self.latend_d
        )

        self.logistic = torch.nn.Sigmoid()


    def forward(self, user_id, item_id, timedeltas, weights):
        user_emb = self.embedding_user(user_id).unsqueeze(1)
        item_emb = self.embedding_item(item_id)

        u_bias = self.user_bias(user_id)
        i_bias = self.item_bias(item_id).squeeze(-1)

        base_dist = torch.norm(user_emb - item_emb, dim=-1)

        lamb = self.global_lamb + self.user_lamb(user_id).unsqueeze(-1)


        timedeltas = torch.pow(torch.clamp(timedeltas + self.cutoff, min=1e-6), -0.5)


        timedeltas = timedeltas * weights

        timedeltas = timedeltas * weights

        base_level = lamb * timedeltas

        base_level = torch.sum(base_level, axis=2)

        output = torch.maximum(torch.zeros_like(base_dist), base_dist - base_level)

        I = self.alpha * output  + self.beta * torch.pow(output, 2) + self.gamma + u_bias + i_bias

        return I