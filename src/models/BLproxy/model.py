import torch
from ..base_model import BaseModel


class BLProxy(BaseModel):
    def __init__(self, config):
        self.config = config

    def forward(self, timedeltas, weights):
        timedeltas = torch.pow(torch.clamp(timedeltas, min=1e-6), -0.5)
        timedeltas = timedeltas * weights
        return torch.sum(timedeltas, axis=2)

    def forward_batch(self, batch, device):
        timedeltas = batch['timedeltas'].to(device)
        weights = batch['weights'].to(device)

        return self.forward(timedeltas, weights)
