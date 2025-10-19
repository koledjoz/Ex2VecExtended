import torch
from abc import ABC, abstractmethod


class BaseModel(torch.nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Forward pass — must be implemented by all subclasses."""
        pass

    @abstractmethod
    def forward_batch(self, batch):
        """Custom method that every model must implement."""
        pass
