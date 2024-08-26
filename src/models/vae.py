"""VAE pytorch module"""

import torch
from torch import nn
from src.models.decoder import VAEDecoder
from src.models.encoder import VAEEncoder

class VAE(nn.Module):
    """VAE"""
    def __init__(self):
        super().__init__()

        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Foward method"""
        x = self.encoder(x,noise)
        x = self.decoder(x)
        return x
