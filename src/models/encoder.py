"""VAE Encoder Pytorch Module"""

import torch
from torch import nn
from torch.nn import functional as F
from src.models.vae_blocks import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Module):
    """VAE Encoder"""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                # (Batch_Size, Channel, Height, Width) ->
                #  (Batch_Size, 128, Height, Width)
                nn.Conv2d(3, 128, kernel_size=3, padding=1),

                # (Batch_Size, 128, Height, Width) ->
                #  (Batch_Size, 128, Height, Width)
                VAEResidualBlock(128, 128),

                # (Batch_Size, 128, Height, Width) ->
                #  (Batch_Size, 128, Height, Width)
                VAEResidualBlock(128, 128),

                # (Batch_Size, 128, Height, Width) ->
                #  (Batch_Size, 128, Height / 2, Width / 2)
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

                # (Batch_Size, 128, Height / 2, Width / 2) ->
                #  (Batch_Size, 256, Height / 2, Width / 2)
                VAEResidualBlock(128, 256),

                # (Batch_Size, 256, Height / 2, Width / 2) ->
                #  (Batch_Size, 256, Height / 2, Width / 2)
                VAEResidualBlock(256, 256),

                # (Batch_Size, 256, Height / 2, Width / 2) ->
                #  (Batch_Size, 256, Height / 4, Width / 4)
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

                # (Batch_Size, 256, Height / 4, Width / 4) ->
                #  (Batch_Size, 512, Height / 4, Width / 4)
                VAEResidualBlock(256, 512),

                # (Batch_Size, 512, Height / 4, Width / 4) ->
                #  (Batch_Size, 512, Height / 4, Width / 4)
                VAEResidualBlock(512, 512),

                # (Batch_Size, 512, Height / 4, Width / 4) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                VAEResidualBlock(512, 512),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                VAEResidualBlock(512, 512),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                VAEResidualBlock(512, 512),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                VAEAttentionBlock(512),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                VAEResidualBlock(512, 512),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                nn.GroupNorm(32, 512),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 512, Height / 8, Width / 8)
                nn.SiLU(),

                # (Batch_Size, 512, Height / 8, Width / 8) ->
                #  (Batch_Size, 8, Height / 8, Width / 8).
                nn.Conv2d(512, 8, kernel_size=3, padding=1),

                # (Batch_Size, 8, Height / 8, Width / 8) ->
                #  (Batch_Size, 8, Height / 8, Width / 8)
                nn.Conv2d(8, 8, kernel_size=1, padding=0),
            ]
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Foward method"""
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)
        for layer in self.layers:

            if getattr(layer, 'stride', None) == (2, 2):
                # (Batch_Size, Channel, Height, Width) ->
                # (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))

            x = layer(x)
        # (Batch_Size, 8, Height / 8, Width / 8) -> 2*(Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the variance is between (circa) 1e-14 and 1e8.
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()

        # Transform N(0, 1) -> N(mean, stdev)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise

        # Scale by a constant
        x = 0.18215 * x

        return x
