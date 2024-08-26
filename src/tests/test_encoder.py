"""Test VAE encoder"""

import torch
from src.configs import cfg
from src.models.encoder import VAEEncoder

VAEEncoder_OUTPUT = VAEEncoder()(torch.rand((1, 3,
                                             cfg.HEIGHT,
                                             cfg.WIDTH)),
                                 torch.randn((1, 4,
                                              cfg.LATENTS_HEIGHT,
                                              cfg.LATENTS_WIDTH)))


def test_type_encoder() -> None:
    """Test the VAE encoder output type"""
    assert isinstance(VAEEncoder_OUTPUT, torch.Tensor), \
        f"The model output type {type(VAEEncoder_OUTPUT)}!={torch.Tensor}"


def test_shape_encoder() -> None:
    """Test the VAE encoder output shape"""
    target_shape = (1, 4,
                    cfg.LATENTS_HEIGHT,
                    cfg.LATENTS_WIDTH)
    shape = VAEEncoder_OUTPUT.shape
    assert shape == target_shape, \
        f"The model output shape is {shape}!={target_shape}"
