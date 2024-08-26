"""Test VAE decoder"""

import torch
from src.configs import cfg
from src.models.decoder import VAEDecoder


VAEDecoder_OUTPUT = VAEDecoder()(torch.randn((1, 4,
                                              cfg.LATENTS_HEIGHT,
                                              cfg.LATENTS_WIDTH)))


def test_type_decoder() -> None:
    """Test the VAE decoder output type"""
    assert isinstance(VAEDecoder_OUTPUT, torch.Tensor), \
        f"The model output type {type(VAEDecoder_OUTPUT)}!={torch.Tensor}"


def test_shape_decoder() -> None:
    """Test the VAE decoder output shape"""
    target_shape = (1, 3, cfg.HEIGHT, cfg.WIDTH)
    shape = VAEDecoder_OUTPUT.shape
    assert shape == target_shape, \
        f"The model output shape is {shape}!={target_shape}"
