"""Test CLIP script"""

import torch
from src.models.clip import CLIP

CLIP_OUTPUT = CLIP()(torch.randint(low=0,
                                   high=49408,
                                   size=(1,77)))

def test_type_clip() -> None:
    """Test the CLIP output type"""
    assert isinstance(CLIP_OUTPUT, torch.Tensor), \
        f"The model output type {type(CLIP_OUTPUT)}!={torch.Tensor}"


def test_shape_clip() -> None:
    """Test the VAE encoder output shape"""
    target_shape = (1, 77, 768)
    shape = CLIP_OUTPUT.shape
    assert shape == target_shape, \
        f"The model output shape is {shape}!={target_shape}"
