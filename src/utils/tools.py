"""Utils functions"""

from typing import Literal
import torch


def get_device(allow_cuda: bool, allow_mps: bool
               ) -> Literal["cpu", "cuda", "mps"]:
    """Get the deivce"""
    if torch.cuda.is_available() and allow_cuda:
        return "cuda"
    if (torch.backends.mps.is_built() or
            torch.backends.mps.is_available()) and allow_mps:
        return "mps"
    return "cpu"
