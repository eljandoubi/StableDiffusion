"""Test VAE"""

from typing import Literal, Optional
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from src.models.vae import VAE
from src.utils.tools import get_device
from src.configs import cfg


def test_vae(img_path: str = "samples/EiffelTower.jpg",
             device: Optional[Literal["cpu", "cuda", "mps"]] = None
             ) -> bool:
    """Test the VAE shape on an image"""
    if device is None:
        device = get_device(allow_cuda=cfg.ALLOW_CUDA,
                            allow_mps=cfg.ALLOW_MPS)

    img = Image.open(img_path)
    img = img.resize((cfg.WIDTH, cfg.HEIGHT))
    input_tensor = pil_to_tensor(img).float() / 255
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    vae = VAE().to(device)
    noise = torch.randn((1, 4,
                         cfg.LATENTS_HEIGHT,
                         cfg.LATENTS_WIDTH))
    out_tensor: torch.Tensor = vae(input_tensor, noise)

    assert out_tensor.shape == input_tensor.shape,\
          "VAE model must retour the same shape as its input"
