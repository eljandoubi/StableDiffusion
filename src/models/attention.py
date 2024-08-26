"""Attention mechanisms"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """Self Attention"""

    def __init__(self, n_heads: int,
                 d_embed: int,
                 in_proj_bias: bool = True,
                 out_proj_bias: bool = True
                 ):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor,
                causal_mask: bool = False) -> torch.Tensor:
        """Foward method"""
        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, _ = input_shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        inter_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3)
        qkv: torch.Tensor = self.in_proj(x)

        #  -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) ->
        # (Batch_Size, Seq_Len, H, Dim / H) ->
        # (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(inter_shape).transpose(1, 2)
        k = k.view(inter_shape).transpose(1, 2)
        v = v.view(inter_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim / H) @
        # (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len, Seq_Len) @
        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output
