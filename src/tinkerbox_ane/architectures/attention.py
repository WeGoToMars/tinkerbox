"""Blockwise full attention with stable softmax-mass merging on ANE."""

import torch
from torch import nn
from torch.nn import functional as F


class BlockAttention(nn.Module):
    """Merge SDPA block means by their softmax masses."""

    def __init__(self, cfg, context: int, scale: float = 1.0):
        super().__init__()
        self.context = context
        self.scale = scale
        self.group = cfg.heads // cfg.kv_heads
        self.block_size = 4096

    def forward(self, q, k, v, mask):
        if self.context <= self.block_size:
            return F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(self.group, dim=1),
                v.repeat_interleave(self.group, dim=1),
                attn_mask=mask,
                scale=self.scale,
            )
        batch, heads, tokens, dim = q.shape
        # Block statistics broadcast query groups over shared KV heads instead of repeating keys.
        grouped = q.view(batch, k.shape[1], self.group, tokens, dim)
        maxima, masses, means = [], [], []
        for start in range(0, self.context, self.block_size):
            keys = k[:, :, start:start + self.block_size]
            kb = keys.repeat_interleave(self.group, dim=1)
            vb = v[:, :, start:start + self.block_size].repeat_interleave(self.group, dim=1)
            block_mask = mask[..., start:start + self.block_size]
            scores = (grouped @ keys.unsqueeze(2).transpose(-2, -1)).view(batch, heads, tokens, -1)
            if self.scale != 1.0:
                scores = scores * self.scale
            scores = scores + block_mask
            maximum = scores.amax(-1, keepdim=True)
            maxima.append(maximum)
            masses.append(torch.exp(scores - maximum).sum(-1, keepdim=True))
            means.append(F.scaled_dot_product_attention(q, kb, vb, attn_mask=block_mask, scale=self.scale))
        # Pairwise rescaling avoids a 32K AOT compiler reduction failure.
        weights = []
        for index, local in enumerate(maxima):
            others = [value for j, value in enumerate(maxima) if j != index]
            shift = torch.clamp(local - others[0], max=0)
            for other in others[1:]:
                shift = torch.minimum(shift, local - other)
            weights.append(torch.exp(shift) * masses[index])
        mass = sum(weights)
        return sum((weight / mass) * mean for weight, mean in zip(weights, means))
