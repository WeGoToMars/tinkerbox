"""Shared FP16 normalization, projections and split feed-forward blocks."""

import math

import torch
from torch import nn


def block_scale(norm_weight: torch.Tensor) -> float:
    typical = 0.66 * norm_weight.float().abs().median().item()
    return float(2 ** max(0, round(math.log2(0.5 / typical))))


def conv1x1(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, bias=False)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, axis: int = 1, input_scale: float = 1.0, compute_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.axis = axis
        self.factor = compute_scale / input_scale
        self.eps_value = eps * compute_scale**2
        self.eps = nn.Buffer(torch.tensor(self.eps_value), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x * self.factor if self.factor != 1.0 else x
        inv = torch.rsqrt((xs * xs).mean(self.axis, keepdim=True) + self.eps)
        shape = [1] * x.dim()
        shape[self.axis] = -1
        return xs * inv * self.weight.view(shape)


def silu(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * (0.5 + 0.5 * torch.tanh((0.5 / scale) * x))


class MLP(nn.Module):
    splits = 8

    def __init__(self, hidden: int, ffn: int, scale: float):
        super().__init__()
        c = ffn // self.splits
        self.w1 = nn.ModuleList(conv1x1(hidden, c) for _ in range(self.splits))
        self.w3 = nn.ModuleList(conv1x1(hidden, c) for _ in range(self.splits))
        self.w2 = nn.ModuleList(conv1x1(c, hidden) for _ in range(self.splits))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for w1, w3, w2 in zip(self.w1, self.w3, self.w2):
            part = w2(silu(w1(x), self.scale) * w3(x))
            out = part if out is None else out + part
        return out
