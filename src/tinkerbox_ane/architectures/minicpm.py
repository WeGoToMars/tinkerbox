"""MiniCPM attention blocks shared by causal LMs and local audio transformers."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tinkerbox_ane.blocks import CompositeRMSNorm, conv1x1
from .qwen3 import MLP


@dataclass(frozen=True)
class Config:
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    eps: float
    rope: bool = True

    @classmethod
    def from_dict(cls, data):
        if data.get('use_mup', False):
            raise ValueError('MiniCPM muP scaling is not implemented')
        return cls(data['hidden_size'], data['intermediate_size'], data['num_attention_heads'],
                   data['num_key_value_heads'], data.get('kv_channels') or
                   data['hidden_size'] // data['num_attention_heads'], data['num_hidden_layers'],
                   data['rms_norm_eps'], not data.get('no_rope', False))


class Block(nn.Module):
    value_scale = 16.0

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_norm = CompositeRMSNorm(cfg.hidden, cfg.eps)
        self.post_norm = CompositeRMSNorm(cfg.hidden, cfg.eps)
        self.q = conv1x1(cfg.hidden, cfg.heads * cfg.head_dim)
        self.k = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)
        self.v = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)
        self.o = conv1x1(cfg.heads * cfg.head_dim, cfg.hidden)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask, past=None):
        cfg, b, n = self.cfg, x.shape[0], x.shape[-1]
        h = self.input_norm(x)
        q = self.q(h).reshape(b, cfg.heads, cfg.head_dim, n)
        k = self.k(h).reshape(b, cfg.kv_heads, cfg.head_dim, n)
        v = self.v(h).reshape(b, cfg.kv_heads, cfg.head_dim, n)
        if cfg.rope:
            def rotate(t):
                a, b = t.chunk(2, dim=2)
                return t * cos + torch.cat((-b, a), dim=2) * sin
            q, k = rotate(q), rotate(k)
        new = (k.reshape(b, -1, 1, n), v.reshape(b, -1, 1, n))
        if past is not None:
            pk, pv = past
            k = torch.cat((pk.reshape(b, cfg.kv_heads, cfg.head_dim, -1), k), dim=-1)
            v = torch.cat((pv.reshape(b, cfg.kv_heads, cfg.head_dim, -1), v), dim=-1)
        group = cfg.heads // cfg.kv_heads
        attention = F.scaled_dot_product_attention(
            q.transpose(2, 3), k.transpose(2, 3).repeat_interleave(group, dim=1),
            v.transpose(2, 3).repeat_interleave(group, dim=1), attn_mask=mask)
        x = x + self.o(attention.transpose(2, 3).reshape(b, -1, 1, n) / self.value_scale)
        return x + self.mlp(self.post_norm(x)), new


class Transformer(nn.Module):
    """A contiguous group of layers, optionally exposing KV rows in its output."""

    def __init__(self, cfg, start=0, end=None, *, cached=False):
        super().__init__()
        self.cfg, self.start = cfg, start
        end = cfg.layers if end is None else end
        self.layers = nn.ModuleList(Block(cfg) for _ in range(start, end))
        self.norm = CompositeRMSNorm(cfg.hidden, cfg.eps) if end == cfg.layers else nn.Identity()
        self.cached = cached

    def forward(self, hidden, cos, sin, mask, cache=None):
        width = self.cfg.kv_heads * self.cfg.head_dim
        keys = cache.split(width, dim=1) if cache is not None else None
        present = []
        for i, layer in enumerate(self.layers):
            past = (keys[2*i], keys[2*i+1]) if keys is not None else None
            hidden, new = layer(hidden, cos, sin, mask, past)
            present.extend(new)
        hidden = self.norm(hidden)
        return torch.cat([hidden, *present], dim=1) if self.cached else hidden

    def load_hf(self, tensors, prefix):
        def copy(target, name, scale=1):
            value = tensors.get_tensor(prefix + name).float() * scale
            target.data.copy_(value.reshape(target.shape))
        for i, layer in enumerate(self.layers, self.start):
            root = f'layers.{i}.'
            copy(layer.input_norm.weight, root + 'input_layernorm.weight')
            copy(layer.post_norm.weight, root + 'post_attention_layernorm.weight')
            for name in ('q', 'k', 'v', 'o'):
                copy(getattr(layer, name).weight, root + f'self_attn.{name}_proj.weight',
                     layer.value_scale if name == 'v' else 1)
            for name, upstream in (('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')):
                weight = tensors.get_tensor(prefix + root + f'mlp.{upstream}.weight')
                chunks = weight.chunk(len(layer.mlp.gate), dim=1 if name == 'down' else 0)
                for target, chunk in zip(getattr(layer.mlp, name), chunks):
                    target.weight.data.copy_(chunk.reshape(target.weight.shape))
        if not isinstance(self.norm, nn.Identity):
            copy(self.norm.weight, 'norm.weight')
