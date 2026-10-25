"""Qwen3 bidirectional diffusion backbone in BC1S layout (no causal KV cache)."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tinkerbox_ane.blocks import CompositeRMSNorm, conv1x1, silu


@dataclass(frozen=True)
class Config:
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    eps: float
    rope_theta: float

    @classmethod
    def from_dict(cls, config):
        if config['model_type'] != 'qwen3' or config.get('attention_bias', False):
            raise ValueError('expected a bias-free Qwen3 backbone')
        return cls(config['hidden_size'], config['intermediate_size'],
                   config['num_attention_heads'], config['num_key_value_heads'],
                   config['head_dim'], config['num_hidden_layers'], config['rms_norm_eps'],
                   config['rope_parameters']['rope_theta'])


class MLP(nn.Module):
    def __init__(self, cfg, splits=4):
        super().__init__()
        if cfg.ffn % splits:
            raise ValueError('MLP split must divide intermediate size')
        self.gate = nn.ModuleList(conv1x1(cfg.hidden, cfg.ffn // splits) for _ in range(splits))
        self.up = nn.ModuleList(conv1x1(cfg.hidden, cfg.ffn // splits) for _ in range(splits))
        self.down = nn.ModuleList(conv1x1(cfg.ffn // splits, cfg.hidden) for _ in range(splits))

    def forward(self, x):
        return sum(down(silu(gate(x), 1.0) * up(x))
                   for gate, up, down in zip(self.gate, self.up, self.down))


class Block(nn.Module):
    # Preserve small V/attention intermediates, then undo the exact power-of-two
    # scale before O. Attention probabilities and exact-arithmetic results agree.
    value_scale = 16.0

    def __init__(self, cfg, splits):
        super().__init__()
        self.cfg = cfg
        self.input_norm = CompositeRMSNorm(cfg.hidden, cfg.eps)
        self.post_norm = CompositeRMSNorm(cfg.hidden, cfg.eps)
        self.q = conv1x1(cfg.hidden, cfg.heads * cfg.head_dim)
        self.k = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)
        self.v = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)
        self.o = conv1x1(cfg.heads * cfg.head_dim, cfg.hidden)
        self.q_norm = CompositeRMSNorm(cfg.head_dim, cfg.eps, axis=2)
        self.k_norm = CompositeRMSNorm(cfg.head_dim, cfg.eps, axis=2)
        self.mlp = MLP(cfg, splits)

    def project(self, x, cos, sin):
        cfg, b, n = self.cfg, x.shape[0], x.shape[-1]
        h = self.input_norm(x)
        q = self.q_norm(self.q(h).reshape(b, cfg.heads, cfg.head_dim, n))
        k = self.k_norm(self.k(h).reshape(b, cfg.kv_heads, cfg.head_dim, n))
        v = self.v(h).reshape(b, cfg.kv_heads, cfg.head_dim, n).transpose(2, 3)
        def rope(t):
            a, b = t.chunk(2, dim=2)
            return (t * cos + torch.cat((-b, a), dim=2) * sin).transpose(2, 3)
        q, k = rope(q), rope(k)
        return q, k, v

    def finish(self, x, attended):
        x = x + self.o(attended / self.value_scale)
        return x + self.mlp(self.post_norm(x))

    def forward(self, x, cos, sin, mask):
        q, k, v = self.project(x, cos, sin)
        group = self.cfg.heads // self.cfg.kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        if q.shape[2] < 1024:
            h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # Long sequences run faster on the ANE as query blocks; the result is
            # identical. 256-row blocks fail ANE compilation above 2,048 positions.
            rows = 256 if q.shape[2] <= 2048 else 512
            h = torch.cat([F.scaled_dot_product_attention(q[:, :, start:start+rows], k, v,
                                                          attn_mask=mask[..., start:start+rows, :])
                           for start in range(0, q.shape[2], rows)], dim=2)
        return self.finish(x, h.transpose(2, 3).reshape(x.shape[0], -1, 1, x.shape[-1]))


class Backbone(nn.Module):
    def __init__(self, cfg, splits=4):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(Block(cfg, splits) for _ in range(cfg.layers))
        self.norm = CompositeRMSNorm(cfg.hidden, cfg.eps)

    def forward(self, x, cos, sin, mask):
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)

    def load_hf(self, tensors, prefix='llm.'):
        """Read Qwen3 weights from an opened safetensors checkpoint."""
        def copy(target, name, scale=1.0):
            value = tensors.get_tensor(prefix + name)
            if scale != 1.0:
                value = value * scale
            target.data.copy_(value.reshape(target.shape))
        for i, layer in enumerate(self.layers):
            root = f'layers.{i}.'
            copy(layer.input_norm.weight, root + 'input_layernorm.weight')
            copy(layer.post_norm.weight, root + 'post_attention_layernorm.weight')
            for name in ('q', 'k', 'v', 'o'):
                copy(getattr(layer, name).weight, root + f'self_attn.{name}_proj.weight',
                     layer.value_scale if name == 'v' else 1.0)
            for name in ('q', 'k'):
                copy(getattr(layer, name + '_norm').weight, root + f'self_attn.{name}_norm.weight')
            for name, upstream in (('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')):
                value = tensors.get_tensor(prefix + root + f'mlp.{upstream}.weight')
                chunks = value.chunk(len(layer.mlp.gate), dim=1 if name == 'down' else 0)
                for target, chunk in zip(getattr(layer.mlp, name), chunks):
                    target.weight.data.copy_(chunk.reshape(target.weight.shape))
        copy(self.norm.weight, 'norm.weight')
