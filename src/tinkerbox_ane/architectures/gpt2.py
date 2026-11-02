"""GPT-2 decoder blocks shared by Small, Medium, Large and XL, in BC1S layout."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    hidden: int
    heads: int
    layers: int
    positions: int
    vocab: int
    eps: float
    ffn: int

    @classmethod
    def from_dict(cls, data):
        if (data['model_type'] != 'gpt2' or data['activation_function'] != 'gelu_new'
                or data.get('add_cross_attention', False)
                or data.get('scale_attn_by_inverse_layer_idx', False)
                or not data.get('scale_attn_weights', True)):
            raise ValueError('expected the standard GPT-2 architecture')
        return cls(data['n_embd'], data['n_head'], data['n_layer'], data['n_positions'],
                   data['vocab_size'], data['layer_norm_epsilon'], data.get('n_inner') or 4*data['n_embd'])


class LayerNorm(nn.Module):
    def __init__(self, hidden, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias,
                            self.eps).permute(0, 3, 1, 2)


class Block(nn.Module):
    value_scale = 16.0

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln_1, self.ln_2 = LayerNorm(cfg.hidden, cfg.eps), LayerNorm(cfg.hidden, cfg.eps)
        self.qkv = nn.Conv2d(cfg.hidden, 3*cfg.hidden, 1)
        self.out = nn.Conv2d(cfg.hidden, cfg.hidden, 1)
        self.up = nn.Conv2d(cfg.hidden, cfg.ffn, 1)
        self.down = nn.Conv2d(cfg.ffn, cfg.hidden, 1)

    def forward(self, x, mask, past):
        batch, _, _, width = x.shape
        dim = self.cfg.hidden // self.cfg.heads
        q, k, v = self.qkv(self.ln_1(x)).chunk(3, dim=1)
        new = (k, v)
        pk, pv = past
        k, v = torch.cat((pk, k), dim=-1), torch.cat((pv, v), dim=-1)
        def heads(t):
            return t.reshape(batch, self.cfg.heads, dim, -1).transpose(2, 3)
        attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=mask)
        attended = attended.transpose(2, 3).reshape(batch, -1, 1, width)
        x = x + self.out(attended / self.value_scale)
        h = self.up(self.ln_2(x))
        h = F.gelu(h, approximate='tanh')
        return x + self.down(h), new


class Transformer(nn.Module):
    """Cached decoder with hidden rows followed by the new KV rows for the native host."""

    def __init__(self, cfg, start=0, end=None):
        super().__init__()
        self.cfg, self.start = cfg, start
        end = cfg.layers if end is None else end
        self.layers = nn.ModuleList(Block(cfg) for _ in range(start, end))
        self.norm = LayerNorm(cfg.hidden, cfg.eps) if end == cfg.layers else nn.Identity()

    def forward(self, hidden, mask, cache):
        past = cache.split(self.cfg.hidden, dim=1)
        present = []
        for i, layer in enumerate(self.layers):
            hidden, new = layer(hidden, mask, (past[2*i], past[2*i+1]))
            present.extend(new)
        return torch.cat([self.norm(hidden), *present], dim=1)

    def load_hf(self, tensors):
        """HF GPT-2 Conv1D weights store input channels first, unlike Conv2D."""
        def copy(target, name, *, transpose=False):
            value = tensors.get_tensor(name).float()
            if transpose:
                value = value.T
            target.data.copy_(value.reshape(target.shape))
        for i, layer in enumerate(self.layers, self.start):
            root = f'h.{i}.'
            for name in ('ln_1', 'ln_2'):
                for part in ('weight', 'bias'):
                    copy(getattr(getattr(layer, name), part), root + name + '.' + part)
            for name, source in [('qkv', 'attn.c_attn'), ('out', 'attn.c_proj'),
                                 ('up', 'mlp.c_fc'), ('down', 'mlp.c_proj')]:
                projection = getattr(layer, name)
                copy(projection.weight, root + source + '.weight', transpose=True)
                copy(projection.bias, root + source + '.bias')
            # Preserve small attention values on ANE, then undo the scale before O.
            layer.qkv.weight.data[2*self.cfg.hidden:] *= layer.value_scale
            layer.qkv.bias.data[2*self.cfg.hidden:] *= layer.value_scale
        if not isinstance(self.norm, nn.Identity):
            for part in ('weight', 'bias'):
                copy(getattr(self.norm, part), 'ln_f.' + part)
