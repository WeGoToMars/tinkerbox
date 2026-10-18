"""FP16 fused LFM2 backbone in BC1S layout."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from tinkerbox_ane.d1 import CHUNK, RESIDUAL_SCALE

PREFIX = "model.language_model."
MASK_MIN = -30000.0
NORM_COMPUTE_SCALE = 4.0


@dataclass(frozen=True)
class Config:
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    layer_types: tuple[str, ...]
    conv_taps: int
    eps: float

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @classmethod
    def from_hf(cls, path: str | Path) -> Config:
        t = json.loads(Path(path).read_text())["text_config"]
        return cls(
            hidden=t["hidden_size"], ffn=t["intermediate_size"], heads=t["num_attention_heads"],
            kv_heads=t["num_key_value_heads"], layer_types=tuple(t["layer_types"]),
            conv_taps=t["conv_L_cache"],
            eps=t["norm_eps"],
        )


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


class ShortConv(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.dim, self.taps = cfg.hidden, cfg.conv_taps
        self.in_proj = conv1x1(cfg.hidden, 3 * cfg.hidden)
        self.conv = nn.Conv2d(cfg.hidden, cfg.hidden, (1, cfg.conv_taps), groups=cfg.hidden, bias=False)
        self.out_proj = conv1x1(cfg.hidden, cfg.hidden)

    def forward(self, x: torch.Tensor, tail: torch.Tensor, route) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, u = self.in_proj(x).split(self.dim, dim=1)
        window = torch.cat((tail, b * u), dim=-1)
        # One-hot FP16 routing stays on ANE; dynamic integer gathers do not.
        # Keep taps in contiguous planes; a vertical kernel avoids the
        # stride-three depthwise width convolution's ANE channel failure.
        routed = window @ route
        width = x.shape[-1]
        planes = routed[..., :width * self.taps].reshape(1, self.dim, self.taps, width)
        convolved = F.conv2d(planes, self.conv.weight.transpose(2, 3), groups=self.dim)
        return self.out_proj(c * convolved), routed[..., width * self.taps:]


class MLP(nn.Module):
    splits = 8

    def __init__(self, cfg: Config, scale: float):
        super().__init__()
        c = cfg.ffn // self.splits
        self.w1 = nn.ModuleList(conv1x1(cfg.hidden, c) for _ in range(self.splits))
        self.w3 = nn.ModuleList(conv1x1(cfg.hidden, c) for _ in range(self.splits))
        self.w2 = nn.ModuleList(conv1x1(c, cfg.hidden) for _ in range(self.splits))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for w1, w3, w2 in zip(self.w1, self.w3, self.w2):
            part = w2(silu(w1(x), self.scale) * w3(x))
            out = part if out is None else out + part
        return out


class ConvBlock(nn.Module):
    def __init__(self, cfg: Config, ffn_scale: float):
        super().__init__()
        norm = dict(input_scale=RESIDUAL_SCALE, compute_scale=NORM_COMPUTE_SCALE)
        self.operator_norm = RMSNorm(cfg.hidden, cfg.eps, **norm)
        self.ffn_norm = RMSNorm(cfg.hidden, cfg.eps, **norm)
        self.operator = ShortConv(cfg)
        self.feed_forward = MLP(cfg, ffn_scale)

    def forward(self, x, tail, route):
        h, tail = self.operator(self.operator_norm(x), tail, route)
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x)), tail


class AttentionBlock(nn.Module):
    def __init__(self, cfg: Config, scales: tuple[float, float]):
        super().__init__()
        self.operator_norm = RMSNorm(cfg.hidden, cfg.eps, input_scale=RESIDUAL_SCALE,
                                     compute_scale=NORM_COMPUTE_SCALE)
        self.heads, self.kv_heads, self.head_dim = cfg.heads, cfg.kv_heads, cfg.head_dim
        self.q_proj = conv1x1(cfg.hidden, cfg.heads * cfg.head_dim)
        self.k_proj = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)
        self.v_proj = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)
        norm = dict(axis=2, input_scale=scales[0], compute_scale=scales[0])
        self.q_layernorm = RMSNorm(cfg.head_dim, cfg.eps, **norm)
        self.k_layernorm = RMSNorm(cfg.head_dim, cfg.eps, **norm)
        self.scale = cfg.head_dim**-0.5
        self.out_proj = conv1x1(cfg.hidden, cfg.hidden)
        self.ffn_norm = RMSNorm(cfg.hidden, cfg.eps, input_scale=RESIDUAL_SCALE,
                                compute_scale=NORM_COMPUTE_SCALE)
        self.feed_forward = MLP(cfg, scales[1])
        self.channels = cfg.hidden

    def _rope(self, x, cos, sin):
        half = self.head_dim // 2
        x1, x2 = x[:, :, :half], x[:, :, half:]
        return x * cos + torch.cat((-x2, x1), dim=2) * sin

    def forward(self, x, cos, sin, past_mask, past_k, past_v, attention):
        residual = x
        x = self.operator_norm(x)
        s = x.shape[-1]
        q = self.q_layernorm(self.q_proj(x).view(1, self.heads, self.head_dim, s))
        k = self.k_layernorm(self.k_proj(x).view(1, self.kv_heads, self.head_dim, s))
        v = self.v_proj(x).view(1, self.kv_heads, self.head_dim, s)
        q = self._rope(q, cos, sin) * self.scale
        k = self._rope(k, cos, sin)
        k, v = k.transpose(2, 3), v.transpose(2, 3)
        attended = attention(q.transpose(2, 3), past_k, past_v, k, v, past_mask)
        x = residual + self.out_proj(attended.transpose(2, 3).reshape(1, self.channels, 1, s))
        return x + self.feed_forward(self.ffn_norm(x)), k, v


class CacheAttention(nn.Module):
    """Merge 4096-key SDPA block means by their softmax masses."""

    def __init__(self, cfg: Config, context: int):
        super().__init__()
        self.context = context
        self.group = cfg.heads // cfg.kv_heads

    def forward(self, q, past_k, past_v, k_new, v_new, past_mask, chunk_mask):
        k, v = torch.cat((past_k, k_new), dim=2), torch.cat((past_v, v_new), dim=2)
        mask = torch.cat((past_mask, chunk_mask), dim=-1)
        if self.context <= 4096:
            return F.scaled_dot_product_attention(
                q, k.repeat_interleave(self.group, dim=1),
                v.repeat_interleave(self.group, dim=1),
                attn_mask=mask, scale=1.0
            )
        maxima, masses, means = [], [], []
        for start in range(0, self.context, 4096):
            kb = k[:, :, start:start + 4096].repeat_interleave(self.group, dim=1)
            vb = v[:, :, start:start + 4096].repeat_interleave(self.group, dim=1)
            block_mask = mask[..., start:start + 4096]
            scores = q @ kb.transpose(-2, -1) + block_mask
            maximum = scores.amax(-1, keepdim=True)
            maxima.append(maximum)
            masses.append(torch.exp(scores - maximum).sum(-1, keepdim=True))
            means.append(F.scaled_dot_product_attention(q, kb, vb, attn_mask=block_mask, scale=1.0))
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


class Backbone(nn.Module):
    def __init__(self, cfg: Config, scales: list[tuple[float, float]]):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            ConvBlock(cfg, scale[1]) if kind == "conv" else AttentionBlock(cfg, scale)
            for kind, scale in zip(cfg.layer_types, scales)
        )
        self.final_norm = RMSNorm(cfg.hidden, cfg.eps, input_scale=RESIDUAL_SCALE,
                                  compute_scale=NORM_COMPUTE_SCALE)


class FusedBackbone(nn.Module):
    """One chunk through all layers; convolution tails precede key/value caches."""

    def __init__(self, backbone: Backbone, capacity: int):
        super().__init__()
        self.layers = backbone.layers
        self.final_norm = backbone.final_norm
        self.cfg = cfg = backbone.cfg
        self.capacity = capacity
        self.attention = CacheAttention(cfg, capacity)
        self.conv_layers = [i for i, kind in enumerate(cfg.layer_types) if kind == "conv"]
        self.attention_layers = [i for i, kind in enumerate(cfg.layer_types) if kind == "full_attention"]
        self.shapes = ([(1, cfg.hidden, 1, 2 * (cfg.conv_taps - 1))] * len(self.conv_layers)
                       + [(1, cfg.kv_heads, capacity - CHUNK, cfg.head_dim)] * (2 * len(self.attention_layers)))
        self.size = sum(math.prod(shape) for shape in self.shapes)

    def example_inputs(self):
        cfg = self.cfg
        shapes = [(1, cfg.hidden, 1, CHUNK), (1, 1, cfg.head_dim, CHUNK),
                  (1, 1, cfg.head_dim, CHUNK), (1, 1, CHUNK, self.capacity - CHUNK),
                  (1, 1, 1, self.size), (1, 1, CHUNK, CHUNK),
                   (1, 1, CHUNK + 2 * (cfg.conv_taps - 1), CHUNK * cfg.conv_taps + 2 * (cfg.conv_taps - 1))]
        return tuple(torch.zeros(shape, dtype=torch.float16) for shape in shapes)

    def forward(self, x, cos, sin, past_mask, packed_state, chunk_mask, conv_route):
        states, offset = [], 0
        for shape in self.shapes:
            count = math.prod(shape)
            states.append(packed_state[..., offset:offset + count].reshape(shape))
            offset += count
        tails = iter(states[:len(self.conv_layers)])
        histories = iter(states[len(self.conv_layers):])
        new_tails, new_kv = [], []
        attention = lambda q, pk, pv, k, v, mask: self.attention(q, pk, pv, k, v, mask, chunk_mask)
        for layer in self.layers:
            if isinstance(layer, ConvBlock):
                x, tail = layer(x, next(tails), conv_route)
                new_tails.append(tail)
            else:
                x, k, v = layer(x, cos, sin, past_mask, next(histories), next(histories), attention)
                new_kv.extend((k, v))
        return self.final_norm(x), torch.cat([v.reshape(1, 1, 1, -1) for v in new_tails + new_kv], dim=-1)


def _layer_weights(f, i: int, kind: str) -> dict[str, torch.Tensor]:
    p = f"{PREFIX}layers.{i}."
    op = "conv." if kind == "conv" else "self_attn."
    names = {n: n for n in ("operator_norm.weight", "ffn_norm.weight", "feed_forward.w1.weight",
                            "feed_forward.w2.weight", "feed_forward.w3.weight")}
    if kind == "conv":
        names |= {f"operator.{n}": f"{op}{n}" for n in ("in_proj.weight", "out_proj.weight", "conv.weight")}
    else:
        names |= {f"operator.{n}": f"{op}{n}" for n in (
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
            "q_layernorm.weight", "k_layernorm.weight")}
    out = {}
    for mine, theirs in names.items():
        t = f.get_tensor(p + theirs).float()
        if t.dim() == 2:
            t = t[:, :, None, None]
        elif t.dim() == 3:
            t = t[:, :, None, :]
        out[mine] = t
    return out


def _fold_scales(w: dict[str, torch.Tensor], kind: str, g_op: float, g_ffn: float) -> None:
    w["operator_norm.weight"] *= g_op
    w["ffn_norm.weight"] *= g_ffn
    w["feed_forward.w2.weight"] *= RESIDUAL_SCALE / g_ffn**2
    if kind == "conv":
        w["operator.conv.weight"] /= g_op
        w["operator.out_proj.weight"] *= RESIDUAL_SCALE / g_op**2
    else:
        w["operator.out_proj.weight"] *= RESIDUAL_SCALE / g_op


def _split_ffn(w: dict[str, torch.Tensor]) -> None:
    for name, axis in (("w1", 0), ("w3", 0), ("w2", 1)):
        for k, t in enumerate(w.pop(f"feed_forward.{name}.weight").chunk(MLP.splits, dim=axis)):
            w[f"feed_forward.{name}.{k}.weight"] = t.contiguous()


def load_weights(checkpoint: str | Path) -> Backbone:
    dtype = torch.float16
    checkpoint = Path(checkpoint)
    cfg = Config.from_hf(checkpoint / "config.json")
    state, scales = {}, []
    with safe_open(checkpoint / "model.safetensors", "pt") as f:
        for i, kind in enumerate(cfg.layer_types):
            w = _layer_weights(f, i, kind)
            g = block_scale(w["operator_norm.weight"]), block_scale(w["ffn_norm.weight"])
            _fold_scales(w, kind, *g)
            _split_ffn(w)
            scales.append(g)
            for name, tensor in w.items():
                target = name if kind == "conv" else name.removeprefix("operator.")
                target = f"layers.{i}.{target}"
                state[target] = tensor.to(dtype)
        state["final_norm.weight"] = f.get_tensor(f"{PREFIX}embedding_norm.weight").to(dtype)
    with torch.device("meta"):
        model = Backbone(cfg, scales)
    model.load_state_dict(state, strict=True, assign=True)
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.eps = torch.tensor(module.eps_value, dtype=dtype)
    return model.requires_grad_(False).eval()
