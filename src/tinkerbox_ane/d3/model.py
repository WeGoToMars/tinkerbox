"""FP16 Qwen3.5 blocks and staged prompt execution in BC1S layout."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from tinkerbox_ane.blocks import MLP, RMSNorm, block_scale, conv1x1, silu
from tinkerbox_ane.d3 import CHUNK, RESIDUAL_SCALE

PREFIX = "language_model."
MASK_MIN = -30000.0
NORM_COMPUTE_SCALE = 4.0
DELTA_STATE_SCALE = 64.0
GATED_NORM_SCALE = 16.0


@dataclass(frozen=True)
class Config:
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    linear_k_heads: int
    linear_v_heads: int
    linear_k_dim: int
    linear_v_dim: int
    layer_types: tuple[str, ...]
    conv_taps: int
    eps: float
    rotary_dim: int
    rope_theta: float
    mrope_section: tuple[int, ...]

    @classmethod
    def from_hf(cls, path: str | Path) -> Config:
        data = json.loads(Path(path).read_text())
        t = data.get("text_config", data)
        return cls(
            hidden=t["hidden_size"],
            ffn=t["intermediate_size"],
            heads=t["num_attention_heads"],
            kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            linear_k_heads=t["linear_num_key_heads"],
            linear_v_heads=t["linear_num_value_heads"],
            linear_k_dim=t["linear_key_head_dim"],
            linear_v_dim=t["linear_value_head_dim"],
            layer_types=tuple(t["layer_types"]),
            conv_taps=t["linear_conv_kernel_dim"],
            eps=t["rms_norm_eps"],
            rotary_dim=int(t["head_dim"] * t["rope_parameters"].get("partial_rotary_factor", 1.0)),
            rope_theta=t["rope_parameters"]["rope_theta"],
            mrope_section=tuple(t["rope_parameters"]["mrope_section"]),
        )


def _non_alias(x: torch.Tensor) -> torch.Tensor:
    # ANE compiler work-around: multiplying a dynamic activation by itself triggers an
    # assertion failure in ZinTensorMap::RemoveTensor due to double-removal during graph rewriting.
    # Chunking and re-concatenating creates a distinct tensor node while preserving identical values bit-for-bit.
    return torch.cat(x.chunk(2, dim=1), dim=1)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    # Numerically stable softplus without branching/where: relu(x) + log1p(exp(-|x|))
    # Never overflows in FP16 and achieves 100% pure ANE placement.
    return F.relu(x) + torch.log1p(torch.exp(-x.abs()))


class LinearAttentionBlock(nn.Module):
    def __init__(self, cfg: Config, ffn_scale: float):
        super().__init__()
        norm = dict(input_scale=RESIDUAL_SCALE, compute_scale=NORM_COMPUTE_SCALE)
        self.operator_norm = RMSNorm(cfg.hidden, cfg.eps, **norm)
        self.ffn_norm = RMSNorm(cfg.hidden, cfg.eps, **norm)

        self.hidden = cfg.hidden
        self.k_heads = cfg.linear_k_heads
        self.v_heads = cfg.linear_v_heads
        self.heads = self.v_heads
        self.k_dim = cfg.linear_k_dim
        self.v_dim = cfg.linear_v_dim
        self.taps = cfg.conv_taps
        self.conv_dim = self.k_heads * self.k_dim * 2 + self.v_heads * self.v_dim
        self.eps = cfg.eps
        self.scaling = self.k_dim ** -0.5

        self.in_proj = conv1x1(cfg.hidden, self.conv_dim + self.v_heads * self.v_dim + 2 * self.v_heads)
        self.conv = nn.Conv2d(self.conv_dim, self.conv_dim, (1, self.taps), groups=self.conv_dim, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.v_heads))
        self.norm_weight = nn.Parameter(torch.ones(self.v_dim))
        self.out_proj = conv1x1(self.v_heads * self.v_dim, cfg.hidden)
        self.feed_forward = MLP(cfg.hidden, cfg.ffn, ffn_scale)

        with torch.device("cpu"):
            sub_chunk = 32
            mask = torch.zeros(sub_chunk, sub_chunk, dtype=torch.float16)
            mask = mask.masked_fill(torch.ones(sub_chunk, sub_chunk, dtype=torch.bool).triu(1), MASK_MIN)
            self.register_buffer("upper_mask_sub", mask, persistent=False)
            self.register_buffer("triu_ones_sub", torch.ones(sub_chunk, sub_chunk, dtype=torch.float16).triu(), persistent=False)
            self.register_buffer("tril_mask_sub", torch.ones(sub_chunk, sub_chunk, dtype=torch.float16).tril(-1), persistent=False)
            self.register_buffer("eye_sub", torch.eye(sub_chunk, dtype=torch.float16), persistent=False)
            m_diag = torch.zeros(sub_chunk, sub_chunk, dtype=torch.float16)
            m_diag[:16, :16] = 1.0
            m_diag[16:, 16:] = 1.0
            self.register_buffer("m_diag", m_diag, persistent=False)
            m_off = torch.zeros(sub_chunk, sub_chunk, dtype=torch.float16)
            m_off[16:, :16] = 1.0
            self.register_buffer("m_off", m_off, persistent=False)

            i, j, t = torch.meshgrid(torch.arange(sub_chunk), torch.arange(sub_chunk), torch.arange(sub_chunk), indexing="ij")
            intervals = ((t > j) & (t <= i)).to(torch.float16).reshape(sub_chunk * sub_chunk, sub_chunk).T.contiguous()
            self.register_buffer("decay_intervals", intervals, persistent=False)

    def forward(self, x: torch.Tensor, tail: torch.Tensor, state: torch.Tensor):
        chunk = x.shape[-1]
        residual = x
        normed_x = self.operator_norm(x)
        fused = self.in_proj(normed_x)
        qkv, z, b, a = fused.split([self.conv_dim, self.v_heads * self.v_dim, self.v_heads, self.v_heads], dim=1)
        z = z.view(1, self.v_heads, self.v_dim, chunk).permute(0, 1, 3, 2)

        window = torch.cat((tail, qkv), dim=-1)
        new_tail = window[..., -(self.taps - 1):]
        # A 1 x taps kernel along tokens is faster on ANE than stacking tap planes.
        convolved = silu(self.conv(window), 1.0)

        q, k, v = convolved.split([self.k_heads * self.k_dim, self.k_heads * self.k_dim, self.v_heads * self.v_dim], dim=1)
        q = q.view(1, self.k_heads, self.k_dim, chunk).permute(0, 1, 3, 2)
        k = k.view(1, self.k_heads, self.k_dim, chunk).permute(0, 1, 3, 2)
        # Keep DeltaNet values and recurrent state above the ANE FP16 subnormal range.
        v = (v * DELTA_STATE_SCALE).view(1, self.v_heads, self.v_dim, chunk).permute(0, 1, 3, 2)

        if self.v_heads // self.k_heads > 1:
            repeat = self.v_heads // self.k_heads
            q = q.repeat_interleave(repeat, dim=1)
            k = k.repeat_interleave(repeat, dim=1)

        q, k = q * NORM_COMPUTE_SCALE, k * NORM_COMPUTE_SCALE
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6 * NORM_COMPUTE_SCALE**2) * self.scaling
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6 * NORM_COMPUTE_SCALE**2)

        beta = (0.5 + 0.5 * torch.tanh(0.5 * b)).view(1, self.v_heads, chunk, 1)
        g = -torch.exp(self.A_log[None, :, None, None]) * _softplus(a + self.dt_bias[None, :, None, None])
        g = g.squeeze(2)

        SUB = 32
        NUM_SUBS = chunk // SUB
        v_beta = (v * beta).view(1, self.v_heads, NUM_SUBS, SUB, self.v_dim)
        k_beta = (k * beta).view(1, self.v_heads, NUM_SUBS, SUB, self.k_dim)
        q_b = q.view(1, self.v_heads, NUM_SUBS, SUB, self.k_dim)
        k_b = k.view(1, self.v_heads, NUM_SUBS, SUB, self.k_dim)
        g_b = g.view(1, self.v_heads, NUM_SUBS, SUB)

        cum_decay = (g_b.unsqueeze(-2) @ self.triu_ones_sub).squeeze(-2)
        # Sum each interval directly; subtracting large FP16 prefix sums loses short decays.
        intervals = (g_b.unsqueeze(-2) @ self.decay_intervals).reshape(1, self.v_heads, NUM_SUBS, SUB, SUB)
        pairwise_decay = (intervals + self.upper_mask_sub).exp()

        ut_system = (k_beta @ k_b.transpose(-1, -2)) * pairwise_decay
        intra_chunk_attn = (q_b @ k_b.transpose(-1, -2)) * pairwise_decay
        decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

        L = ut_system * self.tril_mask_sub
        L_diag = L * self.m_diag
        L_off = L * self.m_off
        L2 = L_diag @ _non_alias(L_diag)
        L4 = L2 @ _non_alias(L2)

        L8 = L4 @ _non_alias(L4)
        inv_diag = (self.eye_sub - L_diag) @ _non_alias(self.eye_sub + L2) @ _non_alias(self.eye_sub + L4) @ _non_alias(self.eye_sub + L8)
        inv = inv_diag - inv_diag @ _non_alias(L_off) @ _non_alias(inv_diag)

        q_decayed = q_b * cum_decay.exp().unsqueeze(-1)
        k_decayed = k_b * pairwise_decay[..., -1, :].unsqueeze(-1)
        chunk_decay = cum_decay[..., -1:].exp().unsqueeze(-1)

        sub_outs = []
        for s in range(NUM_SUBS):
            rhs = v_beta[:, :, s] - decayed_k_beta[:, :, s] @ state
            v_new = inv[:, :, s] @ rhs
            inter_attn = q_decayed[:, :, s] @ state
            core_attn_s = inter_attn + intra_chunk_attn[:, :, s] @ v_new
            state = state * chunk_decay[:, :, s] + k_decayed[:, :, s].transpose(-1, -2) @ v_new
            sub_outs.append(core_attn_s)

        core_attn_out = torch.cat(sub_outs, dim=2)

        # RMSNormGated
        norm_input = core_attn_out * (GATED_NORM_SCALE / DELTA_STATE_SCALE)
        normed = norm_input * torch.rsqrt((norm_input * norm_input).mean(-1, keepdim=True) + self.eps * GATED_NORM_SCALE**2) * self.norm_weight
        normed = normed * silu(z, 1.0)
        attn_out = normed.permute(0, 1, 3, 2).reshape(1, self.v_heads * self.v_dim, 1, chunk)
        h = self.out_proj(attn_out)
        x = residual + h
        return x + self.feed_forward(self.ffn_norm(x)), new_tail, state


class AttentionBlock(nn.Module):
    def __init__(self, cfg: Config, scales: tuple[float, float]):
        super().__init__()
        norm = dict(input_scale=RESIDUAL_SCALE, compute_scale=NORM_COMPUTE_SCALE)
        self.operator_norm = RMSNorm(cfg.hidden, cfg.eps, **norm)
        self.ffn_norm = RMSNorm(cfg.hidden, cfg.eps, **norm)

        self.heads = cfg.heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.hidden = cfg.hidden
        self.scaling = cfg.head_dim ** -0.5
        self.rotary_dim = cfg.rotary_dim

        # Query projection has 2x channels for attention output gate
        self.q_proj = conv1x1(cfg.hidden, cfg.heads * cfg.head_dim * 2)
        self.k_proj = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)
        self.v_proj = conv1x1(cfg.hidden, cfg.kv_heads * cfg.head_dim)

        head_norm = dict(axis=2, input_scale=1.0, compute_scale=NORM_COMPUTE_SCALE)
        self.q_layernorm = RMSNorm(cfg.head_dim, cfg.eps, **head_norm)
        self.k_layernorm = RMSNorm(cfg.head_dim, cfg.eps, **head_norm)

        self.out_proj = conv1x1(self.heads * self.head_dim, cfg.hidden)
        self.feed_forward = MLP(cfg.hidden, cfg.ffn, scales[1])

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [1, heads, head_dim, s]
        rot, pas = x.split([self.rotary_dim, self.head_dim - self.rotary_dim], dim=2)
        r1, r2 = rot.chunk(2, dim=2)
        rotated = torch.cat([r1 * cos - r2 * sin, r1 * sin + r2 * cos], dim=2)
        return torch.cat([rotated, pas], dim=2)

    def project(self, x, cos, sin):
        normed = self.operator_norm(x)
        s = x.shape[-1]
        fused = self.q_proj(normed).view(1, self.heads, 2, self.head_dim, s)
        q = self._rope(self.q_layernorm(fused[:, :, 0]), cos, sin) * self.scaling
        k = self.k_layernorm(self.k_proj(normed).view(1, self.kv_heads, self.head_dim, s))
        k = self._rope(k, cos, sin)
        v = self.v_proj(normed).view(1, self.kv_heads, self.head_dim, s)
        return q.transpose(2, 3), fused[:, :, 1].transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

    def finish(self, x, attended, gate):
        gated = (attended * torch.sigmoid(gate)).transpose(2, 3).reshape(1, -1, 1, x.shape[-1])
        x = x + self.out_proj(gated)
        return x + self.feed_forward(self.ffn_norm(x))


class PromptAttention(nn.Module):
    """Merge SDPA block means by their softmax masses."""

    def __init__(self, cfg: Config, context: int):
        super().__init__()
        self.context = context
        self.group = cfg.heads // cfg.kv_heads
        self.block_size = 4096

    def forward(self, q, k, v, mask):
        if self.context <= self.block_size:
            return F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(self.group, dim=1),
                v.repeat_interleave(self.group, dim=1),
                attn_mask=mask,
                scale=1.0,
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
            scores = (grouped @ keys.unsqueeze(2).transpose(-2, -1)).view(batch, heads, tokens, -1) + block_mask
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
            LinearAttentionBlock(cfg, scale[1]) if kind == "linear_attention" else AttentionBlock(cfg, scale)
            for kind, scale in zip(cfg.layer_types, scales)
        )
        self.final_norm = RMSNorm(
            cfg.hidden, cfg.eps, input_scale=RESIDUAL_SCALE, compute_scale=NORM_COMPUTE_SCALE
        )


class ScanStage(nn.Module):
    """Scan consecutive DeltaNet layers, then project one full-attention layer."""

    def __init__(self, backbone: Backbone, start: int, end: int):
        super().__init__()
        self.layers = nn.ModuleList(backbone.layers[start:end])
        self.attention = backbone.layers[end]
        cfg = backbone.cfg
        conv_dim = cfg.linear_k_heads * cfg.linear_k_dim * 2 + cfg.linear_v_heads * cfg.linear_v_dim
        self.shapes = [shape for _ in self.layers for shape in (
            (1, conv_dim, 1, cfg.conv_taps - 1),
            (1, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim),
        )]
        if any(math.prod(s) % CHUNK for s in self.shapes):
            raise ValueError("recurrent state must pack into whole chunk columns")
        self.channels = sum(math.prod(s) for s in self.shapes) // CHUNK

    def forward(self, x, cos, sin, packed_state):
        states, offset = [], 0
        for shape in self.shapes:
            count = math.prod(shape) // CHUNK
            states.append(packed_state[:, offset:offset + count].reshape(shape))
            offset += count
        outputs = []
        for i, layer in enumerate(self.layers):
            x, tail, state = layer(x, states[2 * i], states[2 * i + 1])
            outputs.extend((tail.reshape(1, -1, 1, CHUNK), state.reshape(1, -1, 1, CHUNK)))
        q, gate, k, v = self.attention.project(x, cos, sin)
        return x, q, gate, k, v, torch.cat(outputs, dim=1)


class AttendStage(nn.Module):
    """Attend over the complete prompt and finish the decoder layer."""

    def __init__(self, backbone: Backbone, layer: int, capacity: int):
        super().__init__()
        self.layer = backbone.layers[layer]
        self.attention = PromptAttention(backbone.cfg, capacity)
        self.final_norm = backbone.final_norm if layer == len(backbone.layers) - 1 else nn.Identity()

    def forward(self, x, q, gate, k, v, mask):
        return self.final_norm(self.layer.finish(x, self.attention(q, k, v, mask), gate))


class PromptStage(nn.Module):
    """A whole short prompt through one stage in one call: scan it from an empty state, then attend over
    its keys. Keeps the recurrent state and keys on the ANE. The scan runs in 128-token chunks and a final
    64-token chunk for the remainder; wider chunks run slower."""

    def __init__(self, backbone: Backbone, start: int, end: int, width: int):
        super().__init__()
        self.scan = ScanStage(backbone, start, end)
        self.attend = AttendStage(backbone, end, width)
        self.chunks = [CHUNK] * (width // CHUNK) + [width % CHUNK] * bool(width % CHUNK)

    def forward(self, x, cos, sin, mask):
        state = torch.zeros(1, self.scan.channels, 1, CHUNK, dtype=x.dtype)
        saved, keys, values, start = [], [], [], 0
        for size in self.chunks:
            part = slice(start, start + size)
            out, q, gate, k, v, state = self.scan(x[..., part], cos[..., part], sin[..., part], state)
            saved.append((out, q, gate))
            keys.append(k)
            values.append(v)
            start += size
        k, v = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        return torch.cat([self.attend(out, q, gate, k, v, mask) for out, q, gate in saved], dim=-1)


def stage_ranges(cfg: Config) -> list[tuple[int, int]]:
    stages, start = [], 0
    for end, kind in enumerate(cfg.layer_types):
        if kind == "full_attention":
            if end == start:
                raise ValueError("each full-attention layer needs preceding DeltaNet layers")
            stages.append((start, end))
            start = end + 1
        elif kind != "linear_attention":
            raise ValueError(f"unsupported Qwen3.5 layer type: {kind}")
    if start != len(cfg.layer_types):
        raise ValueError("Qwen3.5 stack must end with full attention")
    return stages


def _layer_weights(f, i: int, kind: str) -> dict[str, torch.Tensor]:
    p = f"{PREFIX}layers.{i}."
    names = {
        "operator_norm.weight": f"{p}input_layernorm.weight",
        "ffn_norm.weight": f"{p}post_attention_layernorm.weight",
        "feed_forward.w1.weight": f"{p}mlp.gate_proj.weight",
        "feed_forward.w3.weight": f"{p}mlp.up_proj.weight",
        "feed_forward.w2.weight": f"{p}mlp.down_proj.weight",
    }
    if kind == "linear_attention":
        names |= {
            "in_proj_qkv": f"{p}linear_attn.in_proj_qkv.weight",
            "in_proj_z": f"{p}linear_attn.in_proj_z.weight",
            "in_proj_b": f"{p}linear_attn.in_proj_b.weight",
            "in_proj_a": f"{p}linear_attn.in_proj_a.weight",
            "conv.weight": f"{p}linear_attn.conv1d.weight",
            "A_log": f"{p}linear_attn.A_log",
            "dt_bias": f"{p}linear_attn.dt_bias",
            "norm_weight": f"{p}linear_attn.norm.weight",
            "out_proj.weight": f"{p}linear_attn.out_proj.weight",
        }
    else:
        names |= {
            "q_proj.weight": f"{p}self_attn.q_proj.weight",
            "k_proj.weight": f"{p}self_attn.k_proj.weight",
            "v_proj.weight": f"{p}self_attn.v_proj.weight",
            "out_proj.weight": f"{p}self_attn.o_proj.weight",
            "q_layernorm.weight": f"{p}self_attn.q_norm.weight",
            "k_layernorm.weight": f"{p}self_attn.k_norm.weight",
        }

    out = {}
    for mine, theirs in names.items():
        t = f.get_tensor(theirs).float()
        if mine in ("operator_norm.weight", "ffn_norm.weight", "q_layernorm.weight", "k_layernorm.weight"):
            t = 1.0 + t
        if mine in ("A_log", "dt_bias", "norm_weight"):
            out[mine] = t
        elif t.dim() == 2:
            out[mine] = t[:, :, None, None]
        elif t.dim() == 3:
            out[mine] = t[:, :, None, :]
        else:
            out[mine] = t
    return out


def _fold_scales(w: dict[str, torch.Tensor], kind: str, g_op: float, g_ffn: float) -> None:
    w["operator_norm.weight"] *= g_op
    w["ffn_norm.weight"] *= g_ffn
    w["feed_forward.w2.weight"] *= RESIDUAL_SCALE / g_ffn**2
    if kind == "linear_attention":
        # Fuse in_proj_qkv, in_proj_z, in_proj_b, in_proj_a into fused in_proj
        qkv = w.pop("in_proj_qkv")
        z = w.pop("in_proj_z")
        b = w.pop("in_proj_b")
        a = w.pop("in_proj_a")
        fused = torch.cat([qkv, z, b, a], dim=0) / g_op
        w["in_proj.weight"] = fused
        w["out_proj.weight"] *= RESIDUAL_SCALE
    else:
        w["q_proj.weight"] /= g_op
        w["k_proj.weight"] /= g_op
        w["v_proj.weight"] /= g_op
        w["out_proj.weight"] *= RESIDUAL_SCALE


def _split_ffn(w: dict[str, torch.Tensor]) -> None:
    for name, axis in (("w1", 0), ("w3", 0), ("w2", 1)):
        for k, t in enumerate(w.pop(f"feed_forward.{name}.weight").chunk(MLP.splits, dim=axis)):
            w[f"feed_forward.{name}.{k}.weight"] = t.contiguous()


class CheckpointReader:
    def __init__(self, checkpoint: str | Path):
        self.root = Path(checkpoint)
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            self.weight_map = index.get("weight_map", {})
            self._files = {}
        else:
            self.weight_map = None
            self._file = safe_open(self.root / "model.safetensors", "pt")

    def get_tensor(self, name: str) -> torch.Tensor:
        if self.weight_map is not None:
            filename = self.weight_map[name]
            if filename not in self._files:
                self._files[filename] = safe_open(self.root / filename, "pt")
            return self._files[filename].get_tensor(name)
        return self._file.get_tensor(name)


def load_weights(checkpoint: str | Path) -> Backbone:
    dtype = torch.float16
    checkpoint = Path(checkpoint)
    cfg = Config.from_hf(checkpoint / "config.json")
    state, scales = {}, []
    reader = CheckpointReader(checkpoint)
    for i, kind in enumerate(cfg.layer_types):
        w = _layer_weights(reader, i, kind)
        g = block_scale(w["operator_norm.weight"]), block_scale(w["ffn_norm.weight"])
        _fold_scales(w, kind, *g)
        _split_ffn(w)
        scales.append(g)
        for name, tensor in w.items():
            target = f"layers.{i}.{name}"
            state[target] = tensor.to(dtype)
    state["final_norm.weight"] = (1.0 + reader.get_tensor(f"{PREFIX}norm.weight").float()).to(dtype)

    with torch.device("meta"):
        model = Backbone(cfg, scales)
    model.load_state_dict(state, strict=True, assign=True)
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.eps = torch.tensor(module.eps_value, dtype=dtype)
    return model.requires_grad_(False).eval()
