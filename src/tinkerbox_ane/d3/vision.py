"""Qwen3.5 vision tower on the Neural Engine at static patch capacities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
)

from tinkerbox_ane.coreai import count_flops, source_hash
from tinkerbox_ane.d3 import PATCH_TIERS, RESIDUAL_SCALE
from tinkerbox_ane.d3.model import MASK_MIN, CheckpointReader

IMAGE_TOKEN_ID = 248056
# The official processor's smallest image: 64 tokens.
MIN_PIXELS = 65_536
# Attention scores per head that one block of query rows computes.
SCORES = 2048 * 2048
# Heads attend in groups; four groups of four heads run faster than one or two.
HEAD_GROUPS = 4


def conv(linear: nn.Linear) -> nn.Conv2d:
    """A linear layer as the equivalent 1 x 1 convolution over [1, channels, 1, patches]."""
    layer = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
    layer.weight.data = linear.weight.data[:, :, None, None]
    if linear.bias is not None:
        layer.bias.data = linear.bias.data
    return layer


class LayerNorm(nn.Module):
    """LayerNorm over channels. The residual stream reaches thousands, whose squares overflow FP16, so
    each patch is first divided by 32 times its mean absolute deviation, which bounds every value by
    channels / 32; the normalized result is unchanged."""

    def __init__(self, norm: nn.LayerNorm):
        super().__init__()
        self.weight = nn.Parameter(norm.weight.data.view(1, -1, 1, 1))
        self.bias = nn.Parameter(norm.bias.data.view(1, -1, 1, 1))
        self.root_eps = norm.eps ** 0.5

    def forward(self, x):
        centered = x - x.mean(1, keepdim=True)
        inverse = 1 / (32 * centered.abs().mean(1, keepdim=True) + 0.01)
        scaled = centered * inverse
        eps = (self.root_eps * inverse) ** 2
        return scaled * torch.rsqrt((scaled * scaled).mean(1, keepdim=True) + eps) * self.weight + self.bias


class ANEVisionBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.norm1 = LayerNorm(block.norm1)
        self.norm2 = LayerNorm(block.norm2)
        self.qkv = conv(block.attn.qkv)
        self.proj = conv(block.attn.proj)
        self.fc1 = conv(block.mlp.linear_fc1)
        self.fc2 = conv(block.mlp.linear_fc2)
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.scaling = block.attn.scaling

    def forward(self, x, cos, sin, mask):
        # x: [1, hidden, 1, patches]; cos, sin: [1, 1, head_dim // 2, patches]; mask: [1, 1, 1, patches]
        s = x.shape[-1]
        q, k, v = self.qkv(self.norm1(x)).view(1, 3, self.num_heads, self.head_dim, s).unbind(1)
        # Axial rotary over the whole head; both halves share the same angles.
        q1, q2 = q.chunk(2, dim=2)
        k1, k2 = k.chunk(2, dim=2)
        q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=2) * self.scaling
        k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=2)
        # Every query attends over all patches. A row of ones under the values sums the softmax weights
        # in the same matrix product, so the division runs on [head_dim, rows] instead of every score.
        values = torch.cat([v, torch.ones_like(v[:, :, :1])], dim=2)
        queries = q.transpose(-1, -2)
        rows = max(r for r in range(1, s + 1) if s % r == 0 and r * s <= SCORES)
        group = self.num_heads // HEAD_GROUPS
        blocks = []
        for start in range(0, s, rows):
            parts = []
            for first in range(0, self.num_heads, group):
                heads = slice(first, first + group)
                scores = queries[:, heads, start:start + rows] @ k[:, heads] + mask
                weights = torch.exp(scores - scores.amax(-1, keepdim=True))
                total = values[:, heads] @ weights.transpose(-1, -2)
                parts.append(total[:, :, :self.head_dim] / total[:, :, self.head_dim:])
            blocks.append(torch.cat(parts, dim=1))
        x = x + self.proj(torch.cat(blocks, dim=-1).reshape(1, self.num_heads * self.head_dim, 1, s))
        return x + self.fc2(nn.functional.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class VisionTower(nn.Module):
    """Patch embedding, transformer blocks and 2 × 2 merger over padded patches in merge order,
    channels first: inputs are [1, channels, 1, patches] and features [1, hidden, 1, patches / 4].

    The host supplies each image's interpolated position embeddings and rotary tables. Padding
    patches are masked out of attention, and the merger keeps them in their own output columns."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        hidden = visual.config.hidden_size
        self.patch_proj = nn.Conv2d(visual.patch_embed.proj.weight[0].numel(), hidden, 1, bias=True)
        self.patch_proj.weight.data = visual.patch_embed.proj.weight.reshape(hidden, -1, 1, 1)
        self.patch_proj.bias.data = visual.patch_embed.proj.bias.data
        self.blocks = nn.ModuleList([ANEVisionBlock(block) for block in visual.blocks])
        self.merge = visual.config.spatial_merge_size ** 2
        merger = visual.merger
        self.merger_norm = LayerNorm(merger.norm)
        self.merger_fc1 = conv(merger.linear_fc1)
        self.merger_fc2 = conv(merger.linear_fc2)

    def forward(self, pixel_values, pos_embeds, cos, sin, attn_mask):
        h = self.patch_proj(pixel_values) + pos_embeds
        for block in self.blocks:
            h = block(h, cos, sin, attn_mask)
        h = self.merger_norm(h)
        # Each run of `merge` patches becomes one column of their concatenated channels.
        channels, patches = h.shape[1], h.shape[-1]
        h = h.view(1, channels, patches // self.merge, self.merge).permute(0, 3, 1, 2)
        h = h.reshape(1, channels * self.merge, 1, patches // self.merge)
        return self.merger_fc2(nn.functional.gelu(self.merger_fc1(h)))


def load_visual(checkpoint_dir: Path) -> nn.Module:
    config = AutoConfig.from_pretrained(checkpoint_dir).vision_config
    with torch.device("meta"):
        visual = Qwen3_5VisionModel(config)
    reader = CheckpointReader(checkpoint_dir)
    visual.load_state_dict({name: reader.get_tensor(f"visual.{name}") for name in visual.state_dict()},
                           assign=True, strict=True)
    visual.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config)
    return visual.eval().requires_grad_(False)


def export_vision_asset(checkpoint_dir: Path, out_path: Path) -> dict:
    """Export one tower function per patch capacity, sharing weights; returns the manifest fields."""
    from coreai_torch import TorchConverter, get_decomp_table

    visual = load_visual(Path(checkpoint_dir))
    tower = VisionTower(visual).half().eval().requires_grad_(False)
    config = visual.config
    half = config.hidden_size // config.num_heads // 2
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float16)
    converter = TorchConverter()
    flops = {}
    with torch.no_grad():
        for patches in PATCH_TIERS:
            inputs = (zeros(1, tower.patch_proj.in_channels, 1, patches), zeros(1, config.hidden_size, 1, patches),
                      zeros(1, 1, half, patches), zeros(1, 1, half, patches), zeros(1, 1, 1, patches))
            flops[f"p{patches}"] = count_flops(tower, inputs)
            program = torch.export.export(tower, inputs).run_decompositions(get_decomp_table())
            converter.add_exported_program(
                program, input_names=["pixel_values", "pos_embeds", "cos", "sin", "attn_mask"],
                output_names=["features"], entrypoint_name=f"p{patches}")
    out_path = Path(out_path)
    converter.to_coreai().save_asset(out_path)
    return {"vision_asset": out_path.name, "vision_source_hash": source_hash(out_path),
            "vision_functions": [f"p{patches}" for patches in PATCH_TIERS],
            "vision_flops": flops,
            "vision_parameter_bytes": sum(p.numel() * p.element_size() for p in tower.parameters())}


class VisionPipeline:
    """The checkpoint's own image processor, and the tower run on the ANE through the host at the
    bundle's patch capacities (its `p{patches}` functions)."""

    def __init__(self, checkpoint_dir: Path, host, capacities):
        checkpoint_dir = Path(checkpoint_dir)
        self.processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
        self.host = host
        self.capacities = sorted(capacities)
        self.config = AutoConfig.from_pretrained(checkpoint_dir).vision_config
        self.merge = self.config.spatial_merge_size
        self.pos_weight = CheckpointReader(checkpoint_dir).get_tensor("visual.pos_embed.weight").float()
        self.rotary = Qwen3_5VisionRotaryEmbedding(self.config)

    def limit_tokens(self, tokens: int):
        """Resize pictures to at most `tokens` merged patches: the official processor's pixel budget."""
        unit = (self.config.patch_size * self.merge) ** 2
        self.processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": tokens * unit}

    def layout(self, image) -> dict:
        """How one picture reaches the model: resized, never split or cropped, in one vision pass."""
        _, rows, cols = (int(n) for n in self.processor.image_processor(
            images=[image], return_tensors="pt")["image_grid_thw"][0])
        patch = self.config.patch_size
        return {"source": list(image.size), "cols": 1, "rows": 1, "size": [cols * patch, rows * patch],
                "overview": None, "passes": 1, "tokens": rows * cols // self.merge**2}

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor, grids: torch.Tensor) -> list[torch.Tensor]:
        """Each image's features, in order, scaled by RESIDUAL_SCALE, from the processor's
        `pixel_values` and `image_grid_thw` for the request's images."""
        features, start = [], 0
        for grid in grids:
            count = int(grid.prod())
            capacity = next(p for p in self.capacities if p >= count)
            thw = grid[None]
            indices, weights = get_vision_interpolation_indices_and_weights(
                thw, num_grid_per_side=int(self.config.num_position_embeddings ** 0.5), mode="bilinear",
                align_corners=True, spatial_merge_size=self.merge)
            pos = (self.pos_weight[indices] * weights[:, :, None]).sum(1)
            cos, sin = self.rotary(pos, get_vision_position_ids(thw, self.merge))
            half = cos.shape[1] // 2

            def padded(x):
                """[patches, channels] as [1, channels, 1, capacity]."""
                x = torch.cat([x.half(), torch.zeros(capacity - count, x.shape[1], dtype=torch.float16)])
                return x.T[None, :, None]

            mask = torch.full((1, 1, 1, capacity), MASK_MIN, dtype=torch.float16)
            mask[..., :count] = 0
            out = self.host.vision({
                "pixel_values": padded(pixel_values[start:start + count]),
                "pos_embeds": padded(pos),
                "cos": padded(cos[:, :half]).transpose(1, 2),
                "sin": padded(sin[:, :half]).transpose(1, 2),
                "attn_mask": mask,
            }, f"p{capacity}")
            features.append(torch.from_numpy(out[0, :, 0, :count // self.merge**2]).T.float() * RESIDUAL_SCALE)
            start += count
        return features
