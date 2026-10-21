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

from tinkerbox_ane.coreai import source_hash
from tinkerbox_ane.d3 import PATCH_TIERS, RESIDUAL_SCALE
from tinkerbox_ane.d3.model import MASK_MIN, CheckpointReader

IMAGE_TOKEN_ID = 248056
# The official processor's smallest image: 64 tokens.
MIN_PIXELS = 65_536
# Attention scores per head the ANE compiles in one matrix.
SCORES = 2048 * 2048


def as_rgb(image):
    """A Pillow picture in RGB; the processor resizes it."""
    if not hasattr(image, "convert"):
        raise ValueError(f"invalid image input: {image!r}")
    return image.convert("RGB")


class ANEVisionBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.qkv = block.attn.qkv
        self.proj = block.attn.proj
        self.mlp = block.mlp
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.scaling = block.attn.scaling

    def forward(self, x, cos, sin, mask):
        # x: [1, patches, hidden]; cos, sin: [1, 1, patches, head_dim // 2]; mask: [1, 1, 1, patches]
        h = self.norm1(x)
        b, s, d = h.shape
        q, k, v = self.qkv(h).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Axial rotary over the whole head; both halves share the same angles.
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        # Every query row attends over all patches; blocks of rows keep each score matrix within what the
        # ANE compiles (2048 x 2048 per head).
        rows = max(r for r in range(1, s + 1) if s % r == 0 and r * s <= SCORES)
        attended = torch.cat([nn.functional.scaled_dot_product_attention(
            q[:, :, start:start + rows], k, v, attn_mask=mask, scale=self.scaling)
            for start in range(0, s, rows)], dim=2)
        x = x + self.proj(attended.transpose(1, 2).reshape(b, s, d))
        return x + self.mlp(self.norm2(x))


class VisionTower(nn.Module):
    """Patch embedding, transformer blocks and 2 × 2 merger over padded patches in merge order.

    The host supplies each image's interpolated position embeddings and rotary tables. Padding
    patches are masked out of attention, and the merger keeps them in their own output rows."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        hidden = visual.config.hidden_size
        self.patch_proj = nn.Linear(visual.patch_embed.proj.weight[0].numel(), hidden, bias=True)
        self.patch_proj.weight.data = visual.patch_embed.proj.weight.reshape(hidden, -1)
        self.patch_proj.bias.data = visual.patch_embed.proj.bias.data
        self.blocks = nn.ModuleList([ANEVisionBlock(block) for block in visual.blocks])
        self.merger = visual.merger

    def forward(self, pixel_values, pos_embeds, cos, sin, attn_mask):
        h = self.patch_proj(pixel_values) + pos_embeds
        for block in self.blocks:
            h = block(h, cos, sin, attn_mask)
        return self.merger(h.squeeze(0)).unsqueeze(0)


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
    with torch.no_grad():
        for patches in PATCH_TIERS:
            inputs = (zeros(1, patches, tower.patch_proj.in_features), zeros(1, patches, config.hidden_size),
                      zeros(1, 1, patches, half), zeros(1, 1, patches, half), zeros(1, 1, 1, patches))
            program = torch.export.export(tower, inputs).run_decompositions(get_decomp_table())
            converter.add_exported_program(
                program, input_names=["pixel_values", "pos_embeds", "cos", "sin", "attn_mask"],
                output_names=["features"], entrypoint_name=f"p{patches}")
    out_path = Path(out_path)
    converter.to_coreai().save_asset(out_path)
    return {"vision_asset": out_path.name, "vision_source_hash": source_hash(out_path),
            "vision_functions": [f"p{patches}" for patches in PATCH_TIERS],
            "vision_parameter_bytes": sum(p.numel() * p.element_size() for p in tower.parameters())}


class VisionPipeline:
    """The checkpoint's own image processor, and the tower run on the ANE through the host."""

    def __init__(self, checkpoint_dir: Path, host):
        checkpoint_dir = Path(checkpoint_dir)
        self.processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
        self.host = host
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
            capacity = next(p for p in PATCH_TIERS if p >= count)
            thw = grid[None]
            indices, weights = get_vision_interpolation_indices_and_weights(
                thw, num_grid_per_side=int(self.config.num_position_embeddings ** 0.5), mode="bilinear",
                align_corners=True, spatial_merge_size=self.merge)
            pos = (self.pos_weight[indices] * weights[:, :, None]).sum(1)
            cos, sin = self.rotary(pos, get_vision_position_ids(thw, self.merge))
            half = cos.shape[1] // 2

            def padded(x):
                return torch.cat([x.half(), torch.zeros(capacity - count, x.shape[1], dtype=torch.float16)])

            mask = torch.full((1, 1, 1, capacity), MASK_MIN, dtype=torch.float16)
            mask[..., :count] = 0
            out = self.host.vision({
                "pixel_values": padded(pixel_values[start:start + count])[None],
                "pos_embeds": padded(pos)[None],
                "cos": padded(cos[:, :half])[None, None],
                "sin": padded(sin[:, :half])[None, None],
                "attn_mask": mask,
            }, f"p{capacity}")
            features.append(torch.from_numpy(out[0, :count // self.merge**2]).float() * RESIDUAL_SCALE)
            start += count
        return features
