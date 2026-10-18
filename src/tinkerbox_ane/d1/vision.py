"""Apple Neural Engine vision tower and multimodal pipeline for d1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector

from tinkerbox_ane.coreai import source_hash
from tinkerbox_ane.d1 import RESIDUAL_SCALE

VISION_MAX_PIXELS = 1024 * 1024
MAX_PATCHES = 1024
PATCH_DIM = 768
VISION_HIDDEN = 1152
PROJECTOR_HIDDEN = 2048
IMAGE_TOKEN_ID = 124907


def cap_pixels(image, max_pixels: int = VISION_MAX_PIXELS):
    """A picture downscaled to at most `max_pixels`, bicubic."""
    image = image.convert("RGB") if hasattr(image, "convert") else image
    if not hasattr(image, "size"):
        raise ValueError(f"invalid image input: {image!r}")
    w, h = image.size
    if w * h <= max_pixels:
        return image
    from PIL import Image

    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)


class VisionTower(nn.Module):
    """Static-shape 27-layer SigLIP2 vision encoder for Apple Neural Engine."""

    def __init__(self, patch_embed: nn.Module, encoder: nn.Module, post_layernorm: nn.Module):
        super().__init__()
        self.patch_embed = patch_embed
        self.encoder = encoder
        self.post_layernorm = post_layernorm

    def forward(self, pixel_values: torch.Tensor, pos_embeds: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # pixel_values: [1, 1024, 768]
        # pos_embeds:   [1, 1024, 1152]
        # attn_mask:    [1, 1, 1, 1024]
        h = self.patch_embed(pixel_values) + pos_embeds
        h = self.encoder(h, attention_mask=attn_mask)[0]
        return self.post_layernorm(h)


def load_vision_tower(checkpoint_dir: Path) -> VisionTower:
    """Load vision tower weights from safetensors checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    state = load_file(checkpoint_dir / "model.safetensors")
    vt_state = {
        k.replace("model.vision_tower.vision_model.", ""): v
        for k, v in state.items()
        if k.startswith("model.vision_tower.vision_model.")
    }
    vt = AutoModel.from_config(config.vision_config)
    vt.load_state_dict(vt_state)
    tower = VisionTower(
        vt.embeddings.patch_embedding,
        vt.encoder,
        vt.post_layernorm,
    ).eval().to(torch.float16)
    return tower


def load_projector(checkpoint_dir: Path) -> Lfm2VlMultiModalProjector:
    """Load multimodal projector weights from safetensors checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    state = load_file(checkpoint_dir / "model.safetensors")
    proj_state = {
        k.replace("model.multi_modal_projector.", ""): v
        for k, v in state.items()
        if k.startswith("model.multi_modal_projector.")
    }
    proj = Lfm2VlMultiModalProjector(config)
    proj.load_state_dict(proj_state)
    return proj.eval().to(torch.float16)


def export_vision_asset(tower: VisionTower, out_path: Path) -> str:
    """Export VisionTower to a Core AI .aimodel asset."""
    from coreai_torch import TorchConverter, get_decomp_table

    out_path = Path(out_path)
    pv = torch.zeros(1, MAX_PATCHES, PATCH_DIM, dtype=torch.float16)
    pe = torch.zeros(1, MAX_PATCHES, VISION_HIDDEN, dtype=torch.float16)
    mask = torch.zeros(1, 1, 1, MAX_PATCHES, dtype=torch.float16)

    with torch.no_grad():
        ep = torch.export.export(tower, (pv, pe, mask)).run_decompositions(get_decomp_table())
        converter = TorchConverter()
        converter.add_exported_program(
            ep, input_names=["pixel_values", "pos_embeds", "attn_mask"], output_names=["features"],
            entrypoint_name="main",
        )
        coreai_model = converter.to_coreai()
    coreai_model.save_asset(out_path)
    return source_hash(out_path)


class VisionPipeline:
    """Preprocesses images, runs ANE vision tower via Host, and projects features."""

    def __init__(self, checkpoint_dir: Path, host: Any):
        checkpoint_dir = Path(checkpoint_dir)
        self.processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.projector = load_projector(checkpoint_dir)
        self.host = host
        state = load_file(checkpoint_dir / "model.safetensors")
        self.pos_weight = state["model.vision_tower.vision_model.embeddings.position_embedding.weight"].to(torch.float32)

    def image_markup(self, count: int) -> str:
        """Chat template markup for `count` images."""
        if count <= 0:
            return ""
        msgs = [{"role": "user", "content": [*([{"type": "image"}] * count), {"type": "text", "text": "\x00"}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        head = "<|im_start|>user\n"
        return text[text.index(head) + len(head):text.index("\x00")]

    def interpolate_pos_embeds(self, spatial_shapes: torch.Tensor) -> torch.Tensor:
        """Bilinearly resize the 16x16 position embedding grid to (H, W) and pad to 1024."""
        # pos_weight: [256, 1152] -> [1, 1152, 16, 16]
        grid_size = int(self.pos_weight.shape[0] ** 0.5)
        pos = self.pos_weight.reshape(grid_size, grid_size, -1).permute(2, 0, 1).unsqueeze(0)
        h, w = spatial_shapes.tolist()
        resized = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        # [1, 1152, H, W] -> [H * W, 1152]
        flat = resized.squeeze(0).permute(1, 2, 0).reshape(h * w, -1)
        out = torch.zeros(MAX_PATCHES, VISION_HIDDEN, dtype=torch.float16)
        out[:h * w] = flat.to(torch.float16)
        return out.unsqueeze(0)

    @torch.inference_mode()
    def encode_image(self, image) -> torch.Tensor:
        """Encode one image using ANE vision tower + projector, scaled by RESIDUAL_SCALE."""
        image = cap_pixels(image)
        inputs = self.processor(text=["<image>"], images=[[image]], return_tensors="pt", add_special_tokens=False)
        pv = inputs["pixel_values"].to(torch.float16)  # [1, 1024, 768]
        pam = inputs["pixel_attention_mask"]  # [1, 1024]
        spatial = inputs["spatial_shapes"][0]  # [2]
        num_patches = int(pam.sum().item())

        pe = self.interpolate_pos_embeds(spatial)  # [1, 1024, 1152]
        # Attention mask: 0 for active, -65504 for padding
        mask = torch.full((1, 1, 1, MAX_PATCHES), -65504.0, dtype=torch.float16)
        mask[0, 0, 0, :num_patches] = 0.0

        # Execute vision tower on ANE
        ane_out = self.host.vision(
            pv[0].numpy(),
            pe[0].numpy(),
            mask[0, 0, 0].numpy(),
        )  # numpy [1024, 1152]
        hidden_state = torch.from_numpy(ane_out).to(torch.float16)

        # Unpad active tokens and reshape to [1, H, W, 1152]
        active = hidden_state[:num_patches].reshape(1, int(spatial[0]), int(spatial[1]), VISION_HIDDEN)
        # Multimodal projection
        features = self.projector(active)  # [1, H//2, W//2, 2048]
        flat_features = features.reshape(-1, PROJECTOR_HIDDEN)
        return flat_features * RESIDUAL_SCALE
