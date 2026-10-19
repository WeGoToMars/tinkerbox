"""Apple Neural Engine vision tower and multimodal pipeline for d1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector

from tinkerbox_ane.coreai import source_hash
from tinkerbox_ane.d1 import RESIDUAL_SCALE

MAX_PATCHES = 1024
PATCH_DIM = 768
VISION_HIDDEN = 1152
PROJECTOR_HIDDEN = 2048
IMAGE_TOKEN_ID = 124907


def as_rgb(image):
    """A Pillow picture in RGB; the processor resizes and tiles it."""
    if not hasattr(image, "convert"):
        raise ValueError(f"invalid image input: {image!r}")
    return image.convert("RGB")


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


def checkpoint_tensors(checkpoint_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    """The checkpoint tensors under `prefix`, without reading the rest of the weights."""
    with safe_open(Path(checkpoint_dir) / "model.safetensors", "pt") as f:
        return {k.removeprefix(prefix): f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}


def load_vision_tower(checkpoint_dir: Path) -> VisionTower:
    """Load vision tower weights from safetensors checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    vt = AutoModel.from_config(config.vision_config)
    vt.load_state_dict(checkpoint_tensors(checkpoint_dir, "model.vision_tower.vision_model."))
    tower = VisionTower(
        vt.embeddings.patch_embedding,
        vt.encoder,
        vt.post_layernorm,
    ).eval().to(torch.float16)
    return tower


def load_projector(checkpoint_dir: Path) -> Lfm2VlMultiModalProjector:
    """Load multimodal projector weights from safetensors checkpoint.

    Runs on CPU in FP32 to utilize Accelerate's sgemm matrix kernels, which lack
    optimized FP16 implementations on Apple Silicon CPU."""
    checkpoint_dir = Path(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    proj = Lfm2VlMultiModalProjector(config)
    proj.load_state_dict(checkpoint_tensors(checkpoint_dir, "model.multi_modal_projector."))
    return proj.eval().float()


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
    """Preprocesses images, runs ANE vision tower via Host, and projects features on the CPU."""

    def __init__(self, checkpoint_dir: Path, host: Any):
        checkpoint_dir = Path(checkpoint_dir)
        self.processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.projector = load_projector(checkpoint_dir)
        self.host = host
        self.pos_weight = checkpoint_tensors(
            checkpoint_dir, "model.vision_tower.vision_model.embeddings.position_embedding.")["weight"].float()

    def limit_tiles(self, tiles: int):
        """Split pictures into at most `tiles` tiles; one tile resizes instead of splitting."""
        processor = self.processor.image_processor
        processor.min_tiles, processor.max_tiles = (1, 1) if tiles == 1 else (2, tiles)

    def layout(self, image) -> dict:
        """How one picture reaches the model: its tile grid, the size it is resized to, and its overview.

        Nothing is cropped; a split picture is resized to fill its grid, and the overview is the whole
        picture at a single view's size."""
        processor = self.processor.image_processor
        out = processor(images=[[image]], return_tensors="pt")
        rows, cols = int(out["image_rows"][0]), int(out["image_cols"][0])
        height, width = (int(n) for n in out["image_sizes"][0])
        shapes = out["spatial_shapes"].tolist()
        factor = processor.downsample_factor
        tiled = rows * cols > 1
        return {"source": list(image.size), "cols": cols, "rows": rows,
                "size": [cols * processor.tile_size, rows * processor.tile_size] if tiled else [width, height],
                "overview": [width, height] if len(shapes) > rows * cols else None,
                "passes": len(shapes), "tokens": sum((h // factor) * (w // factor) for h, w in shapes)}

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
        """Encode one image using ANE vision tower + projector, scaled by RESIDUAL_SCALE.

        The processor splits large pictures into tiles plus a thumbnail. Each tile runs on its own,
        and the features follow the order of the image placeholder tokens, as upstream."""
        image = as_rgb(image)
        inputs = self.processor(text=["<image>"], images=[[image]], return_tensors="pt", add_special_tokens=False)
        features = []
        # pixel_values: [tiles, 1024, 768]; pixel_attention_mask: [tiles, 1024]; spatial_shapes: [tiles, 2]
        for pv, pam, spatial in zip(inputs["pixel_values"], inputs["pixel_attention_mask"], inputs["spatial_shapes"]):
            num_patches = int(pam.sum().item())
            pe = self.interpolate_pos_embeds(spatial)  # [1, 1024, 1152]
            # Attention mask: 0 for active, -65504 for padding
            mask = torch.full((1, 1, 1, MAX_PATCHES), -65504.0, dtype=torch.float16)
            mask[0, 0, 0, :num_patches] = 0.0

            # Execute vision tower on ANE
            ane_out = self.host.vision({"pixel_values": pv[None].to(torch.float16), "pos_embeds": pe, "attn_mask": mask})
            hidden_state = torch.from_numpy(ane_out[0])  # [1024, 1152]

            # Unpad active tokens and reshape to [1, H, W, 1152]
            active = hidden_state[:num_patches].reshape(1, int(spatial[0]), int(spatial[1]), VISION_HIDDEN)
            # Multimodal projection: [1, H//2, W//2, 2048]
            features.append(self.projector(active.float()).reshape(-1, PROJECTOR_HIDDEN))
        return torch.cat(features) * RESIDUAL_SCALE
