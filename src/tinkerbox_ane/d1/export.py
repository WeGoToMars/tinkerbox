"""Export one fused FP16 backbone with shared weights across all cache tiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from coreai_torch import TorchConverter, get_decomp_table

from tinkerbox_ane.coreai import source_hash
from tinkerbox_ane.d1 import CHUNK, CONTEXT, INPUT, KIND, OUTPUT, REPO, REVISION, SOURCE, TIERS, checkpoint
from tinkerbox_ane.d1.model import FusedBackbone, load_weights
from tinkerbox_ane.d1.vision import export_vision_asset, load_vision_tower


def export_vision(out: Path = SOURCE) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tower = load_vision_tower(checkpoint())
    vision_asset = out / "vision.aimodel"
    vhash = export_vision_asset(tower, vision_asset)
    manifest_path = out / "model.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"kind": KIND, "version": 2}
    manifest.update(
        vision_asset=vision_asset.name,
        vision_source_hash=vhash,
        vision_parameter_bytes=sum(p.numel() * p.element_size() for p in tower.parameters()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported vision tower ({time.monotonic() - started:.1f}s): {out}", flush=True)
    return out


def export(out: Path = SOURCE, include_vision: bool = True) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("source output is not empty; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    backbone = load_weights(checkpoint())
    cfg = backbone.cfg
    converter = TorchConverter()
    # Reuse the same Parameter objects so each tier shares one weight resource.
    with torch.no_grad():
        for capacity in TIERS:
            model = FusedBackbone(backbone, capacity).eval()
            program = torch.export.export(model, model.example_inputs()).run_decompositions(get_decomp_table())
            converter.add_exported_program(
                program, input_names=[INPUT, "cos", "sin", "past_mask", "packed_state", "chunk_mask", "conv_route"],
                output_names=[OUTPUT, "packed_state_out"], entrypoint_name=f"c{capacity}",
            )
            print(f"staged c{capacity} ({time.monotonic() - started:.1f}s)", flush=True)
    asset = out / "fused.aimodel"
    converter.to_coreai().save_asset(asset)
    manifest = {
        "kind": KIND, "version": 2, "width": CHUNK, "context": CONTEXT,
        "hidden": cfg.hidden, "heads": cfg.heads, "kv_heads": cfg.kv_heads,
        "head_dim": cfg.head_dim, "conv_taps": cfg.conv_taps, "tiers": list(TIERS),
        "conv_layers": model.conv_layers, "attention_layers": model.attention_layers,
        "asset": asset.name, "source_hash": source_hash(asset),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in backbone.parameters()),
        "source": {"repo": REPO, "revision": REVISION},
    }
    if include_vision:
        tower = load_vision_tower(checkpoint())
        vision_asset = out / "vision.aimodel"
        vhash = export_vision_asset(tower, vision_asset)
        manifest.update(
            vision_asset=vision_asset.name,
            vision_source_hash=vhash,
            vision_parameter_bytes=sum(p.numel() * p.element_size() for p in tower.parameters()),
        )
    (out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported fused backbone ({time.monotonic() - started:.1f}s): {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SOURCE)
    parser.add_argument("--vision-only", action="store_true", help="Export only vision tower")
    args = parser.parse_args()
    if args.vision_only:
        export_vision(args.out)
    else:
        export(args.out)


if __name__ == "__main__":
    main()
