"""Export Qwen3.5 stages with shared weights across attention capacities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from coreai_torch import TorchConverter, get_decomp_table

from tinkerbox_ane.bundle import checkpoint_digest, write_checkpoint
from tinkerbox_ane.coreai import count_flops, source_hash
from tinkerbox_ane.d3 import CHUNK, INPUT, KIND, OUTPUT, PROMPT_WIDTHS, RUNTIME_TENSORS, VARIANTS, checkpoint
from tinkerbox_ane.d3.model import AttendStage, PromptStage, ScanStage, load_weights, stage_ranges
from tinkerbox_ane.d3.vision import export_vision_asset


def export(model: str, out: Path, include_vision: bool = True) -> Path:
    repo, revision, tiers = VARIANTS[model]
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("source output is not empty; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    path = checkpoint(repo, revision)
    decision = json.loads((path / "decision_config.json").read_text())
    if decision["attention_mode"] != "noncausal_full_attention":
        raise ValueError("d3 export requires noncausal_full_attention")
    backbone = load_weights(path)
    cfg = backbone.cfg
    assets = []
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float16)
    with torch.no_grad():
        for i, (start, end) in enumerate(stage_ranges(cfg)):
            converter = TorchConverter()
            scan = ScanStage(backbone, start, end).eval()
            inputs = (zeros(1, cfg.hidden, 1, CHUNK), zeros(1, 1, cfg.rotary_dim // 2, CHUNK),
                      zeros(1, 1, cfg.rotary_dim // 2, CHUNK), zeros(1, scan.channels, 1, CHUNK))
            # Matmul, convolution and attention FLOPs of each function, for the runtime's compute rate.
            flops = {f"scan{i}": count_flops(scan, inputs)}
            program = torch.export.export(scan, inputs).run_decompositions(get_decomp_table())
            # Keep entrypoint names distinct across assets: avoid specialization cache collisions.
            converter.add_exported_program(
                program, input_names=[INPUT, "cos", "sin", "packed_state"],
                output_names=[OUTPUT, "query", "gate", "key", "value", "packed_state_out"],
                entrypoint_name=f"scan{i}",
            )
            for capacity in tiers:
                attend = AttendStage(backbone, end, capacity).eval()
                inputs = (zeros(1, cfg.hidden, 1, CHUNK), zeros(1, cfg.heads, CHUNK, cfg.head_dim),
                          zeros(1, cfg.heads, CHUNK, cfg.head_dim), zeros(1, cfg.kv_heads, capacity, cfg.head_dim),
                          zeros(1, cfg.kv_heads, capacity, cfg.head_dim), zeros(1, 1, 1, capacity))
                flops[f"attend{i}c{capacity}"] = count_flops(attend, inputs)
                program = torch.export.export(attend, inputs).run_decompositions(get_decomp_table())
                converter.add_exported_program(
                    program, input_names=[INPUT, "query", "gate", "key", "value", "mask"],
                    output_names=[OUTPUT], entrypoint_name=f"attend{i}c{capacity}",
                )
            for width in PROMPT_WIDTHS:
                prompt = PromptStage(backbone, start, end, width).eval()
                inputs = (zeros(1, cfg.hidden, 1, width), zeros(1, 1, cfg.rotary_dim // 2, width),
                          zeros(1, 1, cfg.rotary_dim // 2, width), zeros(1, 1, 1, width))
                flops[f"prompt{i}c{width}"] = count_flops(prompt, inputs)
                program = torch.export.export(prompt, inputs).run_decompositions(get_decomp_table())
                converter.add_exported_program(
                    program, input_names=[INPUT, "cos", "sin", "mask"], output_names=[OUTPUT],
                    entrypoint_name=f"prompt{i}c{width}",
                )
            asset = out / f"stage{i}.aimodel"
            converter.to_coreai().save_asset(asset)
            assets.append({"asset": asset.name, "source_hash": source_hash(asset),
                           "functions": [f"scan{i}"] + [f"attend{i}c{capacity}" for capacity in tiers]
                           + [f"prompt{i}c{width}" for width in PROMPT_WIDTHS], "flops": flops})
            print(f"exported layers {start}..{end} ({time.monotonic() - started:.1f}s)", flush=True)
    manifest = {
        "kind": KIND, "version": 2, "width": CHUNK, "context": tiers[-1], "tiers": list(tiers),
        "prompts": list(PROMPT_WIDTHS),
        "hidden": cfg.hidden, "heads": cfg.heads, "kv_heads": cfg.kv_heads, "head_dim": cfg.head_dim,
        "conv_taps": cfg.conv_taps, "rotary_dim": cfg.rotary_dim,
        "stages": [end - start for start, end in stage_ranges(cfg)],
        "attention_mode": decision["attention_mode"], "assets": assets,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in backbone.parameters()),
        "source": {"repo": repo, "revision": revision},
    }
    if include_vision:
        manifest.update(export_vision_asset(path, out / "vision.aimodel"))
        print(f"exported vision tower ({time.monotonic() - started:.1f}s)", flush=True)
    manifest["checkpoint_hash"] = checkpoint_digest(write_checkpoint(path, out, RUNTIME_TENSORS))
    (out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {model} ({time.monotonic() - started:.1f}s): {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(VARIANTS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-vision", action="store_true", help="Export the text stages only")
    args = parser.parse_args()
    export(args.model, args.out, include_vision=not args.no_vision)


if __name__ == "__main__":
    main()
