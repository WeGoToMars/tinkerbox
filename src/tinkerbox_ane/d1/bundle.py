"""Portable fused bundle manifests and source identity."""

from __future__ import annotations

import json
from pathlib import Path
import re

from tinkerbox_ane.coreai import source_hash, verify_aot


def download_bundle() -> Path:
    from huggingface_hub import snapshot_download

    from tinkerbox_ane.d1 import ARTIFACT_REPO, ARTIFACT_REVISION
    from tinkerbox_ane.d1.host import BUILD_DIR, chip_name, device_architecture, validate_architecture

    architecture = device_architecture()
    validate_architecture(architecture)

    folder = f"macos27/{architecture}"
    root = BUILD_DIR / "models" / ARTIFACT_REVISION
    snapshot_download(ARTIFACT_REPO, revision=ARTIFACT_REVISION,
                      allow_patterns=f"{folder}/*", local_dir=root)
    bundle = root / folder
    if not (bundle / "model.json").is_file():
        chip = chip_name(architecture)
        raise RuntimeError(
            f"Failed to download AOT bundle for {chip} ({architecture}) from {ARTIFACT_REPO}@{ARTIFACT_REVISION}."
        )
    return bundle


def read_bundle(root: Path) -> dict:
    root = Path(root)
    spec = json.loads((root / "model.json").read_text())
    if spec.get("kind") != "d1ane-fused" or spec.get("version") != 2:
        raise ValueError("unsupported fused bundle manifest; re-export and compile the model")
    width, context, tiers = spec["width"], spec["context"], spec["tiers"]
    if (width <= 0 or context <= width or context % width or not tiers
            or tiers != sorted(set(tiers)) or tiers[-1] != context
            or any(t <= width or t > context or t % width for t in tiers)):
        raise ValueError("invalid fused attention capacities")
    for name in ("hidden", "heads", "kv_heads", "head_dim", "conv_taps"):
        if not isinstance(spec[name], int) or spec[name] <= 0:
            raise ValueError(f"invalid {name}")
    if (spec["hidden"] != spec["heads"] * spec["head_dim"]
            or spec["heads"] % spec["kv_heads"] or spec["conv_taps"] < 2):
        raise ValueError("inconsistent model dimensions")
    for name in ("conv_layers", "attention_layers"):
        layers = spec[name]
        if not layers or layers != sorted(set(layers)) or any(i < 0 for i in layers):
            raise ValueError(f"invalid {name}")
    if set(spec["conv_layers"]) & set(spec["attention_layers"]):
        raise ValueError("overlapping convolution and attention layers")
    name = spec["asset"]
    if (not name or name != Path(name).name or "\\" in name or name in {".", ".."}
            or Path(name).suffix not in {".aimodel", ".aimodelc"}):
        raise ValueError(f"unsafe or unsupported asset path: {name!r}")
    expected = spec["source_hash"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or source_hash(root / name) != expected:
        raise ValueError("bundle source identity mismatch")
    if Path(name).suffix == ".aimodelc":
        verify_aot(root / name, expected)
    return spec
