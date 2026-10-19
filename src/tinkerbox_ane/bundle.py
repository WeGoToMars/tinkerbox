"""Portable fused bundle manifests, source identity, download and AOT compilation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import json
from pathlib import Path
import re
import shutil
import tempfile
import warnings

from tinkerbox_ane.coreai import compile_asset, source_hash, verify_aot

# The app checks this layout to show which models are downloaded: <model>/<revision>/macos27/<arch>.
CACHE = Path.home() / "Library/Caches/tinkerbox-ane/models"


def download_bundle(model: str, repo: str, revision: str, progress=None) -> Path:
    from huggingface_hub import HfApi, snapshot_download

    from tinkerbox_ane.host import chip_name, device_architecture, validate_architecture

    architecture = device_architecture()
    validate_architecture(architecture)

    folder = f"macos27/{architecture}"
    root = CACHE / model / revision
    bundle = root / folder
    fetch = dict(repo_id=repo, revision=revision, allow_patterns=f"{folder}/*", local_dir=root)
    try:
        if progress is None:
            snapshot_download(**fetch)
        else:
            files = HfApi().model_info(repo, revision=revision, files_metadata=True).siblings
            total = sum(f.size or 0 for f in files if f.rfilename.startswith(f"{folder}/"))
            with ThreadPoolExecutor(1) as pool:
                done = pool.submit(snapshot_download, **fetch)
                # Partial files grow under the local cache folder until they are moved into place.
                while wait([done], timeout=0.5).not_done:
                    progress(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()), total)
                done.result()
            progress(total, total)
    except Exception as error:
        # Offline, or installed with install_bundle() before the revision is published.
        if not (bundle / "model.json").is_file():
            raise
        warnings.warn(f"using cached {bundle}: {error}", stacklevel=2)
        return bundle
    if not (bundle / "model.json").is_file():
        chip = chip_name(architecture)
        raise RuntimeError(f"Failed to download AOT bundle for {chip} ({architecture}) from {repo}@{revision}.")
    return bundle


def install_bundle(model: str, revision: str, source: Path, kind: str) -> Path:
    """Copy a locally compiled bundle into the cache where download_bundle() expects that revision."""
    spec = read_bundle(source, kind)
    if Path(spec["asset"]).suffix != ".aimodelc" or not spec.get("architecture"):
        raise ValueError("install requires a compiled bundle; compile the export first")
    target = CACHE / model / revision / "macos27" / spec["architecture"]
    target.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the revisions so the app never sees a half-copied bundle.
    with tempfile.TemporaryDirectory(prefix=".install-", dir=CACHE / model) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in (spec["asset"], spec.get("vision_asset"), "model.json"):
            if name:
                copy = shutil.copytree if (Path(source) / name).is_dir() else shutil.copyfile
                copy(Path(source) / name, staged / name)
        if target.exists():
            shutil.rmtree(target)
        staged.rename(target)
    return target


def check_asset_name(name: str) -> None:
    if (not name or name != Path(name).name or "\\" in name or name in {".", ".."}
            or Path(name).suffix not in {".aimodel", ".aimodelc"}):
        raise ValueError(f"unsafe or unsupported asset path: {name!r}")


def read_bundle(root: Path, kind: str) -> dict:
    """The manifest after checking its kind and each asset's identity and placement.

    Model-specific fields are validated by the runtime that uses them."""
    root = Path(root)
    spec = json.loads((root / "model.json").read_text())
    if spec.get("kind") != kind or spec.get("version") != 2:
        raise ValueError("unsupported bundle manifest; re-export and compile the model")
    for name, identity in (("asset", "source_hash"), ("vision_asset", "vision_source_hash")):
        if name not in spec:
            continue
        check_asset_name(spec[name])
        expected = spec[identity]
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or source_hash(root / spec[name]) != expected:
            raise ValueError(f"bundle {name} source identity mismatch")
        if Path(spec[name]).suffix == ".aimodelc":
            verify_aot(root / spec[name], expected)
    return spec


def compile_bundle(
    source: Path,
    out: Path,
    kind: str,
    architecture: str | None = None,
    force: bool = False,
    allow_unsupported: bool = False,
) -> Path:
    """Compile an exported bundle for one architecture, then write its manifest after verification."""
    from tinkerbox_ane.host import device_architecture, validate_architecture

    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError("keep source exports and deployment bundles in separate directories")
    spec = read_bundle(source, kind)
    if Path(spec["asset"]).suffix != ".aimodel":
        raise ValueError("compile requires a source .aimodel bundle")
    if architecture is None:
        architecture = device_architecture()
    if (not architecture or Path(architecture).name != architecture
            or "\\" in architecture or architecture in {".", ".."}):
        raise ValueError(f"invalid architecture: {architecture!r}")
    if not allow_unsupported:
        validate_architecture(architecture)
    target = out / f"fused.{architecture}.aimodelc"
    compile_asset(source / spec["asset"], target, architecture=architecture, min_deployment_version="27.0",
                  expected_functions=(f"c{capacity}" for capacity in spec["tiers"]), force=force)
    spec.update(asset=target.name, architecture=architecture, min_os="27.0", deployment="aot-fused")
    if "vision_asset" in spec:
        vision = out / f"vision.{architecture}.aimodelc"
        compile_asset(source / spec["vision_asset"], vision, architecture=architecture,
                      min_deployment_version="27.0", expected_functions=("main",), force=force,
                      report="compile-vision")
        spec.update(vision_asset=vision.name)
    manifest = out / ".model.json"
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    manifest.replace(out / "model.json")
    return out
