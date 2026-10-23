"""Portable fused bundle manifests, source identity, download and AOT compilation."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, wait
import ctypes
import hashlib
import importlib
from importlib.machinery import SOURCE_SUFFIXES, FileFinder, SourceFileLoader
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
import warnings

from tinkerbox_ane.coreai import compile_asset, source_hash, verify_aot

# The app checks this layout to show which models are downloaded: <model>/<revision>/macos27/<arch>.
CACHE = Path.home() / "Library/Caches/tinkerbox-ane/models"


def download_bundle(model: str, repo: str, revision: str, progress=None) -> Path:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    from tinkerbox_ane.host import chip_name, device_architecture, validate_architecture

    architecture = device_architecture()
    validate_architecture(architecture)

    folder = f"macos27/{architecture}"
    root = CACHE / model / revision
    bundle = root / folder

    def fetch():
        # The manifest comes last, so a bundle that has one is complete.
        snapshot_download(repo, revision=revision, allow_patterns=["config.json", f"{folder}/*"],
                          ignore_patterns=[f"{folder}/model.json"], local_dir=root)
        hf_hub_download(repo, f"{folder}/model.json", revision=revision, local_dir=root)

    try:
        if progress is None:
            fetch()
        else:
            files = HfApi().model_info(repo, revision=revision, files_metadata=True).siblings
            total = sum(f.size or 0 for f in files
                        if f.rfilename == "config.json" or f.rfilename.startswith(f"{folder}/"))
            with ThreadPoolExecutor(1) as pool:
                done = pool.submit(fetch)
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
    if not compiled(spec) or not spec.get("architecture"):
        raise ValueError("install requires a compiled bundle; compile the export first")
    target = CACHE / model / revision / "macos27" / spec["architecture"]
    target.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the revisions so the app never sees a half-copied bundle.
    with tempfile.TemporaryDirectory(prefix=".install-", dir=CACHE / model) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in [*asset_names(spec), CHECKPOINT, "model.json"]:
            clone(Path(source) / name, staged / name)
        replace_directory(staged, target)
    return target


# The upstream files a runtime reads, kept in each bundle so loading downloads only the converted repository.
CHECKPOINT = "checkpoint"


def write_checkpoint(upstream: Path, out: Path, tensors: Iterable[str]) -> Path:
    """Copy upstream's code, tokenizer, configs and other small files into `out/checkpoint`, with a
    model.safetensors that keeps only `tensors`, by name; a name ending in "." keeps every tensor under it."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    upstream, out = Path(upstream), Path(out)
    index = upstream / "model.safetensors.index.json"
    weights = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() \
        else ["model.safetensors"]
    wanted = list(tensors)

    def matches(key, name):
        return key == name or (name.endswith(".") and key.startswith(name))

    selected = {}
    for name in weights:
        with safe_open(upstream / name, "pt") as f:
            selected.update({key: f.get_tensor(key) for key in f.keys() if any(matches(key, n) for n in wanted)})
    if missing := [name for name in wanted if not any(matches(key, name) for key in selected)]:
        raise ValueError(f"{upstream} lacks tensors {missing}")
    staged = Path(tempfile.mkdtemp(prefix=f".{CHECKPOINT}-", dir=out))
    try:
        for path in sorted(upstream.rglob("*")):
            relative = path.relative_to(upstream)
            if (path.is_file() and str(relative) not in weights and path != index
                    and not any(part.startswith(".") or part == "__pycache__" for part in relative.parts)):
                (staged / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, staged / relative)
        save_file(selected, staged / "model.safetensors", metadata={"format": "pt"})
        replace_directory(staged, out / CHECKPOINT)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return out / CHECKPOINT


def code_digest(checkpoint: Path) -> str:
    """SHA-256 over a checkpoint's Python files, which loading imports."""
    digest = hashlib.sha256()
    for path in sorted(Path(checkpoint).rglob("*.py")):
        relative = path.relative_to(checkpoint)
        if not any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            digest.update(str(relative).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def checkpoint_digest(checkpoint: Path) -> str:
    """Release identity of the CPU tensors, configuration and source files, excluding local caches."""
    digest = hashlib.sha256()
    for path in sorted(Path(checkpoint).rglob("*")):
        relative = path.relative_to(checkpoint)
        if (not path.is_file() or path.suffix == ".pyc"
                or any(part.startswith(".") or part == "__pycache__" for part in relative.parts)):
            continue
        digest.update(str(relative).encode() + b"\0")
        with path.open("rb") as stream:
            while data := stream.read(1024 * 1024):
                digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def replace_directory(staged: Path, target: Path) -> None:
    """Publish a complete directory; macOS RENAME_SWAP keeps an existing target continuously available."""
    if not target.exists():
        staged.rename(target)
        return
    if ctypes.CDLL(None, use_errno=True).renamex_np(os.fsencode(staged), os.fsencode(target), 2):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))
    shutil.rmtree(staged)


def clone(source: Path, target: Path) -> None:
    """Copy a file or folder as an APFS clone, which shares blocks until either changes."""
    if ctypes.CDLL(None, use_errno=True).clonefile(os.fsencode(source), os.fsencode(target), 0):
        # Another volume, or a filesystem without clones.
        (shutil.copytree if Path(source).is_dir() else shutil.copyfile)(source, target)


def bundled_checkpoint(bundle: Path, code: str | None = None) -> Path:
    """A bundle's checkpoint files; with `code`, only if its Python is that pinned upstream release's."""
    path = Path(bundle) / CHECKPOINT
    if not (path / "model.safetensors").is_file():
        raise ValueError("bundle lacks its checkpoint files; re-export the model")
    if code is not None and code_digest(path) != code:
        raise ValueError("bundle's checkpoint code is not the pinned upstream release")
    return path


class _SourceOnly(SourceFileLoader):
    """Compiles the .py file itself, never cached bytecode."""

    def get_code(self, fullname):
        return self.source_to_code(self.get_data(self.path), self.path)


def checkpoint_modules(checkpoint: Path, package: str, *names: str) -> list[types.ModuleType]:
    """Import a checkpoint's Python files as `package`. Only their sources load: an extension module
    or bytecode beside them would bypass the digest that pinned them."""
    if package not in sys.modules:
        path = str(checkpoint)
        sys.path_importer_cache[path] = FileFinder(path, (_SourceOnly, SOURCE_SUFFIXES))
        module = types.ModuleType(package)
        module.__path__ = [path]
        sys.modules[package] = module
    return [importlib.import_module(f"{package}.{name}") for name in names]


def check_asset_name(name: str) -> None:
    if (not name or name != Path(name).name or "\\" in name or name in {".", ".."}
            or Path(name).suffix not in {".aimodel", ".aimodelc"}):
        raise ValueError(f"unsafe or unsupported asset path: {name!r}")


def resources(spec: dict) -> list[tuple[dict, str, str]]:
    """Each Core AI asset as (manifest entry, name key, identity key): the text asset, or one per
    stage, then the vision tower."""
    found = [(spec, "asset", "source_hash")] if "asset" in spec else []
    found += [(stage, "asset", "source_hash") for stage in spec.get("assets", [])]
    if "vision_asset" in spec:
        found.append((spec, "vision_asset", "vision_source_hash"))
    return found


def asset_names(spec: dict) -> list[str]:
    return [entry[name] for entry, name, _ in resources(spec)]


def compiled(spec: dict) -> bool:
    return all(Path(name).suffix == ".aimodelc" for name in asset_names(spec))


def read_bundle(root: Path, kind: str) -> dict:
    """The manifest after checking its kind and each asset's identity and placement.

    Model-specific fields are validated by the runtime that uses them."""
    root = Path(root)
    spec = json.loads((root / "model.json").read_text())
    if spec.get("kind") != kind or spec.get("version") != 2:
        raise ValueError("unsupported bundle manifest; re-export and compile the model")
    if not resources(spec) or ("asset" in spec) == ("assets" in spec):
        raise ValueError("bundle needs one text asset or a list of stage assets")
    names = asset_names(spec)
    if len(set(names)) != len(names):
        raise ValueError("bundle assets must be distinct")
    for entry, name, identity in resources(spec):
        check_asset_name(entry[name])
        expected = entry[identity]
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or source_hash(root / entry[name]) != expected):
            raise ValueError(f"bundle {name} source identity mismatch")
        if Path(entry[name]).suffix == ".aimodelc":
            verify_aot(root / entry[name], expected)
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
    if any(Path(name).suffix != ".aimodel" for name in asset_names(spec)):
        raise ValueError("compile requires a source .aimodel bundle")
    checkpoint = bundled_checkpoint(source)
    checkpoint_hash = checkpoint_digest(checkpoint)
    if spec.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("checkpoint identity mismatch; re-export the bundle")
    if architecture is None:
        architecture = device_architecture()
    if (not architecture or Path(architecture).name != architecture
            or "\\" in architecture or architecture in {".", ".."}):
        raise ValueError(f"invalid architecture: {architecture!r}")
    if not allow_unsupported:
        validate_architecture(architecture)
    # Keep verified stages after a compiler failure, without touching the deployment bundle.
    build = out.with_name(f".{out.name}.build")
    identity = {"source": spec, "architecture": architecture, "force": force}
    marker = build / ".source.json"
    if build.exists() and (not marker.is_file() or json.loads(marker.read_text()) != identity):
        shutil.rmtree(build)
    if not build.exists():
        if out.exists() and not force:
            clone(out, build)
        else:
            build.mkdir(parents=True)
        marker.write_text(json.dumps(identity))
    if "asset" in spec:
        target = build / f"fused.{architecture}.aimodelc"
        compile_asset(source / spec["asset"], target, architecture=architecture, min_deployment_version="27.0",
                      expected_functions=(f"c{capacity}" for capacity in spec["tiers"]))
        spec.update(asset=target.name)
    for stage in spec.get("assets", []):
        target = build / f"{Path(stage['asset']).stem}.{architecture}.aimodelc"
        compile_asset(source / stage["asset"], target, architecture=architecture, min_deployment_version="27.0",
                      expected_functions=stage["functions"], timeout=3600,
                      report=f"compile-{Path(stage['asset']).stem}")
        stage.update(asset=target.name)
    spec.update(architecture=architecture, min_os="27.0",
                deployment="aot-staged" if "assets" in spec else "aot-fused")
    if "vision_asset" in spec:
        vision = build / f"vision.{architecture}.aimodelc"
        compile_asset(source / spec["vision_asset"], vision, architecture=architecture,
                      min_deployment_version="27.0", expected_functions=spec.get("vision_functions", ("main",)),
                      report="compile-vision")
        spec.update(vision_asset=vision.name)
    staged = build / f".{CHECKPOINT}"
    shutil.rmtree(staged, ignore_errors=True)
    clone(checkpoint, staged)
    replace_directory(staged, build / CHECKPOINT)
    manifest = build / "model.json"
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    marker.unlink()
    replace_directory(build, out)
    return out
