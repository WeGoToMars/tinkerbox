"""What models have stored on this Mac, and removing them.

Each model keeps its Core AI bundle under `bundle.CACHE`, and Core AI keeps a copy of every asset it
compiled for this Mac's Neural Engine in its specialization cache, usually an APFS clone of the bundle's
files. Earlier versions also downloaded the upstream checkpoint into the Hugging Face cache. Python and
its packages are shared by all models. Sizes count each allocated block once, so files that clone or
hard-link others add only the blocks they do not share with files measured before them."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys

from tinkerbox_ane.bundle import CACHE, resources
from tinkerbox_ane.host import BINARY
from tinkerbox_ane.models import CATALOG

# Core AI keeps specializations per macOS build and per client (the host binary), each in a
# folder named after the hex of the asset's hash file, and recreates a missing one on the next load.
SPECIALIZATIONS = Path.home() / "Library/Application Support/coreai-cache"


# fcntl(2): the device offset and contiguous length of a file's data at a logical offset.
F_LOG2PHYS_EXT = 65
LOG2PHYS = struct.Struct("=Iqq")  # flags, contiguous bytes, device offset (packed to 4 bytes)
BLOCK = 4096  # APFS allocation block


class Meter:
    """Allocated bytes of the paths measured so far, counting a block that several files share once."""

    def __init__(self):
        self.inodes = set()
        self.ranges = {}

    def extent(self, device: int, start: int, length: int) -> int:
        start, end = start // BLOCK * BLOCK, -(-(start + length) // BLOCK) * BLOCK
        ranges = self.ranges.setdefault(device, [])
        first = bisect_left(ranges, (start,))
        if first and ranges[first - 1][1] >= start:
            first -= 1
        last, lo, hi, added = first, start, end, end - start
        while last < len(ranges) and ranges[last][0] <= hi:
            a, b = ranges[last]
            added -= max(0, min(end, b) - max(start, a))
            lo, hi = min(lo, a), max(hi, b)
            last += 1
        ranges[first:last] = [(lo, hi)]
        return added

    def __call__(self, path: Path) -> int:
        """Bytes under `path` that no earlier measurement counted, following no symlinks."""
        return sum(self.file(os.path.join(root, name)) for root, _, files in os.walk(path) for name in files)

    def file(self, path: str) -> int:
        try:
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) in self.inodes:
                return 0
            fd = os.open(path, os.O_RDONLY)
            self.inodes.add((info.st_dev, info.st_ino))
        except OSError:
            return 0
        try:
            extents, offset = [], 0
            while offset < info.st_size:
                request = LOG2PHYS.pack(0, info.st_size - offset, offset)
                _, length, device = LOG2PHYS.unpack(fcntl.fcntl(fd, F_LOG2PHYS_EXT, request))
                if length <= 0:
                    break
                extents.append((device, length))
                offset += length
            total = sum(self.extent(info.st_dev, start, length) for start, length in extents)
            return min(total, info.st_blocks * 512)
        except OSError:
            # Compressed or sparse files have no plain mapping; count what they allocate.
            return info.st_blocks * 512
        finally:
            os.close(fd)


def revisions(model: str) -> list[Path]:
    folder = CACHE / model
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.is_symlink()
                  and not p.name.startswith(".")) if folder.is_dir() and not folder.is_symlink() else []


def keys(revision: Path) -> set[str]:
    """The specialization folder names of every Core AI asset in a downloaded revision."""
    found = set()
    for manifest in revision.glob("macos27/*/model.json"):
        try:
            spec = json.loads(manifest.read_text())
            for entry, name, _ in resources(spec):
                found |= {h.read_bytes().hex() for h in (manifest.parent / entry[name]).glob("*.hash")}
        except (OSError, ValueError, KeyError, TypeError):
            continue  # an incomplete download has nothing compiled
    return found


def specializations() -> tuple[dict[str, Path], list[Path]]:
    """This macOS build's specializations by key, and those of earlier builds."""
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    current, earlier = {}, []
    for entry in sorted(SPECIALIZATIONS.glob(f"*/{BINARY.name}/*")):
        if entry.parent.parent.name == build:
            current[entry.name] = entry
        else:
            earlier.append(entry)
    return current, earlier


def checkpoints() -> dict[str, object]:
    """Cached Hugging Face model repositories by id."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return {}
    return {repo.repo_id: repo for repo in cache.repos if repo.repo_type == "model"}


def files(paths: list[Path], size: Meter) -> dict:
    return {"paths": [str(p) for p in paths], "bytes": sum(map(size, paths))}


def model_ids() -> list[str]:
    """Catalog models, then leftovers of earlier versions that are no longer in it."""
    catalog = [entry["id"] for entry in CATALOG]
    found = sorted(p.name for p in CACHE.iterdir() if p.is_dir() and not p.is_symlink()
                   and not p.name.startswith(".")) \
        if CACHE.is_dir() else []
    return catalog + [model for model in found if model not in catalog]


def usage() -> dict:
    """Every model with files on this Mac, Core AI's compiled copies that no downloaded bundle uses,
    and the Python and packages the models share."""
    cached = checkpoints()
    catalog = {entry["id"]: entry for entry in CATALOG}
    current, earlier = specializations()
    size = Meter()  # bundles first, so Core AI's clones of their files add only what they do not share
    used, models = set(), []
    for model in model_ids():
        entry = catalog.get(model, {})
        bundles = []
        for revision in revisions(model):
            compiled = sorted(current[key] for key in keys(revision) if key in current)
            used |= {p.name for p in compiled}
            bundles.append({"revision": revision.name, "path": str(revision), "bytes": size(revision),
                            "current": revision.name == entry.get("revision"), "compiled": files(compiled, size)})
        repo = cached.get(entry.get("original_model"))
        checkpoint = repo and {"repo": repo.repo_id, "path": str(repo.repo_path), "bytes": repo.size_on_disk}
        if bundles or checkpoint:
            total = sum(b["bytes"] + b["compiled"]["bytes"] for b in bundles)
            total += checkpoint["bytes"] if checkpoint else 0
            models.append({"id": model, "name": entry.get("name", model), "bundles": bundles,
                           "checkpoint": checkpoint, "bytes": total})
    unused = files([p for key, p in current.items() if key not in used] + earlier, size)
    # uv links the packages from its cache, shared with other uv projects, into the environment.
    python = Path(sys.base_prefix)
    sites = {Path(p): size(Path(p)) for p in sys.path
             if p.endswith("site-packages") and Path(p).is_dir() and not Path(p).is_relative_to(python)}
    packages = {"path": str(max(sites, key=sites.get).parents[2]) if sites else "", "bytes": sum(sites.values())}
    shared = {"unused_compiled": unused, "python": {"path": str(python), "bytes": size(python)}, "packages": packages}
    return {"models": models, **shared,
            "bytes": sum(m["bytes"] for m in models) + sum(item["bytes"] for item in shared.values())}


def remove(model: str, outdated: bool = False) -> None:
    """Delete a model's bundles, their compiled copies and its upstream checkpoint. With `outdated`,
    delete only bundle revisions other than the catalog's, and keep the checkpoint."""
    if model not in model_ids() or (CACHE / model).is_symlink():
        raise ValueError(f"unknown model {model!r}")
    entry = next((e for e in CATALOG if e["id"] == model), {})
    doomed = [r for r in revisions(model) if not outdated or r.name != entry.get("revision")]
    # Identical assets share a specialization: keep those another remaining bundle still uses.
    kept = [r for other in model_ids() for r in revisions(other) if r not in doomed]
    used = set().union(*map(keys, kept))
    current, earlier = specializations()
    for key in set().union(*map(keys, doomed)) - used:
        for path in [current.get(key), *(p for p in earlier if p.name == key)]:
            if path:
                shutil.rmtree(path, ignore_errors=True)
    for revision in doomed:
        shutil.rmtree(revision)
    if outdated:
        return
    shutil.rmtree(CACHE / model, ignore_errors=True)  # also interrupted installs and downloads
    repo = checkpoints().get(entry.get("original_model"))
    shared = any(e.get("original_model") == entry.get("original_model") and revisions(e["id"])
                 for e in CATALOG if e["id"] != model)
    if repo is not None and not shared:
        from huggingface_hub import scan_cache_dir

        scan_cache_dir().delete_revisions(*(revision.commit_hash for revision in repo.revisions)).execute()


def remove_unused() -> None:
    """Delete compiled copies that no downloaded bundle uses: those of deleted bundles, local
    exports and earlier macOS builds. Core AI compiles a local export again on its next load."""
    for path in usage()["unused_compiled"]["paths"]:
        shutil.rmtree(path, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report, as JSON, what models have stored on this Mac.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--remove", metavar="MODEL", help="delete a model's files first")
    action.add_argument("--remove-unused", action="store_true",
                        help="delete compiled copies that no downloaded bundle uses first")
    parser.add_argument("--outdated", action="store_true",
                        help="with --remove: only revisions the catalog no longer uses")
    args = parser.parse_args(argv)
    if args.outdated and not args.remove:
        parser.error("--outdated requires --remove")
    if args.remove:
        remove(args.remove, args.outdated)
    if args.remove_unused:
        remove_unused()
    print(json.dumps(usage()))


if __name__ == "__main__":
    main()
