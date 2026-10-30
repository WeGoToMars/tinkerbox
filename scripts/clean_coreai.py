#!/usr/bin/env python3
"""Offline Core AI disk cleanup. Preview by default; --apply deletes the listed paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
GB = 1_000_000_000


def active_clients(names: set[str]) -> list[str]:
    # Keep comm separate: executable paths may contain spaces in ps output.
    processes = {}
    for column in ("comm", "args"):
        result = subprocess.run(["ps", "-axo", f"pid=,{column}="],
                                capture_output=True, text=True, check=True)
        processes[column] = dict(line.strip().split(maxsplit=1)
                                 for line in result.stdout.splitlines() if line.strip())
    active = []
    for pid, executable in processes["comm"].items():
        if int(pid) == os.getpid():
            continue
        arguments = processes["args"].get(pid, executable)
        name = Path(executable).name
        if (name in names | {"coreai-build", "TinkerboxDemo", "ANECompilerService",
                             "ANELargeModelCompilerService"}
                or name.startswith("tinkerbox-ane-host")
                or (name.startswith("python") and (
                    re.search(r"\btinkerbox_ane\..*(?:guard|compile|export|validation)", arguments)
                    or re.search(r"(?:^|/)worker\.py(?:\s|$)", arguments)))):
            active.append(f"PID {pid}: {arguments}")
    return active


def staging(root: Path) -> list[Path]:
    """Abandoned host and compiler staging directories, without descending into assets or apps."""
    found = []
    for parent, directories, _ in os.walk(root):
        for name in directories[:]:
            if name.startswith((".host-", ".compile-")):
                found.append(Path(parent) / name)
                directories.remove(name)
            elif Path(name).suffix in {".aimodel", ".aimodelc", ".app"}:
                directories.remove(name)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete after preflight; never kills clients")
    parser.add_argument("--include-build", action="store_true", help="also remove this repo's artifacts/build")
    parser.add_argument("--include-models", action="store_true",
                        help="also remove Tinkerbox's model cache, including locally installed bundles")
    args = parser.parse_args()

    home = Path.home()
    persistent = home / "Library/Application Support/coreai-cache"
    models = home / "Library/Caches/tinkerbox-ane"
    targets = [("persistent specialization", persistent),
               ("Core AI locks/temp", home / "Library/Caches/coreai-cache")]
    names = {"tinkerbox-ane-host"}
    if persistent.is_dir():
        for version in persistent.iterdir():
            if version.is_dir():
                names.update(client.name for client in version.iterdir() if client.is_dir())

    # Only MPSGraph scratch directories whose creating PID has exited.
    graph = Path(tempfile.gettempdir()) / "com.apple.MetalPerformanceShadersGraphHost"
    for path in sorted(graph.iterdir()) if graph.is_dir() else []:
        match = re.fullmatch(r"mpsgraph-(\d+)-.+", path.name)
        if match and path.is_dir():
            try:
                os.kill(int(match[1]), 0)
            except ProcessLookupError:
                targets.append(("orphan MPSGraph scratch", path))
            except PermissionError:
                pass  # Treat inaccessible/reused PIDs as alive.

    if args.include_build:
        targets.append(("build outputs", REPO / "artifacts/build"))
    if args.include_models:
        targets.append(("downloaded/installed models", models))
    for root in (REPO / "artifacts", models):
        targets.extend(("abandoned host/compiler staging", path) for path in staging(root))

    # Parents subsume staging children; do not count or delete a child twice.
    selected: list[tuple[str, Path]] = []
    for label, path in sorted(targets, key=lambda item: len(item[1].parts)):
        if (path.exists() or path.is_symlink()) and not any(
            path == parent or parent in path.parents for _, parent in selected
        ):
            selected.append((label, path))
    before = shutil.disk_usage(home).free
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — free now: {before / GB:.1f} GB")
    total = 0
    if selected:
        # One du invocation avoids counting hard-linked files multiple times.
        usage = subprocess.run(["du", "-k", "-s", *[str(p) for _, p in selected]],
                               capture_output=True, text=True, check=True)
        sizes = dict((line.split("\t", 1)[1], int(line.split("\t", 1)[0]) * 1024)
                     for line in usage.stdout.splitlines())
        for label, path in selected:
            size = sizes[str(path)]
            total += size
            print(f"  {size / GB:7.2f} GB  {label}: {path}")
    print(f"Selected allocated size: {total / GB:.1f} GB (APFS clones and snapshots can change actual recovery)")
    print("Preserved: other artifacts/research, Hugging Face checkpoints, uv/pip, and system ANE caches.")

    clients = active_clients(names)
    if clients:
        print("\nClose Core AI clients and stop benchmarks/exports/compiles before cleanup:")
        print("\n".join(clients))
        if args.apply:
            print("Refusing cleanup; nothing deleted.", file=sys.stderr)
            return 1
    if not args.apply:
        print("\nNothing deleted. Add --apply to perform this plan.")
        return 0

    for _, path in selected:
        shutil.rmtree(path)  # Refuses a symbolic link rather than following it.
        print(f"Removed {path}", flush=True)
    print(f"Free now: {shutil.disk_usage(home).free / GB:.1f} GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Cleanup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
