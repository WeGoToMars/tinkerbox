"""Model-independent Core AI AOT compilation, identity and ANE verification."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import time

from tinkerbox_ane.guard import run


def source_hash(asset: Path) -> str:
    asset = Path(asset)
    if asset.suffix == ".aimodelc":
        return json.loads((asset / "metadata.json").read_text())["sourceHash"].lower()
    return (asset / "main.hash").read_bytes().hex()


def count_flops(module, inputs) -> int:
    """Matmul, convolution and attention FLOPs of one call, counted from shapes alone: the inputs
    become fake tensors, so nothing is computed. Elementwise work is not counted."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

    # Masked attention runs as the CPU flash kernel, which PyTorch's counter does not know.
    attention = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu:
                 lambda query, key, value, *_args, **_kwargs: sdpa_flop_count(query, key, value)}
    with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = [mode.from_tensor(x) for x in inputs]
        with FlopCounterMode(display=False, custom_mapping=attention) as counter:
            module(*fake)
    return counter.get_total_flops()


def verify_aot(asset: Path, expected: str) -> None:
    asset = Path(asset)
    if asset.suffix != ".aimodelc" or source_hash(asset) != expected:
        raise ValueError(f"{asset}: AOT asset does not match the source identity")
    manifests = list(asset.rglob("manifest.plist"))
    if len(manifests) != 1:
        raise ValueError(f"{asset}: expected one compiled manifest")
    manifest = plistlib.loads(manifests[0].read_bytes())
    modules = next(iter(manifest["Package Version"].values()))["Optimized Modules"]
    if not modules:
        raise ValueError(f"{asset}: missing compiled modules")
    for module in modules.values():
        entries = module.get("Entry Function Attributes")
        if not isinstance(entries, dict) or not entries or not all(
            "mps.fullyPlacedOnANE" in attributes for attributes in entries.values()
        ):
            raise ValueError(f"{asset}: compiled entries are not fully placed on ANE")


def compile_asset(
    source: Path,
    target: Path,
    *,
    architecture: str,
    min_deployment_version: str,
    platform: str = "macOS",
    expected_functions: Iterable[str] | None = None,
    timeout: int = 600,
    force: bool = False,
    report: str = "compile",
) -> Path:
    """Compile a .aimodel package for Apple Neural Engine with AOT verification.

    Validates source identity, ANE placement, architecture, and deployment
    targets before updating the destination. Watchdog logs and reports are
    written to target.parent/{report}.{json,log}. Set force=True to recompile
    an existing target.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if source.suffix != ".aimodel" or target.suffix != ".aimodelc":
        raise ValueError("compile requires a .aimodel source and .aimodelc target")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("keep source and compiled assets in separate directories")
    if not architecture or not min_deployment_version:
        raise ValueError("architecture and min_deployment_version are required")
    identity = source_hash(source)
    expected = None if expected_functions is None else set(expected_functions)
    target.parent.mkdir(parents=True, exist_ok=True)

    def version_parts(value: str) -> tuple[int, ...]:
        parts = tuple(map(int, value.split(".")))
        return parts + (0,) * (3 - len(parts))

    def inspect_target(asset: Path) -> None:
        verify_aot(asset, identity)
        inspected = subprocess.run(["xcrun", "coreai-build", "inspect", str(asset), "--json"],
                                   capture_output=True, text=True, check=True, timeout=60)
        info = json.loads(inspected.stdout)
        if info.get("supportedArchitectures") != [architecture]:
            raise ValueError(f"{asset}: compiled architecture differs; use force=True to rebuild")
        deployments = (entry.rsplit(" ", 1) for entry in info.get("minSupportedOSVersions", []))
        if not any(name == platform and version_parts(version) == version_parts(min_deployment_version)
                   for name, version in deployments):
            raise ValueError(f"{asset}: compiled deployment target differs; use force=True to rebuild")
        functions = {function["name"] for function in info["summary"]["functions"]}
        if expected is not None and functions != expected:
            raise ValueError(f"compiled function set differs: {functions}")

    started = time.monotonic()
    needs_compile = force or not target.exists()
    action = "Compiling" if needs_compile else "Verifying existing"
    print(f"  [{architecture}] {action} {target.name}", flush=True)
    if needs_compile:
        with tempfile.TemporaryDirectory(prefix=".compile-", dir=target.parent) as temporary:
            produced = Path(temporary) / target.name
            command = ["xcrun", "coreai-build", "compile", str(source), "--architecture", architecture,
                       "--platform", platform, "--min-deployment-version", min_deployment_version,
                       "--preferred-compute", "neural-engine", "--output", str(produced)]
            watchdog = Path(temporary) / report
            print(f"    Compiler log: {watchdog.with_suffix('.log')}", flush=True)
            try:
                check = run(command, watchdog, timeout=timeout)
            finally:
                for suffix in (".json", ".log"):
                    if watchdog.with_suffix(suffix).exists():
                        watchdog.with_suffix(suffix).replace(target.parent / f"{report}{suffix}")
            if not check["passed"]:
                reason = check.get("abort") or f"exit code {check.get('returncode')}"
                raise RuntimeError(f"AOT compilation failed ({reason}); see {report}.json and {report}.log")
            print(f"  [{architecture}] Verifying {target.name}", flush=True)
            inspect_target(produced)
            if target.exists():
                shutil.rmtree(target)
            produced.rename(target)
    else:
        inspect_target(target)
    print(f"  [{architecture}] {target.name}: verified in {time.monotonic() - started:.1f}s", flush=True)
    return target
