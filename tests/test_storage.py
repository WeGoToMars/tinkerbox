import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def specialization(home, build, key, weights=None):
    """Core AI's compiled copy of an asset: an APFS clone of its weights, as Core AI makes, or new data."""
    root = home / "Library/Application Support/coreai-cache" / build / "tinkerbox-ane-host" / key / "options"
    root.mkdir(parents=True)
    if weights:
        subprocess.run(["cp", "-c", str(weights), str(root / "weights.bin")], check=True)
    else:
        (root / "model.aimodelx").write_bytes(b"\1" * 50_000)
    return root.parent


def bundle(home, build, model, revision):
    """A downloaded bundle with one asset, and Core AI's compiled copy of it."""
    root = home / "Library/Caches/tinkerbox-ane/models" / model / revision / "macos27/h17s"
    (root / "stage.aimodelc").mkdir(parents=True)
    (root / "stage.aimodelc/weights.bin").write_bytes(b"\1" * 100_000)
    digest = hashlib.sha256(f"{model}{revision}".encode()).digest()
    (root / "stage.aimodelc/main.hash").write_bytes(digest)
    (root / "model.json").write_text(json.dumps({"assets": [{"asset": "stage.aimodelc", "source_hash": "x"}]}))
    return root.parents[1], specialization(home, build, digest.hex(), root / "stage.aimodelc/weights.bin")


def checkpoint(hub, repo):
    root = hub / f"models--{repo.replace('/', '--')}"
    commit = hashlib.sha1(repo.encode()).hexdigest()
    (root / "blobs").mkdir(parents=True)
    (root / "blobs/weights").write_bytes(b"\1" * 200_000)
    (root / "snapshots" / commit).mkdir(parents=True)
    (root / "snapshots" / commit / "model.safetensors").symlink_to("../../blobs/weights")
    (root / "refs").mkdir()
    (root / "refs/main").write_text(commit)
    return root


def storage(home, *arguments):
    environment = {**os.environ, "HOME": str(home), "HF_HOME": str(home / "hf"), "HF_HUB_CACHE": str(home / "hub")}
    result = subprocess.run([sys.executable, "-m", "tinkerbox_ane.storage", *arguments], env=environment,
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    return {model["id"]: model for model in report["models"]}, set(report["unused_compiled"]["paths"])


def test_report_and_remove_a_model_with_everything_it_downloaded(tmp_path: Path):
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True, check=True).stdout.strip()
    old, old_compiled = bundle(tmp_path, build, "d3-nano", "v2.0.0")
    current, current_compiled = bundle(tmp_path, build, "d3-nano", "v2.1.0")
    other, other_compiled = bundle(tmp_path, build, "d1-3b", "v2.0.1")
    nano = checkpoint(tmp_path / "hub", "vllm-sr/d3-nano")
    d1 = checkpoint(tmp_path / "hub", "LiquidAI/d1-3B")
    unused = {specialization(tmp_path, build, "ab" * 32), specialization(tmp_path, "25A000", other_compiled.name)}

    models, found = storage(tmp_path)
    assert set(models) == {"d1-3b", "d3-nano"}
    bundles = [(b["revision"], b["current"], b["compiled"]["paths"]) for b in models["d3-nano"]["bundles"]]
    assert bundles == [("v2.0.0", False, [str(old_compiled)]), ("v2.1.0", True, [str(current_compiled)])]
    assert models["d3-nano"]["checkpoint"]["repo"] == "vllm-sr/d3-nano"
    # Clones share the bundle's blocks, so compiled copies add nothing to the model's size.
    assert all(b["compiled"]["bytes"] == 0 for b in models["d3-nano"]["bundles"])
    assert 2 * 100_000 + 200_000 <= models["d3-nano"]["bytes"] < 2 * 100_000 + 200_000 + 50_000
    assert found == set(map(str, unused))

    models, _ = storage(tmp_path, "--remove", "d3-nano", "--outdated")
    assert [b["revision"] for b in models["d3-nano"]["bundles"]] == ["v2.1.0"]
    assert not old.exists() and not old_compiled.exists()
    assert current.exists() and current_compiled.exists() and nano.exists()

    models, _ = storage(tmp_path, "--remove", "d3-nano")
    assert set(models) == {"d1-3b"}
    assert not current.parent.exists() and not current_compiled.exists() and not nano.exists()
    assert other.exists() and other_compiled.exists() and d1.exists()

    models, found = storage(tmp_path, "--remove-unused")
    assert found == set() and not any(path.exists() for path in unused)
    assert models["d1-3b"]["bundles"][0]["compiled"]["paths"] == [str(other_compiled)]


def test_removal_stays_inside_real_model_directories(tmp_path):
    victim = tmp_path / "unrelated"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    cache = tmp_path / "Library/Caches/tinkerbox-ane/models"
    cache.mkdir(parents=True)
    (cache / "linked-model").symlink_to(victim, target_is_directory=True)
    for name in [str(victim), "../../../../unrelated", "linked-model"]:
        with pytest.raises(subprocess.CalledProcessError):
            storage(tmp_path, "--remove", name)
        assert (victim / "keep.txt").read_text() == "keep"


def test_partial_clones_have_the_same_size_after_renaming(tmp_path):
    folder = tmp_path / "Library/Caches/tinkerbox-ane/models/leftover/v1"
    folder.mkdir(parents=True)
    original, cloned = folder / "a.bin", folder / "b.bin"
    original.write_bytes(os.urandom(4 * 1024**2))
    subprocess.run(["cp", "-c", str(original), str(cloned)], check=True)
    with cloned.open("r+b") as stream:
        stream.seek(2 * 1024**2)
        stream.write(os.urandom(1024**2))
        stream.flush()
        os.fsync(stream.fileno())
    first, _ = storage(tmp_path)
    original.rename(folder / "swap.bin")
    cloned.rename(original)
    (folder / "swap.bin").rename(cloned)
    second, _ = storage(tmp_path)
    assert first["leftover"]["bytes"] == second["leftover"]["bytes"] == 5 * 1024**2
