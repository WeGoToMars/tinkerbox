"""Regressions for compiler recovery and publishing to an existing release."""

import json
from pathlib import Path
import plistlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tinkerbox_ane import coreai
from tinkerbox_ane.bundle import checkpoint_digest, read_bundle
from tinkerbox_ane.d1 import KIND, REPO, REVISION
from tinkerbox_ane.d1 import compile as compiler
from tinkerbox_ane.d1 import publish


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    asset = root / "fused.aimodel"
    asset.mkdir(parents=True)
    (asset / "main.hash").write_bytes(bytes.fromhex("11" * 32))
    spec = dict(kind="lfm2.5-vl", version=2, width=128, context=256, tiers=[256],
                hidden=8, heads=2, kv_heads=1, head_dim=4, conv_taps=2,
                conv_layers=[0], attention_layers=[1], asset=asset.name,
                source_hash="11" * 32, source={"repo": REPO, "revision": REVISION})
    (root / "checkpoint").mkdir()
    (root / "checkpoint/model.safetensors").write_bytes(b"tensors")
    spec["checkpoint_hash"] = checkpoint_digest(root / "checkpoint")
    (root / "model.json").write_text(json.dumps(spec))
    return root


def write_aot(asset, source_hash):
    asset.mkdir(parents=True)
    (asset / "metadata.json").write_text(json.dumps({"sourceHash": source_hash}))
    manifest = {"Package Version": {"1": {"Optimized Modules": {"module": {
        "Entry Function Attributes": {"c256": {"mps.fullyPlacedOnANE": True}}
    }}}}}
    (asset / "manifest.plist").write_bytes(plistlib.dumps(manifest))


@pytest.fixture
def native_tools(monkeypatch):
    def run(command, report, **kwargs):
        assert not report.with_suffix(".json").exists()
        report.with_suffix(".json").write_text('{"passed": true}')
        report.with_suffix(".log").write_text("compiled")
        source = Path(command[3])
        write_aot(Path(command[-1]), (source / "main.hash").read_bytes().hex())
        return {"passed": True}

    build = Mock(side_effect=run)
    inspect = Mock(return_value=SimpleNamespace(stdout=json.dumps({
        "supportedArchitectures": ["h17s"],
        "minSupportedOSVersions": ["macOS 27.0"],
        "summary": {"functions": [{"name": "c256"}]}
    })))
    monkeypatch.setattr(coreai, "run", build)
    monkeypatch.setattr(coreai.subprocess, "run", inspect)
    return build, inspect


def test_force_rebuilds_and_reuses_directory(source, tmp_path, native_tools):
    out = tmp_path / "out"
    compiler.compile_model(source, out, architecture="h17s")
    marker = out / "fused.h17s.aimodelc" / "old"
    marker.touch()
    compiler.compile_model(source, out, architecture="h17s")
    assert marker.exists()
    compiler.compile_model(source, out, architecture="h17s", force=True)
    assert not marker.exists()
    assert read_bundle(out, KIND)["architecture"] == "h17s"
    assert (out / "checkpoint/model.safetensors").read_bytes() == b"tensors"


def test_failed_compilation_can_be_retried(source, tmp_path, native_tools):
    build, _ = native_tools
    success = build.side_effect

    def failed(command, report, **kwargs):
        report.with_suffix(".json").write_text('{"passed": false}')
        report.with_suffix(".log").write_text("failed")
        return {"passed": False, "abort": "deadline exceeded"}

    build.side_effect = failed
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        compiler.compile_model(source, out, architecture="h17s")
    assert not json.loads((tmp_path / ".out.build/compile.json").read_text())["passed"]
    build.side_effect = success
    compiler.compile_model(source, out, architecture="h17s")
    assert read_bundle(out, KIND)["architecture"] == "h17s"
    assert json.loads((out / "compile.json").read_text())["passed"]


def test_failed_rebuild_preserves_previous_bundle(source, tmp_path, native_tools):
    _, inspect = native_tools
    out = tmp_path / "out"
    compiler.compile_model(source, out, architecture="h17s")
    before = (out / "model.json").read_bytes()
    marker = out / "fused.h17s.aimodelc" / "previous"
    marker.touch()
    info = json.loads(inspect.return_value.stdout)
    info["summary"]["functions"] = []
    inspect.return_value.stdout = json.dumps(info)
    with pytest.raises(ValueError, match="function set differs"):
        compiler.compile_model(source, out, architecture="h17s", force=True)
    assert marker.exists()
    assert (out / "model.json").read_bytes() == before
    read_bundle(out, KIND)


@pytest.fixture
def release(source, tmp_path, native_tools):
    staging = tmp_path / "release"
    publish.build_architectures(["h17s"], source, staging)
    metadata = tmp_path / "metadata"
    (metadata / "LICENSES").mkdir(parents=True)
    (metadata / "LICENSES/LFM-Open-License-v1.0.txt").write_text("license")
    for name in ("NOTICE", "README.md"):
        (metadata / name).write_text(name)
    return staging, metadata


def test_existing_tag_fails_before_upload(release, monkeypatch):
    staging, metadata = release
    hub = Mock()
    hub.list_repo_refs.return_value = SimpleNamespace(tags=[SimpleNamespace(name="v1")])
    monkeypatch.setattr("huggingface_hub.HfApi", Mock(return_value=hub))
    with pytest.raises(ValueError, match="already exists"):
        publish.upload_to_hub("user/repo", staging, "v1", metadata_dir=metadata)
    hub.upload_folder.assert_not_called()
    hub.create_tag.assert_not_called()


def test_upload_to_hub_assembles_release_and_synchronizes_patterns(release, monkeypatch):
    staging, metadata = release
    hub = Mock()
    hub.list_repo_refs.return_value = SimpleNamespace(tags=[])
    hub.upload_folder.return_value = SimpleNamespace(oid="commit_sha_123")
    monkeypatch.setattr("huggingface_hub.HfApi", Mock(return_value=hub))

    commit = publish.upload_to_hub("user/d1-3B-CoreAI-ANE", staging, tag="v1.0.1", metadata_dir=metadata)

    assert commit == "commit_sha_123"
    assert (staging / "LICENSE").read_text() == "license"
    assert (staging / "NOTICE").read_text() == "NOTICE"
    assert (staging / "README.md").read_text() == "README.md"
    assert json.loads((staging / "config.json").read_text()) == {
        "version": 2, "source_hash": "11" * 32,
        "checkpoint_hash": checkpoint_digest(staging / "macos27/h17s/checkpoint"),
        "bundles": {"h17s": "macos27/h17s/model.json"},
    }

    hub.upload_folder.assert_called_once()
    kwargs = hub.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "user/d1-3B-CoreAI-ANE"
    assert kwargs["folder_path"] == staging
    assert "LICENSE" in kwargs["allow_patterns"]
    assert "NOTICE" in kwargs["allow_patterns"]
    assert "README.md" in kwargs["allow_patterns"]
    assert "config.json" in kwargs["allow_patterns"]
    assert "macos27/h17s/model.json" in kwargs["allow_patterns"]
    assert "macos27/h17s/fused.h17s.aimodelc/**" in kwargs["allow_patterns"]
    assert "macos27/h17s/checkpoint/**" in kwargs["allow_patterns"]
    assert kwargs["delete_patterns"] == ["macos27/**"]
    assert "**/__pycache__/**" in kwargs["ignore_patterns"]

    hub.create_tag.assert_called_once_with(
        "user/d1-3B-CoreAI-ANE", tag="v1.0.1", revision="commit_sha_123"
    )


def test_failed_later_stage_preserves_the_complete_previous_bundle(source, tmp_path, native_tools):
    from tinkerbox_ane.bundle import compile_bundle

    spec = json.loads((source / "model.json").read_text())
    second = source / "second.aimodel"
    second.mkdir()
    (second / "main.hash").write_bytes(bytes.fromhex("22" * 32))
    spec["assets"] = [{"asset": spec.pop("asset"), "source_hash": spec.pop("source_hash"), "functions": ["c256"]},
                      {"asset": second.name, "source_hash": "22" * 32, "functions": ["c256"]}]
    (source / "model.json").write_text(json.dumps(spec))
    out = tmp_path / "out"
    compile_bundle(source, out, KIND, architecture="h17s")
    before = (out / "model.json").read_bytes()
    for asset, identity in zip(spec["assets"], ["33" * 32, "44" * 32]):
        asset["source_hash"] = identity
        (source / asset["asset"] / "main.hash").write_bytes(bytes.fromhex(identity))
    (source / "checkpoint/model.safetensors").write_bytes(b"changed tensors")
    spec["checkpoint_hash"] = checkpoint_digest(source / "checkpoint")
    (source / "model.json").write_text(json.dumps(spec))
    build, _ = native_tools
    success = build.side_effect

    def fail_second(command, report, **kwargs):
        if Path(command[3]).name == second.name:
            report.with_suffix(".log").write_text("failed")
            report.with_suffix(".json").write_text('{"passed": false}')
            return {"passed": False, "abort": "deadline exceeded"}
        return success(command, report, **kwargs)

    build.side_effect = fail_second
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        compile_bundle(source, out, KIND, architecture="h17s", force=True)
    assert (out / "model.json").read_bytes() == before
    assert (out / "checkpoint/model.safetensors").read_bytes() == b"tensors"
    read_bundle(out, KIND)
    build.side_effect = success
    compile_bundle(source, out, KIND, architecture="h17s", force=True)
    assert read_bundle(out, KIND)["checkpoint_hash"] == spec["checkpoint_hash"]
    assert (out / "checkpoint/model.safetensors").read_bytes() == b"changed tensors"


def test_publish_rejects_different_vision_exports(release):
    import shutil

    staging, metadata = release
    first = staging / "macos27/h17s"
    second = staging / "macos27/h16g"
    shutil.copytree(first, second)
    (second / "fused.h17s.aimodelc").rename(second / "fused.h16g.aimodelc")
    for directory, architecture, identity in [(first, "h17s", "aa" * 32), (second, "h16g", "bb" * 32)]:
        spec = json.loads((directory / "model.json").read_text())
        vision = directory / f"vision.{architecture}.aimodelc"
        write_aot(vision, identity)
        spec.update(architecture=architecture, asset=f"fused.{architecture}.aimodelc",
                    vision_asset=vision.name, vision_source_hash=identity)
        (directory / "model.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="same source identity"):
        publish.upload_to_hub("user/repo", staging, metadata_dir=metadata)


def test_publish_rejects_the_wrong_d3_variant(source):
    from tinkerbox_ane.d3 import KIND as D3_KIND, VARIANTS
    from tinkerbox_ane.d3.publish import main

    spec = json.loads((source / "model.json").read_text())
    repo, revision, _ = VARIANTS["d3-nano"]
    spec.update(kind=D3_KIND, source={"repo": repo, "revision": revision})
    (source / "model.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="selected model"):
        main(["d3-mini", "--source", str(source), "--no-upload"])


def test_publish_rejects_changed_cpu_checkpoint(release):
    staging, metadata = release
    (staging / "macos27/h17s/checkpoint/model.safetensors").write_bytes(b"other tensors")
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        publish.upload_to_hub("user/repo", staging, metadata_dir=metadata)
