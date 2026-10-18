"""Regressions for compiler recovery and publishing to an existing release."""

import json
from pathlib import Path
import plistlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tinkerbox_ane import coreai
from tinkerbox_ane.bundle import read_bundle
from tinkerbox_ane.d1 import KIND
from tinkerbox_ane.d1 import compile as compiler
from tinkerbox_ane.d1 import publish


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    asset = root / "fused.aimodel"
    asset.mkdir(parents=True)
    (asset / "main.hash").write_bytes(bytes.fromhex("11" * 32))
    spec = dict(kind="d1ane-fused", version=2, width=128, context=256, tiers=[256],
                hidden=8, heads=2, kv_heads=1, head_dim=4, conv_taps=2,
                conv_layers=[0], attention_layers=[1], asset=asset.name,
                source_hash="11" * 32)
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
    assert not json.loads((out / "compile.json").read_text())["passed"]
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

    hub.upload_folder.assert_called_once()
    kwargs = hub.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "user/d1-3B-CoreAI-ANE"
    assert kwargs["folder_path"] == staging
    assert "LICENSE" in kwargs["allow_patterns"]
    assert "NOTICE" in kwargs["allow_patterns"]
    assert "README.md" in kwargs["allow_patterns"]
    assert "macos27/h17s/model.json" in kwargs["allow_patterns"]
    assert "macos27/h17s/fused.h17s.aimodelc/**" in kwargs["allow_patterns"]
    assert kwargs["delete_patterns"] == ["macos27/**"]

    hub.create_tag.assert_called_once_with(
        "user/d1-3B-CoreAI-ANE", tag="v1.0.1", revision="commit_sha_123"
    )
