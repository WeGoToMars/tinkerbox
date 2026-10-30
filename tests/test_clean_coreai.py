"""Exercise the standalone cleanup CLI without touching the user's real caches."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "darwin" or os.getuid() == 0,
                                reason="cleanup is for an unprivileged macOS user")


@pytest.fixture
def cleanup(tmp_path):
    root = tmp_path.resolve()
    script = root / "repo/scripts/clean_coreai.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "scripts/clean_coreai.py", script)
    home, temporary = root / "home", root / "temp"
    home.mkdir()
    temporary.mkdir()
    environment = dict(os.environ, HOME=str(home), TMPDIR=str(temporary))

    def run(*arguments):
        return subprocess.run([sys.executable, str(script), *arguments], env=environment,
                              capture_output=True, text=True)

    return root, run


def test_preview_includes_persistent_caches_and_only_opted_in_bundles(cleanup):
    root, run = cleanup
    files = [
        "home/Library/Application Support/coreai-cache/old-os/tinkerbox-ane-host/weights",
        "home/Library/Caches/coreai-cache/tmp/entry",
        "repo/artifacts/.host-abandoned/clones/resources.bin",
        "repo/artifacts/build/model/weights",
        "home/Library/Caches/tinkerbox-ane/models/model/weights",
        "repo/artifacts/autoresearch/campaign/report.json",
        "home/.cache/huggingface/checkpoint",
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep me")
    preview = run()
    assert preview.returncode == 0, preview.stderr
    assert "persistent specialization" in preview.stdout
    assert "Core AI locks/temp" in preview.stdout
    assert "abandoned host/compiler staging" in preview.stdout
    assert "build outputs:" not in preview.stdout
    assert "downloaded/installed models:" not in preview.stdout
    expanded = run("--include-build", "--include-models")
    assert expanded.returncode == 0, expanded.stderr
    assert "build outputs:" in expanded.stdout
    assert "downloaded/installed models:" in expanded.stdout
    assert all((root / name).read_bytes() == b"keep me" for name in files)


def test_never_deletes_through_a_symlinked_cache(cleanup):
    root, run = cleanup
    outside = root / "unrelated"
    outside.mkdir()
    sentinel = outside / "valuable"
    sentinel.write_text("keep")
    cache = root / "home/Library/Application Support/coreai-cache"
    cache.parent.mkdir(parents=True)
    cache.symlink_to(outside, target_is_directory=True)
    result = run("--apply")
    assert result.returncode != 0
    assert sentinel.read_text() == "keep"


def test_apply_refuses_a_live_coreai_host_before_deleting(cleanup):
    root, run = cleanup
    cache = root / "home/Library/Application Support/coreai-cache"
    cache.mkdir(parents=True)
    sentinel = cache / "weights"
    sentinel.write_text("keep")
    host = root / "tinkerbox-ane-host"
    subprocess.run(["cc", "-x", "c", "-", "-o", str(host)],
                   input="#include <unistd.h>\nint main(void) { sleep(60); return 0; }\n",
                   text=True, capture_output=True, check=True)
    process = subprocess.Popen([str(host), "60"])
    try:
        result = run("--apply")
        assert result.returncode != 0
        assert "nothing deleted" in result.stderr
        assert f"PID {process.pid}:" in result.stdout
        assert sentinel.read_text() == "keep"
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)
