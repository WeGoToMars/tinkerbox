"""High-signal process supervision and watchdog abort tests."""

import json
import sys

import pytest

from tinkerbox_ane import guard


def test_watchdog_aborts_and_terminates_process_on_failure_log(tmp_path):
    """Verify watchdog detects compilation failure in logs, kills the process, and records abort."""
    if sys.platform != "darwin":
        pytest.skip("guard uses macOS ioreg and ANE process accounting")

    report_path = tmp_path / "compile"
    # Command prints the abort token and sleeps to simulate a hung or rogue compiler
    command = [
        sys.executable,
        "-c",
        "import time; print('CompilationFailure', flush=True); time.sleep(10)",
    ]

    report = guard.run(command, report_path, timeout=5)

    assert report["passed"] is False
    assert report["abort"] == "required ANE compilation failed; refusing GPU fallback"
    assert report_path.with_suffix(".json").is_file()
    assert report_path.with_suffix(".log").is_file()

    saved = json.loads(report_path.with_suffix(".json").read_text())
    assert saved["passed"] is False
    assert saved["abort"] == report["abort"]
