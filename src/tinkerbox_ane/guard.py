"""Watchdog to enforce execution deadlines, SSD write budgets, and disk space limits."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def disk_written():
    result = subprocess.run(["ioreg", "-r", "-c", "IOBlockStorageDriver", "-w0", "-l"],
                            capture_output=True, text=True, check=True, timeout=5)
    values = re.findall(r'"Bytes \(Write\)"=(\d+)', result.stdout)
    if not values:
        raise RuntimeError("SSD write counter unavailable")
    return sum(map(int, values))


def workers():
    result = subprocess.run(["pgrep", "-fl", "ANE.*CompilerService.xpc/Contents/MacOS/"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode not in (0, 1):
        raise RuntimeError("cannot inventory ANE compiler workers")
    return {int(line.split()[0]): line for line in result.stdout.splitlines()
            if "/ANECompilerService.xpc/Contents/MacOS/" in line
            or "/ANELargeModelCompilerService.xpc/Contents/MacOS/" in line}


def stop(process, initial, privileged):
    kill = (["sudo", "-n"] if privileged else []) + ["/bin/kill"]
    try:
        owned = set(workers()) - set(initial)
        if owned:
            subprocess.run(kill + ["-KILL", *map(str, sorted(owned))], capture_output=True, timeout=5)
    finally:
        # Always reap the client, even if worker discovery or cancellation fails.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)
    return {pid: name for pid, name in workers().items() if pid not in initial}


def run(command, output, timeout=1500):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.with_suffix(".json").exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    initial = workers()
    if any("/ANELargeModelCompilerService.xpc/" in line for line in initial.values()):
        raise RuntimeError(f"compiler workers already active: {initial}")
    privileged = subprocess.run(["sudo", "-n", "/usr/bin/true"], capture_output=True).returncode == 0
    if not privileged:
        print("warning: sudo unavailable; root-owned compiler workers may survive cancellation.",
              file=sys.stderr, flush=True)
    before, started = disk_written(), time.monotonic()
    report = {"command": command, "abort": None}
    written, refresh_at, progress_at, exited_at = 0, 30, 30, None
    tail = ""
    with (
        output.with_suffix(".log").open("w") as sink,
        output.with_suffix(".log").open("r", errors="replace") as diagnostics,
    ):
        process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                returncode = process.poll()
                elapsed = time.monotonic() - started
                written = disk_written() - before
                free = shutil.disk_usage(output.parent).free
                text = tail + diagnostics.read()
                tail = text[-64:]
                if "ANECCompileOffline() failed" in text or "CompilationFailure" in text:
                    report["abort"] = "required ANE compilation failed; refusing GPU fallback"
                elif written > 100_000_000_000:
                    report["abort"] = "SSD write budget exceeded"
                elif free < 15 * 1024**3:
                    report["abort"] = "free disk below 15 GiB"
                elif elapsed > timeout:
                    report["abort"] = "deadline exceeded"
                if report["abort"]:
                    break
                if elapsed >= progress_at:
                    print(f"    {output.name}: still running after {elapsed:.0f}s; "
                          f"system writes {written / 1_000_000_000:.1f} GB; "
                          f"free disk {free / 1024**3:.1f} GiB", flush=True)
                    progress_at = elapsed + 30
                if privileged and elapsed >= refresh_at:
                    privileged = subprocess.run(["sudo", "-n", "-v"], capture_output=True).returncode == 0
                    refresh_at = elapsed + 30
                if returncode is not None:
                    if not set(workers()) - set(initial):
                        break
                    if exited_at is None:
                        exited_at = elapsed
                    if elapsed - exited_at >= 15:
                        report["abort"] = "compiler workers still running after client exit"
                        break
                time.sleep(.2)
        except BaseException as error:
            report["abort"] = f"watchdog interrupted: {type(error).__name__}: {error}"
            raise
        finally:
            remaining = stop(process, initial, privileged)
            if remaining:
                report["abort"] = report["abort"] or f"compiler workers did not stop: {sorted(remaining)}"
                print(f"warning: ANE compiler workers still running: {sorted(remaining)}", file=sys.stderr)
            report.update(returncode=process.returncode, seconds=time.monotonic() - started,
                          system_written_bytes=written,
                          passed=report["abort"] is None and process.returncode == 0)
            output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    result = run(command, args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
