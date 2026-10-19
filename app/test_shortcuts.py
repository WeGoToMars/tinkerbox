"""Run the real App Intent perform methods against the built app's runtime (no mocks)."""

from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile


root = Path(__file__).resolve().parents[1]
source = root / "app"
app = root / "artifacts/swiftui/Tinkerbox.app"
if not app.is_dir():
    raise SystemExit("Build the app first: python3 app/run.py --build-only")

with tempfile.TemporaryDirectory(prefix="shortcuts-", dir=root / "artifacts/swiftui") as temporary:
    bundle = Path(temporary) / "ShortcutsSmoke.app/Contents"
    binaries = bundle / "MacOS"
    binaries.mkdir(parents=True)
    shutil.copytree(app / "Contents/Resources", bundle / "Resources")
    shutil.copyfile(app / "Contents/MacOS/uv", binaries / "uv")
    (binaries / "uv").chmod(0o755)
    version = plistlib.loads((app / "Contents/Info.plist").read_bytes())["CFBundleShortVersionString"]
    (bundle / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": "dev.tinkerbox.ane.shortcuts-smoke",
        "CFBundleExecutable": "ShortcutsSmoke",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
    }))
    executable = binaries / "ShortcutsSmoke"
    subprocess.run([
        "xcrun", "swiftc", "-O", "-parse-as-library", "-target", "arm64-apple-macos27.0",
        str(source / "Catalog.swift"), str(source / "Runtime.swift"), str(source / "Shortcuts.swift"),
        str(root / "tests/swiftui/ShortcutsSmoke.swift"), "-o", str(executable),
    ], check=True)
    subprocess.run([str(executable)], check=True, timeout=600)
