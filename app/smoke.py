"""Run the Swift smokes in tests/swiftui against the built app's runtime and bundled worker (no mocks)."""

import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
app = root / "artifacts/swiftui/Tinkerbox.app"
smokes = {
    "shortcuts": "ShortcutsSmoke",
    "completion": "CompletionSmoke",
    "tokens": "TokenHighlightSmoke",
    "speech": "SpeechSmoke",
    "voxcpm2": "VoxCPM2SpeechSmoke",
    "speech-shortcuts": "SpeechShortcutsSmoke",
    "speech-modes": "SpeechModeSmoke",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "names", nargs="*", metavar="name", help=f"smokes to run, in order (default: all of {', '.join(smokes)})"
)
args = parser.parse_args()
if unknown := set(args.names) - set(smokes):
    parser.error(f"unknown smokes: {', '.join(sorted(unknown))}")
if not app.is_dir():
    raise SystemExit("Build the app first: python3 app/run.py --build-only")
version = plistlib.loads((app / "Contents/Info.plist").read_bytes())["CFBundleShortVersionString"]

for name in args.names or smokes:
    smoke = smokes[name]
    with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=app.parent) as temporary:
        contents = Path(temporary) / f"{smoke}.app/Contents"
        binaries = contents / "MacOS"
        binaries.mkdir(parents=True)
        shutil.copytree(app / "Contents/Resources", contents / "Resources")
        shutil.copyfile(app / "Contents/MacOS/uv", binaries / "uv")
        (binaries / "uv").chmod(0o755)
        identifier = f"dev.tinkerbox.ane.{name}-smoke"
        subprocess.run(["defaults", "delete", identifier], capture_output=True)  # start without saved selections
        (contents / "Info.plist").write_bytes(
            plistlib.dumps(
                {
                    "CFBundleIdentifier": identifier,
                    "CFBundleExecutable": smoke,
                    "CFBundlePackageType": "APPL",
                    "CFBundleShortVersionString": version,
                }
            )
        )
        executable = binaries / smoke
        subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-Onone",
                "-parse-as-library",
                "-swift-version",
                "6",
                "-target",
                "arm64-apple-macos27.0",
                *(str(root / "app" / file) for file in ["Catalog.swift", "Runtime.swift", "Shortcuts.swift"]),
                str(root / "src/tinkerbox_ane/transcription.swift"),
                str(root / "src/tinkerbox_ane/audio.swift"),
                str(root / f"tests/swiftui/{smoke}.swift"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True, timeout=1200)
