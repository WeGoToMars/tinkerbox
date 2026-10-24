"""Exercise VoxCPM2 through the built app's real Swift runtime and bundled worker."""

from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import tomllib

root = Path(__file__).resolve().parents[1]
version = tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
app = root/'artifacts/swiftui/Tinkerbox.app'
with tempfile.TemporaryDirectory(prefix='voxcpm2-speech-', dir=app.parent) as temporary:
    bundle = Path(temporary)/'VoxCPM2SpeechSmoke.app/Contents'
    binaries = bundle/'MacOS'
    binaries.mkdir(parents=True)
    shutil.copytree(app/'Contents/Resources', bundle/'Resources')
    shutil.copyfile(app/'Contents/MacOS/uv', binaries/'uv')
    (binaries/'uv').chmod(0o755)
    (bundle/'Info.plist').write_bytes(plistlib.dumps(dict(
        CFBundleIdentifier='dev.tinkerbox.ane.voxcpm2-speech-smoke',
        CFBundleExecutable='VoxCPM2SpeechSmoke', CFBundlePackageType='APPL',
        CFBundleShortVersionString=version)))
    executable = binaries/'VoxCPM2SpeechSmoke'
    subprocess.run(['xcrun', 'swiftc', '-O', '-parse-as-library', '-target', 'arm64-apple-macos27.0',
                    str(root/'app/Catalog.swift'), str(root/'app/Runtime.swift'),
                    str(root/'src/tinkerbox_ane/transcription.swift'), str(root/'src/tinkerbox_ane/audio.swift'),
                    str(root/'tests/swiftui/VoxCPM2SpeechSmoke.swift'), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=1200)
