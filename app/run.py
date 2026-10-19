"""Build the portable SwiftUI app and release ZIP, then launch it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-only", action="store_true")
parser.add_argument("--wheel", type=Path,
                    help="bundle this tinkerbox-ane wheel instead of installing the release from PyPI")
parser.add_argument("--sign", default="-", metavar="IDENTITY",
                    help="code signing identity; Shortcuts rejects ad-hoc signed apps (default: ad-hoc)")
args = parser.parse_args()

source = Path(__file__).resolve().parent
root = source.parent
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
out = root / "artifacts/swiftui"
out.mkdir(parents=True, exist_ok=True)
app = out / "Tinkerbox.app"
release = out / f"Tinkerbox-ANE-{version}-macos-arm64.zip"

uv_version = "0.11.28"
archive = out / f"uv-{uv_version}-aarch64-apple-darwin.tar.gz"
if not archive.exists():
    download = archive.with_suffix(".part")
    urllib.request.urlretrieve(
        f"https://github.com/astral-sh/uv/releases/download/{uv_version}/uv-aarch64-apple-darwin.tar.gz",
        download,
    )
    download.rename(archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != "33540eb7c883ab857eff79bd5ac2aa31fe27b595abecb4a9c003a2c998447232":
    raise RuntimeError(f"uv archive checksum mismatch; remove {archive} and rebuild")

with tempfile.TemporaryDirectory(prefix=".app-", dir=out) as temporary:
    built = Path(temporary) / app.name
    contents = built / "Contents"
    binaries = contents / "MacOS"
    resources = contents / "Resources"
    binaries.mkdir(parents=True)
    resources.mkdir()
    uv = binaries / "uv"
    with tarfile.open(archive) as bundle:
        uv.write_bytes(bundle.extractfile("uv-aarch64-apple-darwin/uv").read())
    uv.chmod(0o755)
    shutil.copyfile(source / "LICENSE-uv.txt", resources / "LICENSE-uv.txt")
    shutil.copyfile(source / "worker.py", resources / "worker.py")
    # The offline model catalog: exactly the models this package version can run.
    shutil.copyfile(root / "src/tinkerbox_ane/models.json", resources / "models.json")
    icon_plist = Path(temporary) / "icon.plist"
    subprocess.run([
        "xcrun", "actool", str(source / "Tinkerbox.icon"), "--compile", str(resources),
        "--platform", "macosx", "--minimum-deployment-target", "27.0",
        "--app-icon", "Tinkerbox", "--output-partial-info-plist", str(icon_plist),
    ], check=True, stdout=subprocess.DEVNULL)
    requirements = subprocess.check_output([
        str(uv), "export", "--project", str(root), "--locked", "--no-dev",
        "--no-emit-project", "--no-hashes", "--no-header", "--no-annotate",
    ], text=True)
    package = f"tinkerbox-ane=={version}"
    if args.wheel:
        # Development builds reference the wheel inside the final app location.
        shutil.copyfile(args.wheel, resources / args.wheel.name)
        package = f"tinkerbox-ane @ {(app / 'Contents/Resources' / args.wheel.name).as_uri()}"
    (resources / "requirements.txt").write_text(f"{package}\n" + requirements)
    swift_sources = [source / name for name in ("Demo.swift", "Catalog.swift", "Runtime.swift", "Shortcuts.swift")]
    toolchain = Path(subprocess.check_output(["xcrun", "--find", "swiftc"], text=True).strip()).parents[1]
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    const_values = Path(temporary) / "TinkerboxDemo.swiftconstvalues"
    protocols = Path(temporary) / "const-protocols.json"
    protocols.write_text(json.dumps(json.loads(
        (toolchain / "share/swift/SwiftConstantValues/AppIntents.json").read_text()
    )["constValueProtocols"]))
    subprocess.run([
        "xcrun", "--sdk", "macosx", "swiftc", "-O", "-parse-as-library",
        "-whole-module-optimization", "-module-name", "TinkerboxDemo",
        "-target", "arm64-apple-macos27.0", *map(str, swift_sources),
        "-emit-const-values-path", str(const_values),
        "-const-gather-protocols-list", str(protocols),
        "-o", str(binaries / "TinkerboxDemo"),
    ], check=True)
    # swiftc alone does not package the metadata Shortcuts uses to discover actions.
    source_list = Path(temporary) / "sources.txt"
    source_list.write_text("\n".join(map(str, swift_sources)) + "\n")
    const_list = Path(temporary) / "const-values.txt"
    const_list.write_text(str(const_values) + "\n")
    xcode_version = subprocess.check_output(["xcodebuild", "-version"], text=True).split()[-1]
    subprocess.run([
        "xcrun", "appintentsmetadataprocessor", "--module-name", "TinkerboxDemo",
        "--toolchain-dir", str(toolchain), "--sdk-root", sdk,
        "--xcode-version", xcode_version, "--platform-family", "macOS",
        "--deployment-target", "27.0", "--target-triple", "arm64-apple-macos27.0",
        "--source-file-list", str(source_list), "--swift-const-vals-list", str(const_list),
        "--output", str(resources),
    ], check=True)
    info = {
        "CFBundleIdentifier": "dev.tinkerbox.ane.demo",
        "CFBundleName": "Tinkerbox",
        "CFBundleExecutable": "TinkerboxDemo",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "27.0",
        "NSHighResolutionCapable": True,
    }
    info.update(plistlib.loads(icon_plist.read_bytes()))
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    for target in (uv, built):
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", args.sign, str(target)], check=True)
    if app.exists():
        shutil.rmtree(app)
    built.rename(app)

subprocess.run(["/usr/bin/ditto", "-c", "-k", "--keepParent", str(app), str(release)], check=True)
print(f"App: {app}\nRelease: {release}", flush=True)
if not args.build_only:
    os.execv(str(app / "Contents/MacOS/TinkerboxDemo"), [str(app / "Contents/MacOS/TinkerboxDemo")])
