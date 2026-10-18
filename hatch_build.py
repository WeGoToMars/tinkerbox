"""Build the native host into macOS ARM64 wheels and editable installs."""

from pathlib import Path
import platform
import subprocess
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("building tinkerbox-ane requires Apple Silicon and Xcode 27")
        package = Path(self.root) / "src/tinkerbox_ane"
        binary = package / "tinkerbox-ane-host"
        with tempfile.TemporaryDirectory(prefix=".host-", dir=package) as temporary:
            staging = Path(temporary) / "staging.o"
            host = Path(temporary) / "tinkerbox-ane-host"
            subprocess.run([
                "xcrun", "--sdk", "macosx", "clang", "-O2", "-fobjc-arc",
                "-target", "arm64-apple-macos27.0", "-c", str(package / "cow-staging.m"),
                "-o", str(staging),
            ], check=True)
            subprocess.run([
                "xcrun", "--sdk", "macosx", "swiftc", "-O", "-parse-as-library",
                "-target", "arm64-apple-macos27.0", str(package / "host.swift"),
                str(staging), "-o", str(host),
            ], check=True)
            host.replace(binary)
        build_data["artifacts"].append("src/tinkerbox_ane/tinkerbox-ane-host")
        build_data["pure_python"] = False
        # PyPI does not accept macosx_27 tags yet; the host binary itself still requires macOS 27.
        build_data["tag"] = "py3-none-macosx_26_0_arm64"
