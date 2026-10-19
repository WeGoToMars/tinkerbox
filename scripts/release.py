"""Prepare a release locally: preflight checks, tests, wheel, app, and an annotated tag.

Nothing is pushed or uploaded; the script prints the publishing commands to run after review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def run(*command: str) -> None:
    print(f"==> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def output(*command: str) -> str:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def released(version: str, minimum: str) -> bool:
    return tuple(map(int, minimum.split("."))) <= tuple(map(int, version.split(".")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-app", action="store_true", help="Skip the SwiftUI app ZIP")
    parser.add_argument("--sign", metavar="IDENTITY", help="code signing identity for the app (required unless --no-app)")
    args = parser.parse_args(argv)

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    tag = f"v{version}"
    branch = output("git", "branch", "--show-current")
    print(f"=== Preparing {tag} from {branch} ===")

    if not args.no_app and not args.sign:
        fail("pass --sign IDENTITY; Shortcuts cannot run actions from an ad-hoc signed app")
    if output("git", "status", "--porcelain"):
        fail("commit or stash changes before releasing")
    if output("git", "tag", "--list", tag):
        fail(f"tag {tag} already exists")
    from huggingface_hub import HfApi

    catalog = json.loads((ROOT / "src/tinkerbox_ane/models.json").read_text())["models"]
    for model in catalog:
        if not released(version, model["min_version"]):
            continue
        tags = {ref.name for ref in HfApi().list_repo_refs(model["repo"]).tags}
        if model["revision"] not in tags:
            fail(f"{model['repo']} has no tag {model['revision']} for {model['id']}; publish its bundles first")
        print(f"✓ {model['id']}: {model['repo']}@{model['revision']}")

    if not args.skip_tests:
        run("uv", "run", "pytest", "-q")
    run("uv", "build", "--clear")
    assets = list((ROOT / "dist").glob(f"tinkerbox_ane-{version}-*.whl"))
    if len(assets) != 1:
        fail(f"expected one {version} wheel in dist/")
    if not args.no_app:
        run("uv", "run", "--no-project", "--python", "3.12", "python", "examples/swiftui/run.py", "--build-only",
            "--sign", args.sign)
        assets.append(ROOT / f"artifacts/swiftui/Tinkerbox-ANE-{version}-macos-arm64.zip")
    run("git", "tag", "-a", tag, "-m", f"Release {tag}")

    paths = " ".join(str(path.relative_to(ROOT)) for path in assets)
    print(f"""
Prepared {tag}. After review, publish with:

  git push origin {branch} {tag}
  uv publish {assets[0].relative_to(ROOT)}
  gh release create {tag} {paths} --verify-tag --title {tag} --generate-notes
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
