"""Prepare a release locally: preflight checks, tests on the Neural Engine, and an annotated tag.

Pushing the tag starts the Release workflow, which builds the wheel and the signed app, drafts the
GitHub release, and publishes to PyPI. `--verify TAG` runs only the preflight checks, as CI does.
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


def check_catalog(version: str) -> None:
    """Every catalog model this version ships must have its bundles tagged on Hugging Face."""
    from huggingface_hub import HfApi

    catalog = json.loads((ROOT / "src/tinkerbox_ane/models.json").read_text())["models"]
    for model in catalog:
        if not released(version, model["min_version"]):
            continue
        tags = {ref.name for ref in HfApi().list_repo_refs(model["repo"]).tags}
        if model["revision"] not in tags:
            fail(f"{model['repo']} has no tag {model['revision']} for {model['id']}; publish its bundles first")
        print(f"✓ {model['id']}: {model['repo']}@{model['revision']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--verify", metavar="TAG", help="only check that TAG matches the version and the catalog")
    args = parser.parse_args(argv)

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    tag = f"v{version}"
    if args.verify:
        if args.verify != tag:
            fail(f"tag {args.verify} does not match version {version}")
        check_catalog(version)
        return 0

    branch = output("git", "branch", "--show-current")
    print(f"=== Preparing {tag} from {branch} ===")
    if output("git", "status", "--porcelain"):
        fail("commit or stash changes before releasing")
    if output("git", "tag", "--list", tag):
        fail(f"tag {tag} already exists")
    check_catalog(version)
    if not args.skip_tests:
        run("uv", "run", "pytest", "-q")
    run("git", "tag", "-a", tag, "-m", f"Release {tag}")

    print(f"""
Prepared {tag}. After review, push it; the Release workflow builds the wheel and the signed app,
publishes to PyPI through trusted publishing, and drafts the GitHub release:

  git push origin {branch} {tag}
""")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
