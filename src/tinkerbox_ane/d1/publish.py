"""Compile and publish d1 model bundles to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from functools import partial
from pathlib import Path

from tinkerbox_ane import publish
from tinkerbox_ane.d1 import ARTIFACT_REPO, ARTIFACT_REVISION, KIND, REPO, REVISION, SOURCE
from tinkerbox_ane.d1.compile import compile_model
from tinkerbox_ane.models import entry

DEFAULT_ARCHITECTURES = tuple(entry('d1-3b')['architectures'])
DEFAULT_STAGING = Path("artifacts/huggingface/d1-3B-CoreAI-ANE")


def build_architectures(
    architectures: Iterable[str] = DEFAULT_ARCHITECTURES,
    source: Path = SOURCE,
    staging: Path = DEFAULT_STAGING,
    *,
    force: bool = False,
    continue_on_error: bool = False,
    allow_unsupported: bool = False,
) -> dict[str, Path]:
    return publish.build_architectures(
        partial(compile_model, allow_unsupported=allow_unsupported),
        source, Path(staging) / "macos27", architectures,
        force=force, continue_on_error=continue_on_error,
    )


def upload_to_hub(
    repo: str,
    staging: Path,
    tag: str | None = None,
    *,
    architectures: Iterable[str] | None = None,
    metadata_dir: Path | None = None,
    token: str | None = None,
) -> str:
    """Verify and publish a complete macos27 release for the selected architectures."""
    return publish.upload_release(repo, staging, KIND, tag, license_file="LICENSES/LFM-Open-License-v1.0.txt",
                                  expected_source={"repo": REPO, "revision": REVISION},
                                  architectures=architectures, metadata_dir=metadata_dir, token=token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--repo", default=ARTIFACT_REPO)
    parser.add_argument("--tag", default=ARTIFACT_REVISION, help="New release tag to create")
    parser.add_argument("--arch", nargs="+", default=list(DEFAULT_ARCHITECTURES))
    parser.add_argument("--force", action="store_true", help="Recompile existing bundles")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Directory with release metadata (defaults to staging, then repo root)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-upload", action="store_true", help="Compile only")
    modes.add_argument("--upload-only", action="store_true", help="Publish existing bundles")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    architectures = args.arch
    if not args.upload_only:
        bundles = build_architectures(architectures, args.source, args.staging,
                                      force=args.force, continue_on_error=args.continue_on_error,
                                      allow_unsupported=args.allow_unsupported)
        architectures = list(bundles)
    if not args.no_upload:
        commit = upload_to_hub(args.repo, args.staging, args.tag,
                               architectures=architectures, metadata_dir=args.metadata)
        print(f"Published {args.repo} at {commit}")
    return int(set(architectures) != set(args.arch))


if __name__ == "__main__":
    raise SystemExit(main())
