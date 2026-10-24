"""Compile and publish d3 model bundles to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from tinkerbox_ane import publish
from tinkerbox_ane.bundle import read_bundle
from tinkerbox_ane.d3 import KIND, VARIANTS
from tinkerbox_ane.d3.compile import compile_model
from tinkerbox_ane.models import entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(VARIANTS))
    parser.add_argument("--source", type=Path, required=True, help="Export from tinkerbox_ane.d3.export")
    parser.add_argument("--staging", type=Path, default=None,
                        help="Release directory (defaults to artifacts/huggingface/<artifact repo name>)")
    parser.add_argument("--repo", default=None, help="Artifact repository (defaults to the catalog's)")
    parser.add_argument("--tag", default=None, help="New release tag to create (defaults to the catalog revision)")
    parser.add_argument("--arch", nargs="+")
    parser.add_argument("--force", action="store_true", help="Recompile existing bundles")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Directory with release metadata (defaults to staging, then repo root)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-upload", action="store_true", help="Compile only")
    modes.add_argument("--upload-only", action="store_true", help="Publish existing bundles")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    item = entry(args.model)
    args.arch = args.arch or item['architectures']
    repo, tag = args.repo or item["repo"], args.tag or item["revision"]
    staging = args.staging or Path("artifacts/huggingface") / repo.split("/")[-1]
    upstream, revision, _ = VARIANTS[args.model]
    source = {"repo": upstream, "revision": revision}

    architectures = args.arch
    if not args.upload_only:
        if read_bundle(args.source, KIND)["source"] != source:
            raise ValueError("export source does not match the selected model")
        bundles = publish.build_architectures(
            partial(compile_model, allow_unsupported=args.allow_unsupported), args.source,
            staging / "macos27", architectures, force=args.force, continue_on_error=args.continue_on_error)
        architectures = list(bundles)
    if not args.no_upload:
        commit = publish.upload_release(repo, staging, KIND, tag, license_file="LICENSES/Apache-2.0.txt",
                                        expected_source=source,
                                        architectures=architectures, metadata_dir=args.metadata)
        print(f"Published {repo} at {commit}")
    return int(set(architectures) != set(args.arch))


if __name__ == "__main__":
    raise SystemExit(main())
