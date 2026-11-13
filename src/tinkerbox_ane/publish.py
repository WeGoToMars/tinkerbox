"""Helpers for compiling multi-architecture models and publishing releases to Hugging Face Hub."""

from collections.abc import Iterable
import json
import logging
from pathlib import Path
import shutil
import time


def build_architectures(
    kind: str,
    source: Path,
    out: Path,
    architectures: Iterable[str],
    *,
    force: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Path]:
    """Compile an export into `out/<architecture>` for each target; returns the bundles that compiled."""
    from tinkerbox_ane.bundle import compile_bundle

    architectures = list(dict.fromkeys(architectures))
    started = time.monotonic()
    print(f"Building {len(architectures)} architectures: {', '.join(architectures)}", flush=True)
    bundles = {}
    for index, architecture in enumerate(architectures, 1):
        label = f"[{index}/{len(architectures)}] {architecture}"
        architecture_started = time.monotonic()
        print(f"{label}: preparing bundle", flush=True)
        try:
            bundles[architecture] = compile_bundle(source, Path(out) / architecture, kind, architecture, force)
        except Exception as error:
            print(f"{label}: failed after {time.monotonic() - architecture_started:.1f}s: {error}", flush=True)
            if not continue_on_error:
                raise
            logging.exception("Failed to compile %s", architecture)
        else:
            print(f"{label}: ready in {time.monotonic() - architecture_started:.1f}s -> {bundles[architecture]}",
                  flush=True)
    if not bundles:
        raise RuntimeError("no architectures were compiled")
    print(f"Ready: {len(bundles)}/{len(architectures)} architectures in {time.monotonic() - started:.1f}s; "
          f"bundles in {out}", flush=True)
    return bundles


def upload_to_hub(
    repo: str,
    staging: Path,
    tag: str | None = None,
    *,
    allow_patterns: list[str] | None = None,
    delete_patterns: list[str] | None = None,
    token: str | None = None,
) -> str:
    """Upload staged artifacts to Hugging Face Hub and optionally create a release tag.

    When delete_patterns is specified, matching remote files absent from the upload
    directory are deleted. Returns the commit SHA.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    if tag is not None and any(ref.name == tag for ref in api.list_repo_refs(repo).tags):
        raise ValueError(f"release tag {tag!r} already exists; choose a new version")
    # Loading a staged bundle compiles its checkpoint code into __pycache__; releases ship sources only.
    commit = api.upload_folder(repo_id=repo, folder_path=staging, allow_patterns=allow_patterns,
                               ignore_patterns=["**/__pycache__/**", "**/*.pyc"], delete_patterns=delete_patterns,
                               commit_message="Publish model bundles")
    if tag is not None:
        api.create_tag(repo, tag=tag, revision=commit.oid)
    return commit.oid


def prepare_release(
    staging: Path,
    kind: str,
    *,
    license_file: str,
    expected_source: dict | None = None,
    architectures: Iterable[str] | None = None,
    metadata_dir: Path | None = None,
) -> list[str]:
    """Verify and stage a complete macos27 release, returning its upload allowlist.

    Release metadata (LICENSE, NOTICE, README.md) comes from `metadata_dir`, or else from staging and
    then the repository root; `license_file` is the LICENSE source relative to that directory."""
    from tinkerbox_ane.bundle import CHECKPOINT, asset_names, bundled_checkpoint, checkpoint_digest, resources, verify_bundle

    staging = Path(staging)
    if architectures is None:
        architectures = sorted(p.name for p in (staging / "macos27").iterdir()
                               if p.is_dir() and not p.name.startswith("."))
    patterns = ["LICENSE", "LICENSES/**", "NOTICE", "README.md", "config.json"]
    identities = set()
    bundles = {}
    for architecture in architectures:
        directory = staging / "macos27" / architecture
        spec = verify_bundle(directory, kind)
        names = asset_names(spec)
        if (spec.get("architecture") != architecture
                or any(not name.endswith(f".{architecture}.aimodelc") for name in names)):
            raise ValueError(f"{directory}: bundle does not target {architecture}")
        if expected_source is not None and spec["source"] != expected_source:
            raise ValueError(f"{directory}: bundle source does not match the selected model")
        checkpoint_hash = checkpoint_digest(bundled_checkpoint(directory))
        if spec.get("checkpoint_hash") != checkpoint_hash:
            raise ValueError(f"{directory}: checkpoint identity mismatch; re-export and compile the bundle")
        identities.add((tuple(entry[identity] for entry, _, identity in resources(spec)), checkpoint_hash))
        bundles[architecture] = f"macos27/{architecture}/model.json"
        patterns += [f"macos27/{architecture}/model.json", f"macos27/{architecture}/{CHECKPOINT}/**"]
        patterns.extend(f"macos27/{architecture}/{name}/**" for name in names)
    if len(identities) != 1:
        raise ValueError("release requires bundles with the same source identity")
    hashes = tuple(entry[identity] for entry, name, identity in resources(spec) if name == "asset")
    # One shared release index is also the Hub's default download-counting file.
    index = {"version": 2, "source_hash": hashes[0]} if len(hashes) == 1 else {"version": 2, "source_hashes": list(hashes)}
    index.update(checkpoint_hash=checkpoint_hash, bundles=bundles)
    if "vision_source_hash" in spec:
        index["vision_source_hash"] = spec["vision_source_hash"]
    (staging / "config.json").write_text(json.dumps(index, indent=2) + "\n")
    fallback_root = Path(metadata_dir) if metadata_dir is not None else Path(".")
    for name in ("LICENSE", "NOTICE", "README.md"):
        target = staging / name
        if target.is_file() and metadata_dir is None:
            continue
        source = fallback_root / name
        if name == "LICENSE" and not source.is_file():
            source = fallback_root / license_file
        if not source.is_file():
            raise ValueError(f"missing release metadata: {source}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return patterns


def upload_release(
    repo: str,
    staging: Path,
    kind: str,
    tag: str | None = None,
    *,
    license_file: str,
    expected_source: dict | None = None,
    architectures: Iterable[str] | None = None,
    metadata_dir: Path | None = None,
    token: str | None = None,
) -> str:
    """Verify and publish a complete macos27 release of `kind` bundles."""
    patterns = prepare_release(staging, kind, license_file=license_file, expected_source=expected_source,
                               architectures=architectures, metadata_dir=metadata_dir)
    return upload_to_hub(repo, staging, tag, allow_patterns=patterns, delete_patterns=["macos27/**"], token=token)


def main(argv=None) -> int:
    """`python -m tinkerbox_ane.publish MODEL`: compile, stage and publish a catalog model's bundles to Hugging Face
    Hub; returns 1 if any target failed."""
    import argparse
    import importlib

    from tinkerbox_ane.bundle import read_bundle
    from tinkerbox_ane.host import SUPPORTED_ARCHITECTURES
    from tinkerbox_ane.models import CATALOG, entry

    parser = argparse.ArgumentParser(description="Compile, stage and publish a model's bundles to Hugging Face Hub.")
    # Models that share a repository in subfolders, such as GPT-2, are published by hand.
    parser.add_argument('model', choices=[item['id'] for item in CATALOG if 'subfolder' not in item])
    parser.add_argument('--source', type=Path, help='The export to compile; not needed with --upload-only')
    parser.add_argument('--staging', type=Path, help='Defaults to artifacts/huggingface/<repository name>')
    parser.add_argument('--repo', help="Defaults to the catalog's repository")
    parser.add_argument('--tag', help="Defaults to the catalog's revision")
    parser.add_argument('--arch', nargs='+', choices=list(SUPPORTED_ARCHITECTURES),
                        help="Defaults to the catalog's architectures")
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--metadata', type=Path, help='Directory with release README.md, LICENSE and NOTICE')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--no-upload', action='store_true', help='Compile and stage without uploading')
    modes.add_argument('--upload-only', action='store_true', help='Publish staged bundles')
    args = parser.parse_args(argv)
    if args.source is None and not args.upload_only:
        parser.error('--source is required unless --upload-only')
    item = entry(args.model)
    package = importlib.import_module(item['module'])
    kind, expected_source = package.KIND, package.upstream(args.model)
    staging = args.staging or Path('artifacts/huggingface') / item['repo'].split('/')[-1]
    targets = args.arch or item['architectures']
    architectures = targets
    if not args.upload_only:
        if read_bundle(args.source, kind)['source'] != expected_source:
            raise ValueError(f"export source does not match {item['name']}")
        architectures = list(build_architectures(kind, args.source, staging / 'macos27', targets,
                                                 force=args.force, continue_on_error=args.continue_on_error))
    metadata = dict(license_file=package.LICENSE_FILE, expected_source=expected_source, architectures=architectures,
                    metadata_dir=args.metadata)
    if args.no_upload:
        prepare_release(staging, kind, **metadata)
        print(f'Release staged in {staging}')
    else:
        commit = upload_release(args.repo or item['repo'], staging, kind, args.tag or item['revision'], **metadata)
        print(f'Published {args.repo or item["repo"]} at {commit}')
    return int(set(architectures) != set(targets))


if __name__ == '__main__':
    raise SystemExit(main())
