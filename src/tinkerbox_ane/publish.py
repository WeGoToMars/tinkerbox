"""Helpers for compiling multi-architecture models and publishing releases to Hugging Face Hub."""

from collections.abc import Callable, Iterable
import logging
from pathlib import Path
import time


def build_architectures(
    compile_model: Callable[..., Path],
    source: Path,
    out: Path,
    architectures: Iterable[str],
    *,
    force: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Path]:
    """Compile a model across target architectures and return paths to successful bundles."""
    architectures = list(dict.fromkeys(architectures))
    started = time.monotonic()
    print(f"Building {len(architectures)} architectures: {', '.join(architectures)}", flush=True)
    bundles = {}
    for index, architecture in enumerate(architectures, 1):
        label = f"[{index}/{len(architectures)}] {architecture}"
        architecture_started = time.monotonic()
        print(f"{label}: preparing bundle", flush=True)
        try:
            bundles[architecture] = compile_model(
                source=source, out=Path(out) / architecture,
                architecture=architecture, force=force,
            )
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

    if tag is not None and not tag.strip():
        raise ValueError("tag must not be empty")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    if tag is not None and any(ref.name == tag for ref in api.list_repo_refs(repo).tags):
        raise ValueError(f"release tag {tag!r} already exists; choose a new version")
    commit = api.upload_folder(repo_id=repo, folder_path=staging, allow_patterns=allow_patterns,
                               delete_patterns=delete_patterns,
                               commit_message="Publish model bundles")
    if tag is not None:
        api.create_tag(repo, tag=tag, revision=commit.oid)
    return commit.oid
