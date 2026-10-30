# Model Publishing Guide

`tinkerbox_ane` provides reusable utilities for Ahead-of-Time (AOT) Core AI compilation and publishing model releases to Hugging Face Hub. Model packages define their own export steps, deployment manifests, architecture support policies, and release metadata.

## Multi-Architecture Compilation

`build_architectures()` runs a model's compile function across target architectures:

```python
from pathlib import Path

from my_models.encoder import compile_model
from tinkerbox_ane.publish import build_architectures

staging = Path("artifacts/encoder-release")
bundles = build_architectures(
    compile_model,
    source=Path("artifacts/encoder-export"),
    out=staging / "macos28",
    architectures=["h17s", "h16g"],
)
```

The compile callable receives `source`, `out`, `architecture`, and `force` keyword arguments and returns the path to the compiled bundle. By default, the first failure raises immediately. Set `continue_on_error=True` to log errors and continue compiling remaining architectures; if all architectures fail, `RuntimeError` is raised.

## Core AI Asset Compilation

A model's `compile_model()` function can delegate to `tinkerbox_ane.coreai.compile_asset()` to compile `.aimodel` packages into `.aimodelc` bundles:

- **Verification:** Validates source identity, full ANE placement (`mps.fullyPlacedOnANE`), target architecture, and minimum deployment version before accepting an asset.
- **Safe Replacement:** Compiles into a temporary directory and verifies the result before replacing an existing target.
- **Rebuilds:** Pass `force=True` to recompile an existing target when sources or build configurations change. Bundle builds keep verified stages in a sibling `.<name>.build` directory after failure and resume on retry. Only a complete build replaces the deployment directory, using an atomic macOS directory swap.
- **Watchdog Protection:** Monitors compilation time, SSD write budgets, and free disk space, writing reports to `compile.json` and `compile.log` in the target's parent directory (pass `report=` to name them per asset).

Exports write the upstream files their runtime reads into a `checkpoint` folder with `tinkerbox_ane.bundle.write_checkpoint()`: everything but the weights, plus a `model.safetensors` with only the named tensors. Exports record its `checkpoint_hash`; compilation and publishing verify that identity. `compile_bundle()` and `install_bundle()` copy it into each bundle as an APFS clone, and `upload_release()` requires matching text, vision and checkpoint identities across architectures before publishing.

## Publishing to Hugging Face Hub

Stage release metadata (`README.md`, `LICENSE`, `NOTICE`) and compiled bundles in the staging directory, then publish:

```python
from pathlib import Path
from tinkerbox_ane.publish import upload_to_hub

staging = Path("artifacts/encoder-release")
patterns = ["README.md", "LICENSE"]
for architecture in bundles:
    patterns.append(f"macos28/{architecture}/**")

commit = upload_to_hub(
    "your-account/encoder-ANE",
    staging,
    tag="v1.0.0",
    allow_patterns=patterns,
    delete_patterns=["macos28/**"],
)
```

`upload_to_hub()` behaviors:
- **Authentication:** Uses saved Hugging Face credentials (`huggingface-cli login`) or an explicit `token`.
- **Filtering:** `allow_patterns` restricts uploaded files to specified paths.
- **Synchronization:** `delete_patterns` removes matching remote files absent from the upload directory, allowing clean replacement of a release tree. Without `delete_patterns`, existing remote files are preserved.
- **Release Tags:** If `tag` is specified, it creates a new release tag pointing to the uploaded commit. Pre-existing tags are rejected before uploading.

## Model builds

These commands are for developers creating bundles. End users download model
files through the app or `tinkerbox_ane.load()`.

Build from source with Xcode 27 and `uv sync --locked --extra speech`. Export to
empty directories. Compilation uses the existing watchdog and verifies source
identity, function manifests and full ANE placement. It defaults to this Mac's
architecture; standard M4 is `h16g`. Cross-compilation establishes placement,
not on-device correctness or performance.

### d1-3B

```sh
uv run python -m tinkerbox_ane.d1.export
uv run python -m tinkerbox_ane.d1.compile
```

The source is `artifacts/build/d1_fp16_fused128_ctx32768`; the compiled bundle is
`artifacts/d1_fp16_fused128_ctx32768`. Use `--force` to rebuild and
`--architecture` to select a different target.

### d3-nano and d3-mini

```sh
uv run python -m tinkerbox_ane.d3.export d3-nano --out artifacts/build/d3-nano
uv run python -m tinkerbox_ane.d3.compile artifacts/build/d3-nano --out artifacts/d3-nano
```

Substitute `d3-mini` for the larger variant. Failed bundle builds retain verified
stages and resume on retry.

### OmniVoice

```sh
uv run python -m tinkerbox_ane.omnivoice.export \
  --out artifacts/omnivoice/source --compile-to artifacts/omnivoice/ane
```

`--checkpoint` selects a local upstream checkpoint. The export includes diffusion,
fused short-context inference and the ANE acoustic codec decoder. M3 cannot place
the current codec graph; published bundles target M4 and newer.

### VoxCPM2

```sh
uv run python -m tinkerbox_ane.voxcpm2.export \
  --out artifacts/voxcpm2/source --compile-to artifacts/voxcpm2/ane
```

`--checkpoint` selects a local upstream checkpoint. See the export command's
`--help` for weight variants and stage options.

## Model publishing

`tinkerbox_ane.d1.publish` implements this workflow for the d1 model:

```sh
# Compile bundles for specific architectures without uploading
python -m tinkerbox_ane.d1.publish --arch h17s h16g --no-upload

# Publish prepared bundles and create a release tag
python -m tinkerbox_ane.d1.publish --arch h17s h16g --upload-only --tag <new-release-tag>
```

The other entry points are `tinkerbox_ane.d3.publish MODEL`,
`tinkerbox_ane.omnivoice.publish` and `tinkerbox_ane.voxcpm2.publish`.
Pass `--source` to choose the export and `--no-upload` to stage without publishing.
`--staging`, `--repo`, `--tag` and `--metadata` select the output and release
metadata. With `--upload-only`, publish previously staged bundles. Update
`src/tinkerbox_ane/models.json` after publishing.

To use a compiled bundle locally:

```sh
uv run python -m tinkerbox_ane.install <model-id> <bundle-directory>
```

## Validation

Run inference and compilation checks under `python -m tinkerbox_ane.guard`.
Report the device, bundle and whether the check was on-device or cross-compiled.

| Model | Reference and numerical checks | Public behavior |
| --- | --- | --- |
| d1-3B | `scripts/build_reference.py`, `scripts/validate.py` | `tests/test_runtime.py` |
| d3 | `scripts/validate_d3.py` | `tests/test_d3_runtime.py` |
| OmniVoice | `scripts/omnivoice/reference_logits.py`, `scripts/omnivoice/validate_context.py` | `tests/test_speech.py`, `app/test_speech.py` |
| VoxCPM2 | `scripts/voxcpm2/reference.py`, `scripts/voxcpm2/validate.py`, `scripts/voxcpm2/full_length.py` | `tests/test_voxcpm2_speech.py`, `app/test_voxcpm2_speech.py` |

Use each script's `--help` for checkpoint, bundle and output paths. D3's `--smoke`
is a partial numerical check; full acceptance requires the pinned FP32 reference
and coverage of all reachable capacities. Speech checks require a compiled bundle;
set the asset and checkpoint variables documented by the tests. Generated reports
belong under `artifacts/`.

Exercise speech Shortcuts through `app/test_shortcuts.py --speech`; use
`TINKERBOX_SPEECH_MODEL` and an absolute `TINKERBOX_ASSET` for each backend.
