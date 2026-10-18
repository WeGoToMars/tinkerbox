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
- **Rebuilds:** Pass `force=True` to recompile an existing target when sources or build configurations change.
- **Watchdog Protection:** Monitors compilation time, SSD write budgets, and free disk space, writing reports to `compile.json` and `compile.log` in the target's parent directory (pass `report=` to name them per asset).

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

## Reference CLI: d1

`tinkerbox_ane.d1.publish` implements this workflow for the d1 model:

```sh
# Compile bundles for specific architectures without uploading
python -m tinkerbox_ane.d1.publish --arch h17s h16g --no-upload

# Publish prepared bundles and create a release tag
python -m tinkerbox_ane.d1.publish --arch h17s h16g --upload-only --tag v2.0.0
```
