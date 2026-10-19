# Tinkerbox for Apple Neural Engine

Run [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) on Apple's Neural Engine through Core AI. FP16 backbone, 128-token chunks, 32,768-token context. Tokenization, embeddings and answer readout run on the CPU.

Requires Apple Silicon (M4 or newer), macOS 27, and Python 3.12 or 3.13. Earlier generations (M1–M3) are not supported due to Neural Engine hardware and offline compiler constraints.

## Quickstart

Install the wheel from the GitHub release (Xcode is not required):

```sh
python -m pip install ./tinkerbox_ane-*.whl
```

```python
from tinkerbox_ane import d1

with d1.load() as model:
    result = model.system_one(
        "I was charged twice this month.",
        {"refund": {"type": "noul", "instructions": "Is this a refund request?"}},
    )
    print(result)
```

`load()` downloads the matching Core AI architecture from [wegotomars/d1-3B-CoreAI-ANE](https://huggingface.co/wegotomars/d1-3B-CoreAI-ANE) (`v2.0.0`) into `~/Library/Caches/tinkerbox-ane/models/d1-3b/`. Pass a directory path to `load(path)` to select a local AOT bundle. This runtime requires the version 2 packed backbone; older bundles must be re-exported and compiled. Models can also be loaded by catalog ID: `tinkerbox_ane.load("d1-3b")`.

The API exposes `system_one`, `system_one_batch`, `probabilities`, and `probabilities_batch` for binary, choice, and score queries. Multiple decisions share prefix computation and run in packed chunks. Prompts exceeding 32,768 tokens are rejected.

## Images

Pass Pillow images with any request. The SigLIP2 vision tower runs on the Neural Engine as a separate Core AI asset; pictures above one megapixel are downscaled. The state may be `None` when the pictures carry the input.

```python
from PIL import Image

with d1.load() as model:
    result = model.system_one(None, {"receipt": {"type": "noul", "instructions": "Is this a receipt?"}},
                              images=[Image.open("photo.jpg")])
```

Image tokens join the shared trunk, so several questions about the same pictures encode them once.

## Multiple decisions in one pass

The backbone mirrors upstream Torch: the state is a shared trunk, and each question is an isolated branch. Attention masks prevent questions from seeing each other, convolution windows restart from the trunk history, and RoPE positions restart at the end of the trunk for each branch. Several short decisions can share one ANE invocation; longer trees stream through 128-token chunks.

```python
with d1.load() as model:
    result = model.system_one("Hi", {
        "greeting": {"type": "noul", "instructions": "Is this a greeting?"},
        "refund": {"type": "noul", "instructions": "Is this a refund request?"},
    })
```

Single-question batch requests are packed too. Each logical prompt must fit the 32,768-token context; trees exceeding that physical token budget split into groups. Multi-question usage counts the state once plus all question suffixes, matching upstream.

The bundle keeps nine cache-tier functions with shared weights. Single decisions skip tree planning and reuse cached routing and causal masks in the native host, updating only newly visible history. Packed requests rebuild branch-specific metadata. Both paths execute the same ANE graph.

## SwiftUI Demo

A desktop demo is available in [examples/swiftui](examples/swiftui). To build and launch it from source:

```sh
uv run --no-project --python 3.12 python examples/swiftui/run.py
```

Pass `--build-only` to create `artifacts/swiftui/Tinkerbox.app` and the release ZIP without launching. The prebuilt app manages its own Python environment and dependencies via an embedded `uv` binary.

## Architecture & Staging

The Swift host manages recurrent state and allocates attention cache buffers dynamically. Initial execution compiles and specializes the AOT asset for the local ANE hardware, after which Core AI reuses its persistent cache.

Weight staging uses an [APFS copy-on-write workaround](docs/cow-staging.md) to eliminate redundant disk writes during specialization.

## Adding a Model

Model-independent pieces live at the package top level and are shared by every model:

| Module | Role |
| --- | --- |
| `host.py`, `host.swift` | Native Core AI host: tiered caches, packed chains and trees, RoPE tables, and a vision asset with named inputs |
| `bundle.py` | Bundle manifest kind and asset identity checks, per-architecture download with progress, AOT compilation; each runtime validates the manifest fields it uses |
| `coreai.py`, `guard.py`, `publish.py` | Compiler watchdog, ANE placement verification, Hugging Face releases |
| `models.json`, `models.py` | The model catalog read by `tinkerbox_ane.load()` and the app |

A model package such as `tinkerbox_ane.d1` supplies the rest: the fused FP16 graph (`model.py`), its export (`export.py`), tokenization and answer readout (`runtime.py`), decision-tree planning over LFM convolution routing (`tree.py`), and optionally a vision tower (`vision.py`). It exposes `load(asset=None)` and a bundle `KIND`; its `compile` and `publish` CLIs are thin wrappers over the shared modules. Add a catalog entry with the model's artifact repository, revision, inputs, and the first `min_version` that ships it. Older apps list the entry but keep it disabled until updated.

## Building from Source

Building requires Xcode 27 / Swift 6.4 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv build
```

The wheel in `dist/` bundles the native host and links against macOS system frameworks. Attach it to the GitHub release. Source and editable installs compile the host during installation.

To export and compile model weights locally:

```sh
uv run python -m tinkerbox_ane.d1.export
uv run python -m tinkerbox_ane.d1.compile
```

Export writes intermediate artifacts to `artifacts/build/d1_fp16_fused128_ctx32768`, and compilation produces the final deployment bundle in `artifacts/d1_fp16_fused128_ctx32768`. Load the local result with `d1.load(d1.DEFAULT_ASSET)`.

Key compiler options:
- `--force`: Rebuild an existing bundle. The compiler verifies the new artifact before replacing the previous build.
- `--architecture <arch>`: Target a specific architecture instead of the current machine.
- `--allow-unsupported`: Attempt compilation for an architecture outside the verified support list.

Compilation verifies source identity, function manifests, and full ANE placement. Without cached sudo credentials, worker cancellation on failure or timeout is best effort and emits a warning.

Reusable Core AI compilation and Hugging Face publishing utilities are provided in `tinkerbox_ane.coreai` and `tinkerbox_ane.publish`. See the [model publishing guide](docs/model-publishing.md) for compiling and publishing additional models.

### Publishing d1 Releases

Use the d1 publishing CLI to build and upload multi-architecture release bundles:

```sh
# Compile bundles for specific architectures without uploading
uv run python -m tinkerbox_ane.d1.publish --arch h17s h16g --no-upload

# Publish prepared bundles and create a release tag
uv run python -m tinkerbox_ane.d1.publish --arch h17s h16g --upload-only --tag v2.0.0
```

Omitting `--arch` targets all supported architectures. The upload synchronizes the remote `macos27` tree with the selected architectures, pruning omitted targets and obsolete files while keeping existing release tags intact.

Common publishing options:
- `--arch`: Target architectures to build (defaults to all supported architectures).
- `--tag`: New release tag to create on Hugging Face (must not already exist).
- `--source`, `--staging`, `--repo`: Source export directory, local staging path, and Hugging Face repository.
- `--metadata`: Directory containing release metadata files (`LICENSE`, `NOTICE`, `README.md`).

When publishing new bundles, update the model's `revision` in `src/tinkerbox_ane/models.json` so `load()` resolves the new tag by default. (The legacy script `scripts/publish_all_chips.py` forwards to this CLI.)

### Releasing the Package

```sh
uv run python scripts/release.py --sign "Apple Development: you@example.com"
```

From a clean tree, the script checks that every catalog model shipped by this version has its Hugging Face tag, runs the tests, builds the wheel and the signed app ZIP, and creates a local annotated tag. It pushes and uploads nothing; it prints the `git push`, `uv publish`, and `gh release create` commands to run after review.

## Validation

```sh
uv run python scripts/build_reference.py
uv run python -m tinkerbox_ane.d1.guard artifacts/validation/check -- uv run python scripts/validate.py
uv run python -m tinkerbox_ane.d1.guard artifacts/validation/smoke -- uv run pytest -q
```

Reference generation downloads benchmark inputs and runs the official FP32 CPU model. Validation checks prompt tokens and probabilities across all nine cache capacities. The two public API smoke tests require a local bundle and cached checkpoint.

Weights and upstream code are pinned to `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432` under the [LFM Open License v1.0](LICENSES/LFM-Open-License-v1.0.txt). See [NOTICE](NOTICE).
