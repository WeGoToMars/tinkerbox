# Tinkerbox for Apple Neural Engine

Run models on Apple's Neural Engine through Core AI. Critical inference runs on the Neural Engine, the CPU does supplementary work such as tokenization and small heads, and the GPU stays idle.

| Model | Task | Notes |
| --- | --- | --- |
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (`d1-3b`) | Decisions over text and images | [docs/models/d1.md](docs/models/d1.md) |

Requires Apple Silicon (M4 or newer), macOS 27, and Python 3.12 or 3.13. Earlier generations (M1–M3) are not supported due to Neural Engine hardware and offline compiler constraints.

## Quickstart

Install the wheel from the GitHub release (Xcode is not required):

```sh
python -m pip install ./tinkerbox_ane-*.whl
```

```python
import tinkerbox_ane

with tinkerbox_ane.load("d1-3b") as model:
    result = model.system_one(
        "I was charged twice this month.",
        {"refund": {"type": "noul", "instructions": "Is this a refund request?"}},
    )
    print(result)
```

`tinkerbox_ane.load(model_id)` loads any model in the catalog and returns the API for its task. It downloads the Core AI bundle for this Mac into `~/Library/Caches/tinkerbox-ane/models/<model>/<revision>/`; pass a directory as the second argument to select a local AOT bundle. Each model package also has its own `load()`, such as `tinkerbox_ane.d1.load()`.

Decision models expose the System One API: `system_one`, `system_one_batch`, `probabilities`, and `probabilities_batch` for binary, choice, and score queries, plus `count_tokens`. Several questions about one input share its computation. Inputs, limits, and execution details are in each model's notes.

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

A model package such as `tinkerbox_ane.d1` supplies the rest: the fused FP16 graph (`model.py`), its export (`export.py`), tokenization and answer readout (`runtime.py`), decision-tree planning over LFM convolution routing (`tree.py`), and optionally a vision tower (`vision.py`). It exposes `load(asset=None)` and a bundle `KIND`; its `compile` and `publish` CLIs are thin wrappers over the shared modules. Add a catalog entry with the model's `task` (`decision`; planned: `embedding`, `transcription`, `speech`, `segmentation`), optional `base_model` (the foundation it was fine-tuned from) and `original_model` (the model that was converted), artifact repository, revision, inputs, and the first `min_version` that ships it. Older apps list the entry but keep it disabled until updated, and skip entries they cannot read; only `id`, `name`, `publisher`, `summary`, `task`, `repo`, `revision` and `min_version` are required. Write the model's implementation notes in `docs/models/<package>.md` and link them from the table at the top.

## Building from Source

Building requires Xcode 27 / Swift 6.4 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv build
```

The wheel in `dist/` bundles the native host and links against macOS system frameworks. Attach it to the GitHub release. Source and editable installs compile the host during installation.

Compilation verifies source identity, function manifests, and full ANE placement. Without cached sudo credentials, worker cancellation on failure or timeout is best effort and emits a warning.

Each model's notes describe its export, compile, and publishing commands. Reusable Core AI compilation and Hugging Face publishing utilities are provided in `tinkerbox_ane.coreai` and `tinkerbox_ane.publish`; see the [model publishing guide](docs/model-publishing.md). When publishing new bundles, update the model's `revision` in `src/tinkerbox_ane/models.json`. To use a compiled bundle before its revision is on Hugging Face, install it into the model cache; `load()`, the app, and Shortcuts then use it in place of the download:

```sh
uv run python -m tinkerbox_ane.install <model> <bundle directory>
```

### Releasing the Package

```sh
uv run python scripts/release.py --sign "Apple Development: you@example.com"
```

From a clean tree, the script checks that every catalog model shipped by this version has its Hugging Face tag, runs the tests, builds the wheel and the signed app ZIP, and creates a local annotated tag. It pushes and uploads nothing; it prints the `git push`, `uv publish`, and `gh release create` commands to run after review.

## Tests

```sh
uv run python -m tinkerbox_ane.d1.guard artifacts/validation/smoke -- uv run pytest -q
```

The public API tests require a local bundle and cached checkpoint. Reference validation against the official model is per model; see its notes.

## Licenses

Model weights and upstream code keep their own licenses: d1-3B is pinned to `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432` under the [LFM Open License v1.0](LICENSES/LFM-Open-License-v1.0.txt). See [NOTICE](NOTICE).
