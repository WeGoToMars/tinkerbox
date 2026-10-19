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

## Tinkerbox app

A standalone desktop application is in [app](app). To build and launch it from source:

```sh
uv run --no-project --python 3.12 python app/run.py
```

Pass `--build-only` to create `artifacts/swiftui/Tinkerbox.app` and the release ZIP without launching. The prebuilt app manages its own Python environment and dependencies via an embedded `uv` binary.

## Architecture & Staging

The Swift host manages recurrent state and allocates attention cache buffers dynamically. Initial execution compiles and specializes the AOT asset for the local ANE hardware, after which Core AI reuses its persistent cache.

Weight staging uses an [APFS copy-on-write workaround](docs/cow-staging.md) to eliminate redundant disk writes during specialization.

## Adding a Model

Model-independent infrastructure lives at the package top level and is shared across models:

| Module | Role |
| --- | --- |
| `host.py`, `host.swift` | Native Core AI host: tiered caches, packed chains and trees, RoPE tables, and named vision inputs |
| `bundle.py` | Bundle manifest and asset identity verification, downloads, and AOT compilation |
| `coreai.py`, `guard.py`, `publish.py` | Compiler watchdog, ANE placement verification, and Hugging Face releases |
| `models.json`, `models.py` | Model catalog shared by `tinkerbox_ane.load()` and the app |

A model package (such as `tinkerbox_ane.d1`) provides:
- **Neural Graph & Export:** Fused FP16 graph (`model.py`) and Core AI export (`export.py`).
- **Runtime:** Tokenization, prompt formatting, readout (`runtime.py`), and optional tree planning (`tree.py`) or vision processing (`vision.py`).
- **Package Interface:** Exposes `load(asset=None)`, bundle `KIND`, and compile/publish CLI wrappers.

To register a model, add an entry to `src/tinkerbox_ane/models.json` specifying `id`, `name`, `publisher`, `summary`, `task`, `repo`, `revision`, and `min_version`. Add notes in `docs/models/<id>.md`.

## Building from Source

Building requires Xcode 27 / Swift 6.4 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv build
```

The wheel in `dist/` bundles the native `tinkerbox-ane-host` executable and links against macOS system frameworks. Source and editable installs compile the host during installation. Compilation verifies source identity, function manifests, and full ANE placement.

Each model's notes describe its export, compile, and publishing commands. Reusable Core AI compilation and Hugging Face publishing utilities are provided in `tinkerbox_ane.coreai` and `tinkerbox_ane.publish`; see the [model publishing guide](docs/model-publishing.md). To use a compiled bundle locally before publishing:

```sh
uv run python -m tinkerbox_ane.install <model> <bundle directory>
```

### Continuous Integration and Releases

The Build workflow runs on pushes and pull requests on GitHub's Xcode 27 runners: it builds the wheel with the native host, type-checks the app and the Shortcuts smoke target, and builds the app. Tests, model export and compilation, and Hugging Face publishing stay local, since they need the Neural Engine and the checkpoints.

To release, run from a clean tree:

```sh
uv run python scripts/release.py
```

The script checks that every catalog model shipped by this version has its Hugging Face tag, runs the tests, and creates a local annotated tag; it pushes nothing. Pushing the tag starts the Release workflow, which builds the wheel and the app signed with the developer identity, publishes the wheel to PyPI through trusted publishing, then attaches both to a draft GitHub release. The app installs that wheel from PyPI on first launch. The PyPI trusted publisher uses the `WeGoToMars/tinkerbox` repository, `release.yml` workflow, and `pypi` environment. The workflow reads the repository secrets `APPLE_DEVELOPER_P12` (the base64-encoded identity) and `APPLE_DEVELOPER_P12_PASSWORD`; no PyPI token is needed.

## Tests

```sh
uv run python -m tinkerbox_ane.d1.guard artifacts/validation/smoke -- uv run pytest -q
```

The public API tests require a local bundle and cached checkpoint. Reference validation against the official model is per model; see its notes.

## Licenses

Model weights and upstream code keep their own licenses: d1-3B is pinned to `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432` under the [LFM Open License v1.0](LICENSES/LFM-Open-License-v1.0.txt). See [NOTICE](NOTICE).
