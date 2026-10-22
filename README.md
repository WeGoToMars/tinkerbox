<img width="1920" height="388" alt="banner" src="https://github.com/user-attachments/assets/52cfb1e7-c841-4898-9a94-d199db85f2d7" />

# Tinkerbox for Apple Neural Engine

Run models on Apple's Neural Engine through Core AI. Critical inference always runs on the Neural Engine, the CPU does only supplementary work, and the GPU stays idle.

| Model | Task | Notes |
| --- | --- | --- |
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (`d1-3b`) | Decisions over text and images | [docs/models/d1.md](docs/models/d1.md) |
| [vllm-sr/d3-nano](https://huggingface.co/vllm-sr/d3-nano) (`d3-nano`), [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini) (`d3-mini`) | Decisions over text and images, one question per pass | [docs/models/d3.md](docs/models/d3.md) |

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

`tinkerbox_ane.load(model_id)` loads any model in the catalog and returns the API for its task. It downloads the Core AI bundle for this Mac to `~/Library/Caches/tinkerbox-ane/models/<model>/<revision>/`; pass a directory as the second argument to select a local AOT-compiled bundle. Each bundle carries a `checkpoint` folder with everything the runtime reads from the upstream checkpoint: its code, tokenizer and configuration, and only the weights that run on the CPU, such as the token embeddings. Loading never downloads the upstream checkpoint, and imports only upstream code whose digest the package pins. Each model package also has its own `load()`, such as `tinkerbox_ane.d1.load()`.

`python -m tinkerbox_ane.storage` reports, as JSON, each model's bundles, Core AI's compiled copies of their assets, and any upstream checkpoint in the Hugging Face cache, which exports and earlier versions download, then the compiled copies no downloaded bundle uses and the Python and packages the models share. `--remove MODEL` deletes all of a model's files; add `--outdated` to delete only bundle revisions the catalog no longer loads. `--remove-unused` deletes unused compiled copies, such as those of local exports, which compile again on their next load.

Decision models expose the System One API: `system_one`, `system_one_batch`, `probabilities`, and `probabilities_batch` for binary, choice, and score queries, plus `count_tokens`. Models with parallel decisions (`question_trees` in the catalog, such as d1-3B) answer several questions about one input in a shared pass; the others answer them one after another and warn when a call asks more than one. Inputs, limits, and execution details are in each model's notes.

## Tinkerbox app

A standalone desktop application is in [app](app). To build and launch it from source:

```sh
uv run --no-project --python 3.12 python app/run.py
```

Pass `--build-only` to create `artifacts/swiftui/Tinkerbox.app` and the release ZIP without launching. The prebuilt app manages its own Python environment and dependencies via an embedded `uv` binary.

The Storage page, opened from the bottom of the sidebar, lists every file the models keep on this Mac and deletes a model with everything it downloaded.

## Architecture & Staging

The Swift host manages persistent state and dynamically allocates attention cache buffers. Initial execution compiles and specializes the AOT asset for the local ANE hardware, after which Core AI reuses its persistent cache.

Weight staging uses an [APFS copy-on-write workaround](docs/cow-staging.md) to eliminate redundant disk writes during specialization.

## Adding a Model

Model-independent infrastructure is located at the package top level and is shared across models:

| Module | Role |
| --- | --- |
| `host.py`, `host.swift` | Native Core AI host: tiered caches, packed chains and trees, RoPE tables, and named vision inputs |
| `bundle.py` | Bundle manifest and asset identity verification, downloads, and AOT compilation |
| `coreai.py`, `guard.py`, `publish.py` | Compiler watchdog, ANE placement verification, and Hugging Face releases |
| `models.json`, `models.py` | Model catalog shared by `tinkerbox_ane.load()` and the app |
| `storage.py` | Disk usage of downloaded models, and their removal |

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

Each model's notes describe its export, compile, and publishing commands. Reusable Core AI compilation and Hugging Face publishing utilities are provided in `tinkerbox_ane.coreai` and `tinkerbox_ane.publish`. To use a compiled bundle locally before publishing:

```sh
uv run python -m tinkerbox_ane.install <model> <bundle directory>
```

## Tests

```sh
uv run python -m tinkerbox_ane.d1.guard artifacts/validation/smoke -- uv run pytest -q
```
