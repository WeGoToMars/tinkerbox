# Developer guide

Building the package, exporting and publishing models, and testing. End users get model bundles through the app or `tinkerbox_ane.load()` and need none of this.

## Build from source

Requires Xcode 27 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked --all-extras
uv build
```

Installing the package compiles the native host, `tinkerbox-ane-host`, from `src/tinkerbox_ane/*.swift` and `cow-staging.m`. The wheel in `dist/` includes it. To build the app, see the [app guide](../app/README.md#development).

## Layout

Shared infrastructure sits at the package top level:

| Module | Role |
| --- | --- |
| `host.py`, `host.swift`, `cached.swift`, `chunked.swift`, `staged.swift`, `bidirectional.swift` | Native Core AI host: KV caches by capacity, packed batches and question trees, RoPE tables, vision inputs |
| `bundle.py`, `compile.py`, `install.py` | Bundle manifests, downloads, AOT compilation and local installs |
| `coreai.py`, `guard.py`, `publish.py` | Core AI export helpers, the compiler watchdog, and Hugging Face releases |
| `generation.py`, `completion.py`, `speech.py`, `tokenization.py` | Task APIs shared by model families |
| `models.json`, `models.py`, `storage.py` | The catalog shared by `load()` and the app, and disk usage |

A model package, such as `tinkerbox_ane.d1`, has a graph (`model.py`), a Core AI export (`export.py`), a runtime (`runtime.py`) and optional vision or tree planning. Its `__init__.py` defines `KIND` (the bundle kind), `load(asset)`, and the `SETTINGS` and `INFO` the app shows in **Model Settings**. Speech packages also define `SPEECH_PARAMETERS`.

To add a model, register it in `src/tinkerbox_ane/models.json`: `id`, `name`, `publisher`, `summary`, `task`, `module`, `repo`, `revision`, `min_version` and `architectures`, plus `subfolder`, `context`, `inputs`, `question_trees`, `download_bytes` and `license` where they apply. `family`, `size` and `quantization` group variants on the app's Models page. Add usage notes in `docs/models/<family>.md`.

## Export and compile

Exports need an empty destination. Compile with `tinkerbox_ane.compile`, which reads the kind from the export's manifest:

```sh
uv run python -m tinkerbox_ane.compile SOURCE --out BUNDLE [--architecture h16g] [--force]
```

Compilation runs under the watchdog, verifies source identity, function manifests and full ANE placement, and defaults to this Mac's architecture (standard M4 is `h16g`). A failed build keeps its verified stages and resumes on retry. Cross-compiling verifies placement, not on-device correctness or speed.

| Model | Export |
| --- | --- |
| d1-3B | `python -m tinkerbox_ane.d1.export` (to `artifacts/build/d1_fp16_fused128_ctx32768`) |
| d3 | `python -m tinkerbox_ane.d3.export d3-nano --out artifacts/build/d3-nano`; `--no-vision` exports text only |
| OmniVoice | `python -m tinkerbox_ane.omnivoice.export` (to `artifacts/omnivoice/export`) |
| VoxCPM2 | `python -m tinkerbox_ane.voxcpm2.export` (to `artifacts/voxcpm2/export`); `--main-weights` selects LUT8 |
| GPT-2 | `python -m tinkerbox_ane.gpt2.export gpt2 --batch-sizes 1 4 8 --out artifacts/gpt2/export` |

`--checkpoint` selects a local upstream checkpoint. GPT-2 takes any size (`gpt2-medium`, `gpt2-large`, `gpt2-xl`) and `--weights int8-per-channel` or `lut8-group32`. Each export writes a `checkpoint` folder with the tokenizer, configuration and CPU weights, and records its hash; compilation and publishing check it.

To use a bundle locally before publishing:

```sh
uv run python -m tinkerbox_ane.install MODEL BUNDLE
```

## Publish

Each family except GPT-2 has a release command that compiles every catalog architecture, stages the bundles with their license, and uploads them as a tagged Hugging Face release:

```sh
uv run python -m tinkerbox_ane.d1.publish --no-upload       # compile and stage only
uv run python -m tinkerbox_ane.d1.publish --upload-only     # publish the staged bundles
```

The others are `tinkerbox_ane.d3.publish MODEL --source EXPORT`, `tinkerbox_ane.omnivoice.publish` and `tinkerbox_ane.voxcpm2.publish`. `--arch`, `--staging`, `--repo`, `--tag` and `--metadata` override the catalog defaults; `--continue-on-error` keeps going past a failed architecture. Update `revision` in `models.json` after publishing.

GPT-2 is published by hand: every size and format shares [wegotomars/GPT2-CoreAI-ANE](https://huggingface.co/wegotomars/GPT2-CoreAI-ANE), with bundles under `<model>/<format>/macos27/<architecture>` (the catalog's `subfolder`). Compile each export with `tinkerbox_ane.publish.build_architectures`, then upload the tree with `upload_to_hub(repo, staging, tag, allow_patterns=..., delete_patterns=...)`.

## Tests

Run tests under the watchdog, with a new report path each time:

```sh
uv run python -m tinkerbox_ane.guard artifacts/validation/run-1 -- uv run pytest -q -rs
```

Model tests use this Mac's installed catalog bundle, or a bundle from `D1_ASSET`, `D3_NANO_ASSET`, `D3_MINI_ASSET`, `GPT2_ASSET` (`GPT2_XL_LUT8_ASSET` and so on), `TINKERBOX_SPEECH_TEST_ASSET` or `TINKERBOX_VOXCPM2_TEST_ASSET`. Without one they skip; `scripts/release.py` fails on any skip. The OmniVoice portability test also needs `TINKERBOX_OMNIVOICE_CHECKPOINT`. App behavior is covered by the [Swift smokes](../app/README.md#development).

Numerical checks compare against upstream references. Report the device, the bundle, and whether a check ran on-device or was cross-compiled.

| Model | Reference checks | Tests |
| --- | --- | --- |
| d1-3B | `scripts/build_reference.py`, `scripts/validate.py` | `tests/test_runtime.py` |
| d3 | `scripts/validate_d3.py` (`--smoke` is partial; full acceptance needs the FP32 reference) | `tests/test_d3_runtime.py` |
| OmniVoice | `scripts/omnivoice/reference_logits.py`, `scripts/omnivoice/validate_context.py` | `tests/test_speech.py` |
| VoxCPM2 | `scripts/voxcpm2/reference.py`, `scripts/voxcpm2/validate.py`, `scripts/voxcpm2/full_length.py` | `tests/test_voxcpm2_speech.py` |
| GPT-2 | — | `tests/test_gpt2_generation.py` |

Each script's `--help` lists its paths. Reports belong under `artifacts/`.

## Maintenance

- [Core AI disk cleanup](cache-cleanup.md): removes leftover specializations and staging directories.
- [Copy-on-write staging](cow-staging.md): the host workaround that stops weight files being copied during specialization.
