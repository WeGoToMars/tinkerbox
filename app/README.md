# SwiftUI Demo

A standalone macOS application demonstrating `tinkerbox-ane` on Apple Silicon.

Requires macOS 27 on Apple Silicon.

## Prebuilt App

1. Download and extract the latest `Tinkerbox-ANE-*-macos-arm64.zip` from the GitHub releases page.
2. Move **Tinkerbox.app** to `/Applications` and launch it. The app is not notarized, so macOS blocks the first launch: allow it under **System Settings → Privacy & Security → Open Anyway**, or run `xattr -dr com.apple.quarantine Tinkerbox.app`.

The app bundles a standalone `uv` binary. On first run, it configures an isolated Python 3.12 environment, installs the precompiled `tinkerbox-ane` wheel, and fetches the selected model's weights from Hugging Face. Subsequent launches reuse the cached environment.

The app downloads the bundle matching the host Mac's Core AI architecture (`h16g`/`h16s`/`h16c` for M4/Pro/Max, `h17g`/`h17s`/`h17c`/`h17d` for M5/Pro/Max/Ultra). The Ultra mapping follows compiler research; Core AI detection and inference on an Ultra remain unverified. Setup progress and logs are written to `~/Library/Logs/tinkerbox-ane/demo.log`.

## Building from Source

Building requires Xcode 27 and [uv](https://docs.astral.sh/uv/):

```sh
uv run --no-project --python 3.12 python app/run.py --build-only
```

This generates `artifacts/swiftui/Tinkerbox.app` and a distributable ZIP archive. Omit `--build-only` to launch immediately after building.

To enable Shortcuts App Intents, sign with a development identity (`security find-identity -v -p codesigning`):

```sh
uv run --no-project --python 3.12 python app/run.py --sign "Apple Development: you@example.com"
```

To bundle a locally built wheel before release:

```sh
uv build --wheel
uv run --no-project --python 3.12 python app/run.py --wheel dist/tinkerbox_ane-*.whl
```

To run against a local model bundle:

```sh
TINKERBOX_ASSET="$PWD/artifacts/d1_fp16_fused128_ctx32768" uv run --no-project --python 3.12 python app/run.py
```

## Supported Tasks

- `noul`: Binary yes/no decisions with normalized probabilities.
- `choice`: Selection across two or more discrete options.
- `score`: Ranked evaluation across 2–10 ordered levels.

Each question is configured as a card. With multiple questions, models supporting decision trees offer **Parallel** execution (a single shared pass over the input) or **Sequential** execution (independent passes per question).

## Model Management

The sidebar **Models** view displays the catalog grouped by task, fetched from `models.json` on the main branch (falling back to the bundled catalog offline). Downloaded models can be activated or removed; entries requiring a newer app version remain disabled until updated. Downloads run in the background.

**Model Settings** in the task toolbar configures parameters declared by the active model (such as `max_image_tiles`). Settings persist per model and apply to both UI runs and Shortcuts automations.

The model unloads after five minutes without inference. **Unload** in the task toolbar releases it immediately, after any active request finishes. The selection and settings stay saved; the next UI run or Shortcut loads the model again. Token previews do not extend the idle timeout or reload an unloaded model. Closing the window keeps the app running; quitting stops its worker processes.

Downloaded bundles are stored in `~/Library/Caches/tinkerbox-ane/models/<model>/`.

## Images

When the selected model supports vision, images can be added via drag-and-drop, pasteboard (⌘V), or file picker. Text input becomes optional when an image is attached. Images are automatically rotated via EXIF orientation and partitioned into tiles based on the model's tile limit setting.

Clicking an attached picture shows the tile grid layout and resolution retained by the vision tower. The token counter next to **Run** previews input tokens and vision tiles before execution.

## Shortcuts

Build and launch **Tinkerbox.app** once to complete initial model setup. In Shortcuts, search for **Tinkerbox** in the action library:

| Action | Inputs | Result |
| --- | --- | --- |
| **Yes/No Decision** | Input Text, Question, Images, Yes Threshold (default 0.5) | Text formatted as `Yes (93%)`; typed properties: **Question**, **Answer** (Boolean), **Yes Probability** (0–1) |
| **Choose Option** | Input Text, Question, Options (≥2 unique), Images, Minimum Confidence | Text formatted as `Refund request (81%)`; typed properties: **Chosen Option**, **Most Likely Option**, **Confidence** |
| **Score Text** | Input Text, Question, Levels (2–10 ordered labels), Images | Text formatted as `Friendly (1.8 / 2)`; typed properties: **Score**, **Maximum Score**, **Most Likely Level** |
| **Ask Several Questions** | Input Text, Questions (List), Images, Yes Threshold | List of Yes/No Decisions from a single shared pass over the input |
| **Filter List by Question** | List, Question, Yes Threshold | Items evaluating to Yes, preserving input order |
| **Sort List by Score** | List, Question, Levels, Highest First (default on) | Ranked items with attached **Score**, **Maximum Score**, and **Most Likely Level** |

Results format as plain text when passed to Text actions or notifications, while exposing typed properties for conditional logic.

**Images** is optional and accepts image files (e.g., from **Take Screenshot** or **Select Photos**); **Input Text** may be omitted when images are supplied.

**Ask Several Questions** reads the input once and answers every question in one pass, so extra questions cost little. List actions process items together in a single batch, automatically dropping blank lines.

Actions run in the background, sharing the loaded runtime model and serializing requests. Setup requires network access to download weights; inference executes entirely on-device.

The source build runs Xcode's App Intents metadata extractor before signing. Keep `Contents/Resources/Metadata.appintents` in the app bundle so the actions remain discoverable. Every Tinkerbox build shares one bundle identifier, so keep a single copy installed: Shortcuts may resolve the actions to an older copy (1.x has none) or to an ad-hoc signed build, which fails with "Shortcuts couldn't communicate with the app".

To exercise the actions against real inference, including concurrent calls with the UI runtime and recovery after errors:

```sh
python3 app/run.py --build-only
python3 app/test_shortcuts.py
```

The smoke test uses the built app's bundled worker and pinned package requirements. It requires the same supported Apple Silicon hardware and cached/downloadable model as the app. Set `TINKERBOX_ASSET` to use a local compatible model bundle. It calls the App Intent entry points directly; discovery and composing result properties should also be checked in the Shortcuts app after a build.
