# SwiftUI Demo

A standalone macOS application demonstrating `tinkerbox-ane` on Apple Silicon.

Requires macOS 27 on Apple Silicon.

## Prebuilt App

1. Download and extract the latest `Tinkerbox-ANE-*-macos-arm64.zip` from the GitHub releases page.
2. Move **Tinkerbox.app** to `/Applications` and launch it. The app is not notarized, so macOS blocks the first launch: allow it under **System Settings → Privacy & Security → Open Anyway**, or run `xattr -dr com.apple.quarantine Tinkerbox.app`.

The app bundles a standalone `uv` binary. On first run, it configures an isolated Python 3.12 environment, installs the precompiled `tinkerbox-ane` wheel, and fetches the selected model's weights from Hugging Face. Subsequent launches reuse the cached environment.

The app detects the Mac's Core AI architecture directly and supports M1–M6.
Each model declares its available architectures in the shared catalog. Models
without a bundle for this Mac explain why and offer a local bundle picker;
another chip's cached bundle never counts as downloaded. Compatible local
bundles can be used before an architecture's release is published. The picker
rejects mismatched bundles before changing the selection. On-device correctness
and performance are checked on M5 Pro; other targets are cross-compiled.
Setup progress and logs are written to `~/Library/Logs/tinkerbox-ane/demo.log`.

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

## Text to Speech

OmniVoice and VoxCPM2 support automatic voice, voice design, reference-audio cloning and saved
voice prompts. The Text to Speech page provides generation controls, playback and
WAV/M4A export, and displays the seed for replay. Voice preparation and speech
generation times are shown separately. With models that stream, such as VoxCPM2,
**Play while generating** (on by default) starts playback as the first audio
arrives and shows the time to first audio; the finished clip then replaces the live
player for replay and export. Select a verified local bundle in
the page toolbar or set `TINKERBOX_ASSET` before launch. Bundles are not distributed
yet; see local setup and validation for [OmniVoice](../docs/models/omnivoice.md)
and [VoxCPM2](../docs/models/voxcpm2.md).
Saved voices are specific to the selected model.
Synthesis modes and control fields come from the backend's [speech capabilities](../docs/speech.md).
The page shows native numeric, categorical, text and optional controls as supported
by the selected model; switching models clears incompatible values and saved prompts.

Reference audio, including M4A and WAV, is decoded using Apple's AVFoundation codecs.
Modes that need a reference transcript transcribe it on-device using Apple's
SpeechTranscriber. VoxCPM2's reference-only clone mode uses the audio directly;
its continuation modes show the transcript controls. Saved voice preparation
retains the selected conditioning mode.
Choose the recording's language and review or edit its transcript before cloning.
The system downloads speech recognition assets when needed; setup and transcription
status appear beside the transcript. You can also enter the transcript manually.
Click **Arm microphone**, then hold Space to record and release to finish.
Click again to disarm. Samples show the same waveform player
as generated speech. Recording stops at 20 seconds or when the app loses focus.

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
