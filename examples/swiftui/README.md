# SwiftUI Demo

A standalone macOS application demonstrating `tinkerbox-ane` on Apple Silicon.

Requires macOS 27 on Apple Silicon.

## Prebuilt App

1. Download and extract the latest `Tinkerbox-ANE-*-macos-arm64.zip` from the GitHub releases page.
2. Move **Tinkerbox.app** to `/Applications` and launch it. The app is not notarized, so macOS blocks the first launch: allow it under **System Settings → Privacy & Security → Open Anyway**, or run `xattr -dr com.apple.quarantine Tinkerbox.app`.

The app bundles a standalone `uv` binary. On first run, it configures an isolated Python 3.12 environment, installs the precompiled `tinkerbox-ane` wheel, and fetches the selected model's weights from Hugging Face. Subsequent launches reuse the cached environment.

The default model bundle targets `h17s` (M5-family) Neural Engine cores. Setup progress and error logs are written to `~/Library/Logs/tinkerbox-ane/demo.log`.

## Building from Source

Building requires Xcode 27 and [uv](https://docs.astral.sh/uv/):

```sh
uv run --no-project --python 3.12 python examples/swiftui/run.py --build-only
```

This generates `artifacts/swiftui/Tinkerbox.app` and a distributable ZIP archive. Omit `--build-only` to launch the application immediately after building.

The app is ad-hoc signed by default, which is enough for the UI, but Shortcuts refuses to run actions from an app without a team identity. Pass a signing identity from `security find-identity -v -p codesigning` to use the actions:

```sh
uv run --no-project --python 3.12 python examples/swiftui/run.py --sign "Apple Development: you@example.com"
```

Until the matching `tinkerbox-ane` version is on PyPI, bundle a locally built wheel; the app then installs it from inside the bundle:

```sh
uv build --wheel
uv run --no-project --python 3.12 python examples/swiftui/run.py --wheel dist/tinkerbox_ane-*.whl
```

To test against a local model bundle instead of downloading from Hugging Face:

```sh
TINKERBOX_ASSET="$PWD/artifacts/d1_fp16_fused128_ctx32768" uv run --no-project --python 3.12 python examples/swiftui/run.py
```

## Supported Tasks

- `noul`: Binary decisions with normalized probabilities.
- `choice`: Selection across two or more discrete options.
- `score`: Ranked evaluation across 2–10 ordered levels.

Each question is a card with its own type, instructions and options; **Add question** asks several about the same input, and each answer appears in its card. With more than one question, models that support decision trees offer two modes: **Parallel** answers every question in one shared pass over the input, while **Sequential** gives each question its own pass and shows its time as it finishes. Other models always run sequentially.

## Models

The sidebar lists a page per task, and **Models** lists the catalog grouped by task. The app fetches [`models.json`](../../src/tinkerbox_ane/models.json) from the repository's `main` branch and falls back to the copy bundled with the app when offline. **Download** fetches a model with byte progress in its row and switches to it; **Use** switches to a model already on disk, and the trash button removes one. The arrow menu opens a model's Hugging Face pages: the **base model** it was fine-tuned from, the **original model** that was converted, and the **converted model** with its Core AI bundles. The model menu in a task page's toolbar switches between that task's downloaded models. Entries newer than the installed app, or of tasks it does not run, are listed with the version they require. Each task remembers its own model across launches, which Shortcuts also use; one model is loaded at a time, and switching cancels a download in progress.

The sidebar button at the right of a task page's toolbar opens **Model Settings**: values the loaded model lets you change, such as d1's **Image tiles**, with its fixed properties below. Values are kept per model, also apply to Shortcuts, and **Reset to Defaults** clears them. The model's Python package declares them, so they match the installed version.

Downloaded bundles live in `~/Library/Caches/tinkerbox-ane/models/<model>/`.

## Images

When the selected model accepts images, an **Images** strip appears above the input text. Drop pictures onto it, paste them, or click **Add**; text becomes optional while an image is attached. Photos are rotated by their EXIF orientation, and large ones are split into tiles up to the model's **Image tiles** setting. Each picture shows its tile grid and, below it, the grid size and how much of its resolution the model keeps. Click a picture to see what the model sees: the picture resized to fill its tiles, at their real pixel size when they fit, and the overview it gets of the whole picture, with the sizes, any stretching, vision passes and tokens.

Pasting with ⌘V anywhere on the Decide page adds copied pictures or image files to the strip and copied text to the input text; a focused text field pastes text as usual.

Next to **Run**, the app shows how many tokens and image tiles the next run will read, counted by the model's own tokenizer and image processor as you edit; a prompt over the context limit is reported there before running.

Displayed run latencies measure synchronous inference time (prompt formatting, tokenization, embeddings, and ANE execution). They exclude one-time model initialization and UI IPC overhead, but the session median and p95 include first-use warm-up of each context tier. A run covers every question, so a sequential run adds up its passes. Throughput divides the last run's rendered input tokens by its latency; a shared pass reads the input once. Switching modes, question count or a model setting resets the median and p95.

## Shortcuts

Build and launch **Tinkerbox.app** once, and let model setup finish. In Shortcuts, create a shortcut and search for **Tinkerbox** in the action library. The app provides six actions:

| Action | Inputs | Result |
| --- | --- | --- |
| **Yes/No Decision** | Input Text, Question, Images, Yes Threshold (default 0.5) | Reads as `Yes (93%)`; **Question**, **Answer** (Boolean), **Yes Probability** (0–1) |
| **Choose Option** | Input Text, Question, Options (List of at least two unique names), Images, Minimum Confidence (default 0) | Reads as `Refund request (81%)`; **Chosen Option** (empty below Minimum Confidence, then reads `Unsure: …`), **Most Likely Option**, **Confidence** (0–1) |
| **Score Text** | Input Text, Question, Levels (List of 2–10 labels, lowest to highest), Images | Reads as `Friendly (1.8 / 2)`; **Score** (fractional expected value from 0 to N−1), **Maximum Score**, **Most Likely Level** |
| **Ask Several Questions** | Input Text, Questions (List), Images, Yes Threshold | A list of Yes/No Decisions, one per question in order, from a single pass over the input |
| **Filter List by Question** | List, Question, Yes Threshold | The items answered Yes, in their original order |
| **Sort List by Score** | List, Question, Levels, Highest First (default on) | Ranked items that read as the original text, with **Score**, **Maximum Score**, and **Most Likely Level** |

Each result reads as text when used in a Text action, notification, or **Show Result**, and keeps typed properties for logic: select a property from the action's result variable, or use **Get Details of**. The percentage in a Yes/No result is the probability of the answer it gives, so `No (88%)` means a 12% Yes Probability.

**Images** is optional and accepts any image files, such as the output of **Take Screenshot** or **Select Photos**; **Input Text** may be left empty when images are supplied. Actions use the model selected in the app, and report an error if it does not accept images.

**Ask Several Questions** reads the input once and answers every question in one pass, so extra questions cost little. Each answer is independent, so several can be Yes; use it to tag an input with every label that applies (Choose Option picks exactly one). The list actions evaluate all items together in one batch and drop blank items, so they accept the output of **Split Text** directly.

Single-input results also include **Probabilities** (a list of Label/Probability values), **Input Tokens**, and **Inference Time (ms)**. Probabilities are normalized across the supplied options or levels; they are not guarantees of correctness. Score probability labels include the zero-based level number and its label.

For a refund-routing prototype:

1. Add **Text** with `I was charged twice. Please refund the duplicate payment.`
2. Add **Yes/No Decision**, use the Text output as **Input Text**, and set **Question** to `Is this a refund request?`
3. Add **If**, select the decision result's **Answer** property, and branch on whether it is true. Raise **Yes Threshold** to act only on confident answers.

For classification, feed a **List** such as `Refund request`, `Technical support`, `Other` into **Options**. For scoring, feed an ordered **List** such as `Not urgent`, `Somewhat urgent`, `Very urgent` into **Levels**.

Actions run in the background, reuse the UI's loaded model, and queue overlapping calls. Closing the window keeps the app available for automations; use **Quit Tinkerbox** to exit. A cold action can initialize the runtime, but completing first-run downloads in the app avoids Shortcuts' execution time limits. Setup requires network access; inference runs locally. Blank inputs, invalid options/levels, and model context-limit errors are reported to Shortcuts.

The source build runs Xcode's App Intents metadata extractor before signing. Keep `Contents/Resources/Metadata.appintents` in the app bundle so the actions remain discoverable. Every Tinkerbox build shares one bundle identifier, so keep a single copy installed: Shortcuts may resolve the actions to an older copy (1.x has none) or to an ad-hoc signed build, which fails with "Shortcuts couldn't communicate with the app".

To exercise the actions against real inference, including concurrent calls with the UI runtime and recovery after errors:

```sh
python3 examples/swiftui/run.py --build-only
python3 examples/swiftui/test_shortcuts.py
```

The smoke test uses the built app's bundled worker and pinned package requirements. It requires the same supported Apple Silicon hardware and cached/downloadable model as the app. Set `TINKERBOX_ASSET` to use a local compatible model bundle. It calls the App Intent entry points directly; discovery and composing result properties should also be checked in the Shortcuts app after a build.
