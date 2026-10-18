# SwiftUI Demo

A standalone macOS application demonstrating `tinkerbox-ane` on Apple Silicon.

Requires macOS 27 on Apple Silicon.

## Prebuilt App

1. Download and extract the latest `Tinkerbox-ANE-*-macos-arm64.zip` from the GitHub releases page.
2. Launch **Tinkerbox.app**. The app is ad-hoc signed, so macOS blocks the first launch: allow it under **System Settings → Privacy & Security → Open Anyway**, or run `xattr -dr com.apple.quarantine Tinkerbox.app`.

The app bundles a standalone `uv` binary. On first run, it configures an isolated Python 3.12 environment, installs the precompiled `tinkerbox-ane` wheel, and fetches the required model weights from Hugging Face. Subsequent launches reuse the cached environment.

The default model bundle targets `h17s` (M5-family) Neural Engine cores. Setup progress and error logs are written to `~/Library/Logs/tinkerbox-ane/demo.log`.

## Building from Source

Building requires Xcode 27 and [uv](https://docs.astral.sh/uv/):

```sh
uv run --no-project --python 3.12 python examples/swiftui/run.py --build-only
```

This generates `artifacts/swiftui/Tinkerbox.app` and a distributable ZIP archive. Omit `--build-only` to launch the application immediately after building.

To test against a local model bundle instead of downloading from Hugging Face:

```sh
TINKERBOX_ASSET="$PWD/artifacts/d1_fp16_fused128_ctx32768" uv run --no-project --python 3.12 python examples/swiftui/run.py
```

## Supported Tasks

- `noul`: Binary decisions with normalized probabilities.
- `choice`: Selection across two or more discrete options.
- `score`: Ranked evaluation across 2–10 ordered levels.

Displayed decision latencies measure synchronous inference time (prompt formatting, tokenization, embeddings, and ANE execution). They exclude one-time model initialization and UI IPC overhead, but the session median and p95 include first-use warm-up of each context tier.

## Shortcuts

Build and launch **Tinkerbox.app** once, and let model setup finish. In Shortcuts, create a shortcut and search for **Tinkerbox** in the action library. The app provides three actions:

| Action | Inputs | Result properties |
| --- | --- | --- |
| **Yes/No Decision** | Input Text, Question | **Answer** (Boolean; Yes when its probability is at least 0.5), **Yes Probability** (0–1) |
| **Choose Option** | Input Text, Question, Options (List of at least two unique names) | **Chosen Option** (text), **Confidence** (0–1) |
| **Score Text** | Input Text, Question, Levels (List of 2–10 labels, lowest to highest) | **Score** (fractional expected value from 0 to N−1), **Maximum Score** |

Every result also includes **Probabilities** (a list of Label/Probability values), **Input Tokens**, and **Inference Time (ms)**. Select a property from the action's result variable, or use **Get Details of** on the result. Probabilities are normalized across the supplied options or levels; they are not guarantees of correctness. Score probability labels include the zero-based level number and its label.

For a refund-routing prototype:

1. Add **Text** with `I was charged twice. Please refund the duplicate payment.`
2. Add **Yes/No Decision**, use the Text output as **Input Text**, and set **Question** to `Is this a refund request?`
3. Add **If**, select the decision result's **Answer** property, and branch on whether it is true. To use your own threshold, select **Yes Probability** instead.

For classification, feed a **List** such as `Refund request`, `Technical support`, `Other` into **Options**. For scoring, feed an ordered **List** such as `Not urgent`, `Somewhat urgent`, `Very urgent` into **Levels**.

Actions run in the background, reuse the UI's loaded model, and queue overlapping calls. Closing the window keeps the app available for automations; use **Quit Tinkerbox** to exit. A cold action can initialize the runtime, but completing first-run downloads in the app avoids Shortcuts' execution time limits. Setup requires network access; inference runs locally. Blank inputs, invalid options/levels, and model context-limit errors are reported to Shortcuts.

The source build runs Xcode's App Intents metadata extractor before signing. Keep `Contents/Resources/Metadata.appintents` in the app bundle so the actions remain discoverable.

To exercise all three actions against real inference, including concurrent calls with the UI runtime and recovery after errors:

```sh
python3 examples/swiftui/run.py --build-only
python3 examples/swiftui/test_shortcuts.py
```

The smoke test uses the built app's bundled worker and pinned package requirements. It requires the same supported Apple Silicon hardware and cached/downloadable model as the app. Set `TINKERBOX_ASSET` to use a local compatible model bundle. It calls the App Intent entry points directly; discovery and composing result properties should also be checked in the Shortcuts app after a build.
