# Tinkerbox app

A macOS app for the models in this repository: decisions, text to speech and text completion, plus Shortcuts actions. Requires macOS 27 on Apple Silicon.

## Install

1. Download `Tinkerbox-ANE-*-macos-arm64.zip` from the [latest release](https://github.com/WeGoToMars/tinkerbox/releases/latest).
2. Move **Tinkerbox.app** to Applications and open it. The app is not notarized, so macOS blocks the first launch: allow it under **System Settings → Privacy & Security → Open Anyway**, or run `xattr -dr com.apple.quarantine Tinkerbox.app`.

On first launch the app uses its bundled `uv` to set up a private Python 3.12 environment with the pinned `tinkerbox-ane` wheel. A model downloads the first time you select it. Logs are in `~/Library/Logs/tinkerbox-ane/demo.log`.

## Tasks

**Home** has a card for each task. Every task page has a model picker, **Unload** and **Model Settings** in its toolbar. Settings such as `max_image_tiles` are saved per model and also apply to Shortcuts. **Highlight tokens** colors the input by the loaded model's tokenizer.

A model unloads after five minutes without a request, or right away with **Unload** (after any running request finishes). The next run or Shortcut loads it again. Token previews never load a model or keep it loaded. Closing the window keeps the app running; quitting stops the model.

### Decision

Ask yes/no (`noul`), choice and score questions about text, images or both. Add images by dragging, pasting or **Add**; text is optional when an image is attached. Click an image to see the tiles and resolution the vision model gets. The token counter shows how much of the context a run will use.

With several questions, d1-3B can answer them in one **Parallel** pass or run each one separately (**Sequential**). The d3 models always run them one after another.

### Text to Speech

OmniVoice and VoxCPM2 offer automatic voices, voice design from a description, and cloning from reference audio or a saved voice. VoxCPM2 can also continue a recording. The page shows only the controls the selected model supports (see the [speech API](../docs/speech.md)).

For a reference, choose an audio file or click **Arm microphone**, then hold Space to record (up to 20 seconds). Modes that need a transcript get one from on-device speech recognition; check or edit it before generating. **Save voice…** stores the prepared voice for reuse with the same model and mode.

Generated audio plays in place and exports as WAV or M4A. The seed is shown so you can replay a result. With VoxCPM2, **Play while generating** starts playback as soon as the first audio arrives.

### Text Completion

Continue one or more prompts with GPT-2 (any size, FP16, INT8 or LUT8). Each prompt shows its token count against the 1,024-token context. Generated text streams into the prompt in the accent color, and **Continue** generates more from the full text. Decoding defaults to sampling (temperature 0.8, top-k 40, top-p 0.95); a seed replays each prompt exactly. **Batch size** caps how many prompts run together; **Automatic** picks a size from the prompts. See [GPT-2](../docs/models/gpt2.md).

## Models

**Models** in the sidebar lists every model by task, from the catalog on GitHub (the bundled copy when offline). Pick a size and quantization to see that variant's download size and status. **Select** or **Download & Select** makes it the model for its task. Downloads continue in the background.

Each installed model shows its disk usage, **Show in Finder** and **Remove**. **Remove Outdated** deletes earlier revisions after an update. A model with no bundle for this Mac can use a local one through **Choose bundle…**.

## Storage

**Storage** in the sidebar shows everything Tinkerbox keeps on this Mac, split into model bundles, Core AI's compiled copies, earlier revisions, upstream checkpoints, unused compiled copies and Python. Hover over the bar or a row to highlight that kind in every model's bar. Earlier revisions, upstream checkpoints and unused compiled copies are never loaded, so each can be deleted in one step. The models are ranked by size, and each one's menu shows it in Finder, removes its earlier revisions or deletes it.

## Shortcuts

Search for **Tinkerbox** in the Shortcuts action library. Decision actions use the model selected on the Decision page; speech actions save their own **Model** choice.

| Action | Inputs | Result |
| --- | --- | --- |
| **Yes/No Decision** | Input Text, Question, Images, Yes Threshold (default 0.5) | `Yes (93%)`, with **Answer**, **Yes Probability** and **Question** |
| **Choose Option** | Input Text, Question, Options (at least two), Images, Minimum Confidence | `Refund request (81%)`, with **Chosen Option**, **Most Likely Option** and **Confidence** |
| **Score Text** | Input Text, Question, Levels (2–10, lowest first), Images | `Friendly (1.8 / 2)`, with **Score**, **Maximum Score** and **Most Likely Level** |
| **Ask Several Questions** | Input Text, Questions, Images, Yes Threshold | A Yes/No Decision per question, in order; d1-3B answers them in one pass |
| **Filter List by Question** | List, Question, Yes Threshold | The items answered Yes, in order |
| **Sort List by Score** | List, Question, Levels, Highest First | The items ranked, each with its score |
| **Generate Speech** | Text or a list of texts, Model, Mode, Voice Description, Reference Audio and Transcript or Saved Voice, Language, Duration, Speed, Seed, generation settings | WAV files in input order, named like `speech-42-1.wav` (seed 42, first item) |
| **Create Voice Prompt** | Reference Audio, Model, Reference Transcript, Mode | A `voice.pt` file for **Generate Speech** or the app |

Input Text is optional when you pass images. List actions skip blank lines and run the items as one batch. Leave Mode empty to pick automatic voice, design or cloning from the inputs; after changing Model, choose a Mode again. Controls the model doesn't support return an error rather than being ignored. Actions run in the background and share the app's loaded model; inference runs entirely on this Mac.

If Shortcuts says it "couldn't communicate with the app", another copy of Tinkerbox (or an ad-hoc signed build) has claimed the actions. Keep a single copy installed.

## Development

Building requires Xcode 27 and [uv](https://docs.astral.sh/uv/):

```sh
uv run --no-project --python 3.12 python app/run.py              # build and launch
uv run --no-project --python 3.12 python app/run.py --build-only # artifacts/swiftui/Tinkerbox.app and its ZIP
```

Shortcuts actions need a development signature (`security find-identity -v -p codesigning`); add `--sign "Apple Development: you@example.com"`. Use `--wheel dist/tinkerbox_ane-*.whl` to bundle a local wheel from `uv build --wheel`, and set `TINKERBOX_ASSET=/absolute/path/to/bundle` to run every model from a local bundle.

`app/smoke.py` compiles each smoke in `tests/swiftui` against the app's Swift sources and runs it with the built app's worker and real models:

```sh
uv run --no-project --python 3.12 python app/smoke.py                       # all
uv run --no-project --python 3.12 python app/smoke.py shortcuts completion  # selected
```

| Smoke | Covers |
| --- | --- |
| `shortcuts` | Decision actions, concurrency with the UI, validation, unload and idle reload |
| `completion` | Streaming, ordering, token previews, seeded replay, limits and reload (`TINKERBOX_COMPLETION_TEST_MODEL`, default `gpt2`) |
| `tokens` | Token highlighting for every family (`TINKERBOX_TOKEN_TEST_MODELS`) |
| `speech`, `voxcpm2` | Speech requests, cloning, transcription, replay and limits |
| `speech-shortcuts` | Speech actions (`TINKERBOX_SPEECH_MODEL`, default `omnivoice`) |
| `speech-modes` | Speech pickers before any model is downloaded |

Smokes call the App Intents directly. Check discovery and result properties in the Shortcuts app itself after changing actions.
