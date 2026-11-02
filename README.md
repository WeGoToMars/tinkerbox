<img width="1920" height="388" alt="banner" src="https://github.com/user-attachments/assets/52cfb1e7-c841-4898-9a94-d199db85f2d7" />

# Tinkerbox for Apple Neural Engine

Run models on Apple's Neural Engine through Core AI. Inference runs on the Neural Engine, the CPU handles tokenization and other light work, and the GPU stays idle.

| Model | Task | Chips | Notes |
| --- | --- | --- | --- |
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (`d1-3b`) | Decisions over text and images, several questions per pass | M4–M6 | [d1.md](docs/models/d1.md) |
| [vllm-sr/d3-nano](https://huggingface.co/vllm-sr/d3-nano), [d3-mini](https://huggingface.co/vllm-sr/d3-mini) (`d3-nano`, `d3-mini`) | Decisions over text and images | M4–M6 | [d3.md](docs/models/d3.md) |
| [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (`omnivoice`) | Speech, voice design and cloning | M4–M6 | [omnivoice.md](docs/models/omnivoice.md) |
| [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (`voxcpm2`, `voxcpm2-lut8`) | Speech, voice design, cloning and continuation | M1–M6 | [voxcpm2.md](docs/models/voxcpm2.md) |
| [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) | Text completion | M1–M6 | [gpt2.md](docs/models/gpt2.md) |

Requires macOS 27 on Apple Silicon.

## App

Download `Tinkerbox-ANE-*-macos-arm64.zip` from the [latest release](https://github.com/WeGoToMars/tinkerbox/releases/latest), move **Tinkerbox.app** to Applications and open it. It sets up its own Python environment and downloads each model the first time you select it. Its tasks are also available as Shortcuts actions. See the [app guide](app/README.md).

## Python

Python 3.12 or 3.13:

```sh
python -m pip install tinkerbox-ane            # add [speech] for OmniVoice and VoxCPM2
```

```python
import tinkerbox_ane

with tinkerbox_ane.load("d1-3b") as model:
    result = model.system_one(
        "I was charged twice this month.",
        {"refund": {"type": "noul", "instructions": "Is this a refund request?"}},
    )
    print(result["answers"])
```

`tinkerbox_ane.load(model_id)` downloads the Core AI bundle for this Mac to `~/Library/Caches/tinkerbox-ane/models/` and returns the API for the model's task. Pass a directory as the second argument to load a local bundle instead. Bundles include the tokenizer, configuration and the few weights that run on the CPU, so loading never downloads the upstream checkpoint.

- **Decisions** (d1, d3): `system_one`, `system_one_batch`, `probabilities`, `probabilities_batch` and `count_tokens` answer yes/no (`noul`), `choice` and `score` questions. d1-3B answers several questions about one input in a shared pass; d3 runs them one after another.
- **Speech** (OmniVoice, VoxCPM2): `generate`, reusable voice prompts and on-device transcription. VoxCPM2 also streams. See the [speech API](docs/speech.md).
- **Text completion** (GPT-2): OpenAI-style `complete(prompt=..., max_tokens=...)` with streaming, sampling and batching.

`python -m tinkerbox_ane.storage` lists what each model keeps on disk. `--remove MODEL` deletes a model's files (with `--outdated`, only revisions the catalog no longer loads; `--outdated` alone does that for every model). `--remove-checkpoint REPO...` deletes upstream checkpoints, and `--remove-unused` deletes compiled copies that no bundle uses.

## Development

Building from source, exporting and publishing models, and running tests are covered in the [developer guide](docs/model-publishing.md). Releases follow [Releasing](docs/releasing.md).
