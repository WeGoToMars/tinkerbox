# VoxCPM2

[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) generates multilingual speech
with automatic voices, voice design, cloning and continuation. Output is mono
**48 kHz**, with streaming for early playback.

Requires macOS 27 and Apple Silicon M1–M6. In the app, select a VoxCPM2 variant on
Text to Speech; model files download automatically. Python users need Python
3.12 or 3.13 and the speech extra:

```sh
python -m pip install "tinkerbox-ane[speech]==2.2.0"
```

| Model ID | Download | Measured peak wired memory |
| --- | ---: | ---: |
| `voxcpm2` | About 4.2 GB | 3.9 GiB |
| `voxcpm2-lut8` | About 3.1 GB | 2.7 GiB |

LUT8 uses less memory with slightly lower fidelity; it also ran faster on M5 Pro.
Measurements are from M5 Pro; other supported chips are cross-compiled and have
not been validated on-device.

## Generate speech

```python
import soundfile as sf
import tinkerbox_ane

with tinkerbox_ane.load("voxcpm2") as model:
    audio = model.generate("Hello! Welcome to Tinkerbox.", seed=42)[0]
    sf.write("speech.wav", audio, model.sampling_rate)
```

Use `"voxcpm2-lut8"` to select the smaller variant. The app provides playback,
recording, saved voices and WAV/M4A export. **Play while generating** starts
playback before the clip finishes. Python's `model.stream(text, seed=42)` yields
successive audio chunks.

## Voice modes

| Mode | What it does | Reference transcript |
| --- | --- | --- |
| `auto` | Chooses a voice | None |
| `design` | Uses a description passed as `instruct` | None |
| `clone` | Uses a recording's voice | Not needed |
| `continuation` | Continues speech from a recording | Supplied or recognized on-device |
| `clone_continuation` | Clones the voice and continues speech | Supplied or recognized on-device |

For design, try `instruct="A warm, calm voice; speak slowly and clearly."`.
Cloning and continuation modes also accept optional delivery instructions.

```python
import soundfile as sf
import tinkerbox_ane

with tinkerbox_ane.load("voxcpm2") as model:
    voice = model.create_voice_clone_prompt("reference.wav", mode="clone")
    voice.save("voice.pt")
    audio = model.generate(
        "A new sentence in the reference voice.",
        mode="clone", voice_clone_prompt=voice, seed=23,
    )[0]
    sf.write("cloned.wav", audio, model.sampling_rate)
```

Reload a prompt with `model.load_voice_clone_prompt("voice.pt")`. Saved prompts
must match their model and conditioning mode. To prepare a continuation prompt,
use the corresponding mode and pass the recording's words as `ref_text`, or let
on-device recognition supply them.

## Controls and limits

Language is detected from text; omit `language`. Describe speed and delivery in
`instruct`; numeric `speed` and `duration` are unsupported. Generation defaults
to `num_step=10` and `guidance_scale=2`. A seed replays the same request, and
`generate` accepts ordered batches; item `i` uses `seed + i`.

Text, reference audio and generated speech share an **8,192-position** context.
Use short, clear reference recordings. If a request reaches its generation limit
before speech finishes, it returns an error rather than a truncated clip.
First use can take several minutes; subsequent requests reuse the model cache.

Model weights and upstream code use [Apache-2.0](../../LICENSES/VoxCPM-Apache-2.0.txt).
[Full speech API](../speech.md) · [Speech Shortcuts](../../app/README.md#shortcuts) ·
[Developer commands](../model-publishing.md#model-builds).
