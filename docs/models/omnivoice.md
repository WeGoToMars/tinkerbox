# OmniVoice

[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) generates multilingual speech
with automatic voices, voice design and voice cloning. Output is mono **24 kHz**.

Requires macOS 27 and Apple M4 or newer. The model download is about 3.4 GB.
In the app, select **OmniVoice** on Text to Speech; model files download
automatically. Python users need Python 3.12 or 3.13 and the speech extra:

```sh
python -m pip install "tinkerbox-ane[speech]==2.2.0"
```

## Generate speech

```python
import soundfile as sf
import tinkerbox_ane

with tinkerbox_ane.load("omnivoice") as model:
    audio = model.generate("Hello! Welcome to Tinkerbox.", language="en", seed=42)[0]
    sf.write("speech.wav", audio, model.sampling_rate)
```

The app provides playback, microphone recording, saved voices and WAV/M4A export.
[Speech Shortcuts](../../app/README.md#shortcuts) can generate audio and prepare
reusable voices.

## Voices and controls

| Mode | Inputs |
| --- | --- |
| `auto` | Text; the model chooses a voice |
| `design` | Text and `instruct`, such as `"female, low pitch, british accent"` |
| `clone` | Text and reference audio or a saved voice prompt |

Voice design uses supported comma-separated attributes, such as `female`,
`low pitch` and `british accent`.

```python
import soundfile as sf
import tinkerbox_ane

with tinkerbox_ane.load("omnivoice") as model:
    voice = model.create_voice_clone_prompt(
        "reference.m4a", ref_text="The words spoken in the recording."
    )
    voice.save("voice.pt")
    audio = model.generate("A new sentence.", voice_clone_prompt=voice, seed=42)[0]
    sf.write("cloned.wav", audio, model.sampling_rate)
```

Reload a saved voice with `model.load_voice_clone_prompt("voice.pt")`. If you omit
`ref_text`, on-device speech recognition supplies the transcript when available;
you can also provide it manually.

`duration` sets a target length in seconds. `speed` adjusts the duration estimate
(above 1 is faster); an explicit duration takes precedence. `num_step` controls
generation steps, with a default of 32. A seed replays the same request;
`generate` also accepts ordered lists of texts.

## Limits

OmniVoice returns completed audio and does not stream. Long text is split into
chunks automatically. Text, reference audio and output share a **40,960-position**
context per chunk; long requests use more memory and take longer.

Very short text uses at least three seconds of generation context unless you
set an explicit duration. Write numbers as spoken words when exact pronunciation
matters. Saved voices belong to the model that created them.

Model weights use **CC-BY-NC-4.0**. The codec has separate license terms;
see the [bundle's license and notices](https://huggingface.co/wegotomars/OmniVoice-CoreAI-ANE/tree/v2.2.0).
[Full speech API](../speech.md) · [Developer commands](../model-publishing.md#model-builds).
