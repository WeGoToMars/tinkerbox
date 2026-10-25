# Shared speech API

TTS models share text input, language selection, instructions, reference audio,
saved voices, ordered audio results and seed replay. Each backend advertises its
native synthesis modes and controls through `model.capabilities`; the app renders
that metadata without branching on model names or architecture types.

```python
import tinkerbox_ane

with tinkerbox_ane.load("omnivoice") as model:
    print(model.capabilities)
    audio = model.generate(
        "A familiar voice can tell a new story.", language="en",
        mode="design", instruct="A warm, calm voice.",
        controls={"speed": 1.25}, seed=42,
    )[0]
```

`capabilities["modes"]` describes each mode's ID, title, instruction input,
reference-voice support, language support and mode-specific controls.
`capabilities["controls"]` describes controls shared by that model's modes.
`capabilities["parameters"]` describes generation settings such as diffusion/flow
steps and sampling temperature. Fields may be numeric, boolean, text or enum;
enum values keep their backend meaning, with separate human-readable labels.
An optional field, such as a target duration, is omitted until explicitly enabled.

`generate` accepts a text string or an ordered list. `language`, `instruct`,
`mode`, `controls` and `voice_clone_prompt` can be single values or matching
lists. Omitted mode selects automatic, design or cloning from the supplied
instruction/reference. Explicit mode selects exactly the advertised behavior.
Unsupported controls and incompatible combinations return errors.

A model whose capabilities include `streaming: True` also offers `stream`, which
takes one text and the same arguments and yields its waveform in order while it
is generated. The pieces join into exactly what `generate` returns for the same
request and seed. An error after the first piece, such as running out of
context, ends the stream by raising it.

```python
with tinkerbox_ane.load("voxcpm2") as model:
    for piece in model.stream("Hello! Welcome to Tinkerbox.", seed=42):
        play(piece)  # float32 at model.sampling_rate
```

| Capability | OmniVoice | VoxCPM2 |
| --- | --- | --- |
| Modes | Automatic, voice design, cloning | Automatic, design, cloning, continuation, combined cloning/continuation |
| Duration | Native numeric target, in seconds | Unsupported |
| Speed | Numeric factor used by the duration estimator | Descriptive instruction |
| Instructions | Supported comma-separated voice attributes | Natural-language voice or delivery description |
| Reference transcript | Used for reference cloning; transcribed if absent | Not used for reference-only cloning; required for continuation |
| Language | Explicit selection or automatic | Detected from text |
| Audio | 24 kHz | 48 kHz |
| Streaming | Unsupported: every frame is refined together until the last step | One text, in 0.64-second pieces |

For example, OmniVoice accepts `controls={"duration": 3.0}` or
`controls={"speed": 1.25}`. Its existing direct `duration` and `speed` arguments
remain native API options. Other backends can advertise different fields and
categorical controls. The shared task does not approximate unsupported controls
through audio postprocessing.

Each reference-capable mode may advertise `reference_transcript="none"` or
`"required"`. The app hides transcription controls for `"none"`; an omitted
field preserves transcript-based preparation. `create_voice` accepts `mode` so
the backend can prepare the corresponding representation. Saved voices may be
specific to a conditioning mode as well as a model; incompatible combinations
return an error. Audio preparation and any missing-transcript recognition are
owned by the backend, so Python, the worker, and app automation share behavior.

The worker protocol carries per-item intent separately from request-wide
generation settings:

```json
{
  "items": [
    {"text": "Hello.", "language": "en", "mode": "auto", "controls": {"speed": 1.25}},
    {"text": "Welcome.", "mode": "design", "instruct": "A warm, calm voice."}
  ],
  "parameters": {"num_step": 8},
  "seed": 42
}
```

Replies preserve item order and contain PCM WAV data, sample rate, duration,
request seed and timing. With `"stream": true`, a request for one text to a
streaming model first sends each piece as `{"chunk": {"pcm": ..., "sampleRate": ...}}`
(base64 mono little-endian PCM16; joined in order, the WAV's samples), then the
complete reply, which adds `firstAudioMs`. Clone preparation is timed separately. Saved voice
files belong to their originating model. Pynini normalization is not part of
the shared task API or app.
