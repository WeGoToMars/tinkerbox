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

| Capability | OmniVoice |
| --- | --- |
| Modes | Automatic, voice design, cloning |
| Duration | Native numeric target, in seconds |
| Speed | Numeric factor used by the duration estimator |
| Instructions | Supported comma-separated voice attributes |
| Audio | 24 kHz |

For example, OmniVoice accepts `controls={"duration": 3.0}` or
`controls={"speed": 1.25}`. Its existing direct `duration` and `speed` arguments
remain native API options. Other backends can advertise different fields and
categorical controls. The shared task does not approximate unsupported controls
through audio postprocessing.

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
request seed and timing. Clone preparation is timed separately. Saved voice
files belong to their originating model. Pynini normalization is not part of
the shared task API or app.
