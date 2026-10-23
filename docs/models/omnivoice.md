# OmniVoice speech synthesis

OmniVoice runs its Qwen3 diffusion transformer and audio projection on ANE.
Text processing, embeddings, scheduling, sampling and the 24 kHz audio codec run
on CPU through `omnivoice==0.2.1`. Bundles include the CPU embeddings and codec.
The checkpoint is `k2-fsa/OmniVoice@c5fdb5ccb189668d56333f77ba2629f4cd7535f4`;
weights are CC-BY-NC-4.0 and upstream code is Apache-2.0.

## Local setup

```sh
uv sync --extra speech
.venv/bin/python -m tinkerbox_ane.install omnivoice /path/to/verified-bundle
```

```python
import soundfile as sf
import tinkerbox_ane

with tinkerbox_ane.load("omnivoice") as model:
    audio = model.generate("Hello! Welcome to Tinkerbox.", language="en", seed=42)[0]
    sf.write("speech.wav", audio, model.sampling_rate)
    voice = model.create_voice_clone_prompt("reference.wav", ref_text="Reference transcript.")
    voice.save("voice.pt")
    cloned = model.generate("A new sentence.", voice_clone_prompt=voice, seed=42)[0]
```

Pass a bundle path as the second argument to `load` to use it directly.
`generate` accepts ordered lists for text, language, instruction, duration and
speed, plus upstream generation settings. Use `instruct` for voice design and
`model.load_voice_clone_prompt("voice.pt")` for saved voices. Omitting a reference
transcript invokes CPU Whisper transcription, which may download its model.
The task API returns a seed for replay of initially unseeded requests.

The app's Text to Speech page provides automatic voice, design, cloning, saved
voices, playback and WAV export. Choose a local bundle in the toolbar or set
`TINKERBOX_ASSET` before launch. Bundles are not distributed yet.

## Export and check

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.omnivoice.export \
  --checkpoint /path/to/checkpoint --out artifacts/omnivoice/source \
  --compile-to artifacts/omnivoice/ane
```

Use fresh output directories. The compiler selects the host architecture and
verifies full ANE placement under the existing watchdog. Capacities default to
96/128/256/512/1024 positions; use `--architecture h16g` for standard M4
cross-compilation. Correctness and performance require execution on the target Mac.

Run public speech regressions with `TINKERBOX_SPEECH_TEST_ASSET` and
`TINKERBOX_OMNIVOICE_CHECKPOINT` set. Set `TINKERBOX_SPEECH_TEST_ASR` to a local
Whisper checkpoint to include transcription checks. Build the current wheel and
app, then run `app/test_speech.py` with `TINKERBOX_ASSET` to check the Swift worker.
Run ANE checks under `python -m tinkerbox_ane.guard`.

## Limits

The combined context limit is 1024 text, reference and generated-audio positions
per call. Upstream chunks long-form generation; oversized references or unchunked
requests return an error. Automatic unreferenced speech gains missing terminal
punctuation and at least three seconds of generation context at normal speed to
avoid short-input noise. Explicit durations bypass that minimum; speed scales it.
Optional English/Chinese normalization dependencies are not packaged in the app.
