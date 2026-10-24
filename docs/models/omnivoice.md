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
`model.load_voice_clone_prompt("voice.pt")` for saved voices. Reference files,
including M4A and WAV, are decoded with Apple's AVFoundation codecs. Omitting a reference
transcript invokes Apple's on-device SpeechTranscriber through the native host.
The system installs language assets as needed. If transcription is unavailable,
provide `ref_text` manually. There is no alternative transcription backend.
Transcription defaults to the system language. For another language, call
`model.transcribe(audio, locale="en-US")` with its locale and pass the result as `ref_text`.
The task API returns a seed for replay of initially unseeded requests.

The app's Text to Speech page provides automatic voice, design, cloning, saved
voices, waveform previews, microphone recording and WAV/M4A export. Clone preparation
and speech generation times are reported separately. Choose a local bundle in the toolbar or set
`TINKERBOX_ASSET` before launch. Bundles are not distributed yet.
The **Non-verbal tags** list reads `non_verbal_tags` from the shared `models.json`
manifest, following the [upstream supported tags](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2#non-verbal--pronunciation-control).

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

The selected release export is `artifacts/omnivoice/export`: centered audio
projection, attention value scale 16, source hash
`493992810aa448c799f42f61b90ea00f488ebcbbc73a25ccdff79cda9a36fc0d`.
Stage all supported architectures without publishing:

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.omnivoice.publish --no-upload
```

Release files are staged in `artifacts/huggingface/OmniVoice-CoreAI-ANE`, including
the model card, notices, licenses and `config.json` architecture index. Metadata
must already be present there, or supplied with `--metadata`. Publishing uses
`python -m tinkerbox_ane.omnivoice.publish --upload-only` and creates
`wegotomars/OmniVoice-CoreAI-ANE@v2.2.0`. The catalog remains `local_only` until
that release is published; remove that flag when enabling downloads.
The staged release covers 21 Core AI targets across M1–M6. All five capacities
pass full ANE placement checks on each target. Only M5 Pro has been checked
on-device; earlier generations and other variants are cross-compiled. Max/Ultra
variants and the two M3 Max CPU configurations are listed separately in the
shared catalog, following the [SoC architecture reference](https://github.com/freedomtan/coreml_to_ane_hwx/blob/main/hwx_dump/README.md#soc-architecture-reference-table).

Run public speech regressions with `TINKERBOX_SPEECH_TEST_ASSET` and
`TINKERBOX_OMNIVOICE_CHECKPOINT` set. Transcription checks use Apple's system
speech assets. Build the current wheel and
app, then run `app/test_speech.py` with `TINKERBOX_ASSET` to check the Swift worker.
Run ANE checks under `python -m tinkerbox_ane.guard`.

## Limits

The combined context limit is 1024 text, reference and generated-audio positions
per call. Upstream chunks long-form generation; oversized references or unchunked
requests return an error. Automatic unreferenced speech gains missing terminal
punctuation and at least three seconds of generation context at normal speed to
avoid short-input noise. Explicit durations bypass that minimum; speed scales it.
Pynini-based number/date normalization is not exposed by the app. Write numbers
as spoken words when pronunciation needs to be explicit.
