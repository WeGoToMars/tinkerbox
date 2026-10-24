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
verifies full ANE placement under the existing watchdog. The default export
contains two assets: a fused short-input transformer with capacities
96/128/256/512/1024, and a blockwise transformer with attention tiers
128/256/512/1024/2048/4096/8192/16384/32768/40960. Use `--architecture h16g`
for standard M4 cross-compilation. Correctness and performance require execution
on the target Mac.

The blockwise path projects all keys and values for a layer before attending
128-query blocks over the complete visible prefix. Large attention tiers use
4,096-key blocks merged by their softmax masses. It composes the same Qwen3
projections, norms and MLP with the full-attention block shared by D3. All
transformer and audio-head computation stays on ANE. Diffusion steps recompute
each layer's keys and values; a causal decode cache cannot be reused between
those steps. Sequences up to 1,024 positions use the fused asset for latency.

The validated full-context source is `artifacts/omnivoice/full-context-v2-export`;
the deployment bundle is `artifacts/omnivoice/full-context-v2-ane`. Its centered
audio head and attention value scale 16 are retained. Both assets passed full
ANE placement on M5 Pro (`h17s`), and only that architecture was compiled for
this extension. The previously staged five-capacity release across 21 targets
predates this change and does not establish full-context support on those chips.
New release staging must use this export and separately validate the targets:

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.omnivoice.publish \
  --source artifacts/omnivoice/full-context-v2-export --arch h17s --no-upload
```

The catalog remains local-only until publishing is explicitly requested.

Run public speech regressions with `TINKERBOX_SPEECH_TEST_ASSET` and
`TINKERBOX_OMNIVOICE_CHECKPOINT` set. Transcription checks use Apple's system
speech assets. Build the current wheel and
app, then run `app/test_speech.py` with `TINKERBOX_ASSET` to check the Swift worker.
Run ANE checks under `python -m tinkerbox_ane.guard`.

## Full-context validation

Capture upstream `omnivoice==0.2.1` CPU FP16 logits independently, then compare
the ANE output and the 40,960-position attention kernel. The optional speech
case exercises a complete 28-second request beyond the old 1,024-position cap:

```sh
.venv/bin/python scripts/omnivoice/reference_logits.py \
  --checkpoint artifacts/omnivoice/checkpoint

PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.guard artifacts/omnivoice/context-run -- \
  .venv/bin/python -u scripts/omnivoice/validate_context.py \
  --asset artifacts/omnivoice/full-context-v2-ane --speech
```

On Apple M5 Pro, 24 GB, macOS 27.0.1 (`h17s`), independent full-transformer
comparisons at lengths 96 and 1,100 had cosine at least 0.999986 and relative
L2 at most 0.005131. The full 40,960-key attention kernel passed both fully
visible and mostly masked cases with cosine at least 0.999998 and relative L2
at most 0.001561. The 28-second passage finished with zero normalized
transcription error in roughly 14 seconds including transcription. These are
FP16 arithmetic/content checks, not bitwise waveform equivalence or a full
40,960-position transformer performance benchmark. The watchdog passed.
Full identities and app checks are in
local validation reports under `artifacts/omnivoice/`.

## Limits

The combined context limit is now the checkpoint's 40,960 text, reference and
generated-audio positions per diffusion call. Upstream still chunks long-form
generation above its default thirty-second threshold into roughly fifteen-second
text chunks. Oversized conditioning is rejected before allocating upstream's
square batch mask. The upstream scheduler still creates a CPU mask proportional
to the square of its active batch length, and returned logits also grow with
context: maximum context is not negligible total RAM or a constant-latency
operation. Native blockwise attention avoids a full square ANE attention graph.

Automatic unreferenced speech gains missing terminal punctuation and at least
three seconds of generation context at normal speed to avoid short-input noise.
Explicit durations bypass that minimum; speed scales it. Pynini-based number/date
normalization is not exposed. Write numbers as spoken words when pronunciation
needs to be explicit.
