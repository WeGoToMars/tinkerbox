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
contains two assets: a fused transformer with 18 capacities from 128 to 4,096
positions, and a blockwise transformer with attention tiers
128/256/512/1024/2048/4096/8192/16384/32768/40960. Use `--architecture h16g`
for standard M4 cross-compilation. Correctness and performance require execution
on the target Mac.

The blockwise path projects all keys and values for a layer before attending
128-query blocks over the complete visible prefix. Large attention tiers use
4,096-key blocks merged by their softmax masses. It composes the same Qwen3
projections, norms and MLP with the full-attention block shared by D3. All
transformer and audio-head computation stays on ANE. Diffusion steps recompute
each layer's keys and values; a causal decode cache cannot be reused between
those steps. Sequences up to 4,096 positions use the fused asset for latency.

The validated full-context source is `artifacts/omnivoice/perf-src`; the
deployment bundle is `artifacts/omnivoice/perf-ane`. Its centered
audio head and attention value scale 16 are retained. Both assets were
cross-compiled on M5 Pro for all 13 M3-and-newer targets (`h15*`, `h16*`,
`h17*`, `h18g`) and passed full ANE placement verification on each; numerical,
speech and performance results above were measured only on `h17s`. Clean
sequential compile times ranged from 625 s (M4, M4 Pro) to 1,780 s (14-core
M3 Max), dominated by the fused asset.

M1 and M2 are not supported. Their ANE compiler rejects the long-context
merge's rescaling, `exp(clamp(m_i - m_j, max=0))`, in every attention tier
above 4,096 positions; the identical `softmax(block maxima) * block mass`
form compiled on M1 in isolated probes, but has not been exported or
validated. M1 also failed fused buckets of 2,048 and 2,560 positions while
1,024 compiled. Stage the release with:

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.omnivoice.publish \
  --source artifacts/omnivoice/perf-src --no-upload --continue-on-error
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
case generates a 60-second passage as a single chunk:

```sh
.venv/bin/python scripts/omnivoice/reference_logits.py \
  --checkpoint artifacts/omnivoice/checkpoint --lengths 96 1100 2400 3000

PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.guard artifacts/omnivoice/context-run -- \
  .venv/bin/python -u scripts/omnivoice/validate_context.py \
  --asset artifacts/omnivoice/perf-ane --speech
```

On Apple M5 Pro, 24 GB, macOS 27.0.1 (`h17s`), independent full-transformer
comparisons at lengths 96, 1,100, 2,400 and 3,000 had cosine at least 0.999985
and relative L2 at most 0.00541, identical to the previous v2.2.0 bundle. The
full 40,960-key attention kernel passed fully visible and mostly masked cases
with cosine at least 0.999998 and relative L2 at most 0.00156.

The speech case generates the 60-second passage for seeds 1, 2, 3, 5 and 42 and
gates their mean normalized transcription error at 0.20. A single seed is not a
usable gate here: upstream CPU FP16 itself exceeded 0.20 on two of these five
seeds. The engine seeds upstream's generator the same way, so each seed starts
from the same random stream:

| Seed | 1 | 2 | 3 | 5 | 42 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Upstream CPU FP16 | 0.246 | 0.148 | 0.236 | 0.180 | 0.157 | 0.193 |
| This bundle | 0.238 | 0.175 | 0.163 | 0.131 | 0.226 | 0.187 |

The bundle passed with mean 0.187. Each clip took about 23 seconds including
transcription, versus 70.7 seconds for seed 42 with v2.2.0 (same transcript) and
about 21 minutes for upstream on CPU. Capture the upstream clips one seed per
watchdog run:

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.guard artifacts/omnivoice/speech-reference-42 -- \
  .venv/bin/python -u scripts/omnivoice/reference_speech.py --seed 42
```

These are FP16 arithmetic/content checks, not bitwise waveform equivalence.
The commands above write local validation reports under `artifacts/omnivoice/`.
Generated validation JSON is not committed.

## Fused execution

Upstream evaluates each guidance row padded to the batch length, but reads only
the row's visible prefix. The runtime packs consecutive rows into one fused call
with block-diagonal attention and per-row positions, which is exact. A row is
added to a packed call when the bundle's `fused_cost` model predicts it cheaper
than a separate call. Latency is modeled as `call + n + n**2/attention` positions
of linear work, fitted on Apple M5 Pro (`h17s`) to within about 15% across all
buckets. Rows longer than 4,096 positions use the blockwise transformer.

Fused attention splits queries into 256-row blocks from 1,024 to 2,048 positions
and 512-row blocks above that. The results are identical to one attention over
all queries; 256-row blocks above 2,048 positions failed ANE compilation.
Steady-state fused latency on `h17s`, including Python/native transfers:

| Positions | 128 | 256 | 512 | 1,024 | 2,048 | 3,072 | 4,096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Milliseconds | 10.1 | 21.9 | 54.8 | 135 | 377 | 972 | 1,575 |

Unblocked attention took 170 ms at 1,024 and 504 ms at 2,048. The blockwise
runner took 1.27 s at 2,048 and 2.9 s at 4,096 positions, so fusion now extends
to 4,096. Fused 6,144 and 8,192 buckets failed ANE compilation. Above 4,096 the
blockwise runner is ANE-bound: native copies are about 6% of a 12.9-second
8,192-position forward, and 512-query calls were no faster than 128. Per-channel
INT8 weights did not change fused latency and are not used.

Warm end-to-end requests on `h17s`, compared with v2.2.0:

| Request | Audio | v2.2.0 | This bundle |
| --- | --- | --- | --- |
| Automatic voice | 4.4 s | 1.66 s | 0.95 s |
| Saved clone | 2.9 s | 1.62 s | 0.89 s |
| Two-text batch | 5.5 s | 1.71 s | 1.63 s |
| Passage, default chunking | 18.5 s | 12.2 s | 5.1 s |

The two-text batch's four short rows were already near their best buckets. The
fused-versus-blockwise comparison can be reproduced on a target Mac with:

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.guard artifacts/omnivoice/paths-run -- \
  .venv/bin/python scripts/omnivoice/benchmark_paths.py \
  --asset artifacts/omnivoice/perf-ane --repeats 3 --out artifacts/omnivoice/paths.json
```

These results do not establish the best buckets or cost model for other chips.

## Limits

The combined context limit is now the checkpoint's 40,960 text, reference and
generated-audio positions per diffusion call. Upstream still chunks long-form
generation above its default thirty-second threshold into roughly fifteen-second
text chunks. Oversized target lengths are rejected before target allocation;
combined conditioning is checked before the batch mask and embeddings are allocated. The upstream scheduler still creates a CPU mask proportional
to the square of its active batch length, and returned logits also grow with
context: maximum context is not negligible total RAM or a constant-latency
operation. Native blockwise attention avoids a full square ANE attention graph.

Automatic unreferenced speech gains missing terminal punctuation and at least
three seconds of generation context at normal speed to avoid short-input noise.
Explicit durations bypass that minimum; speed scales it. Pynini-based number/date
normalization is not exposed. Write numbers as spoken words when pronunciation
needs to be explicit.
