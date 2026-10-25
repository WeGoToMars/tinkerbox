# VoxCPM2

VoxCPM2 uses the shared speech API and Text to Speech page alongside OmniVoice.
The main MiniCPM language model, residual language model, local audio encoder,
and local diffusion transformer run on ANE. Tokenization, small projections,
scalar quantization, diffusion integration, and AudioVAE V2 run on CPU. The
production runtime does not use the GPU.

Weights are pinned to
[`openbmb/VoxCPM2@32279effe8c19989596f05d353d1447f51d9e915`](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915).
The upstream implementation reference is
[`OpenBMB/VoxCPM@cce58a5b59303c9bd63d12f23afe6a49a4a80c59`](https://github.com/OpenBMB/VoxCPM/tree/cce58a5b59303c9bd63d12f23afe6a49a4a80c59).
Code and weights are Apache-2.0 licensed; see `LICENSES/VoxCPM-Apache-2.0.txt`.

## API and app

```sh
uv sync --extra speech
.venv/bin/python -m tinkerbox_ane.install voxcpm2 /path/to/verified-bundle
```

```python
import soundfile as sf
import tinkerbox_ane

with tinkerbox_ane.load('voxcpm2') as model:
    waveform = model.generate('Hello! Welcome to Tinkerbox.', seed=42)[0]
    sf.write('hello.wav', waveform, model.sampling_rate)
    designed = model.generate('A familiar voice can tell a new story.',
        mode='design', instruct='A warm, calm female voice', seed=7)[0]
    voice = model.create_voice_clone_prompt('reference.wav', mode='clone')
    voice.save('voice.pt')
    cloned = model.generate('This is a short voice cloning demonstration.',
        mode='clone', voice_clone_prompt=voice, seed=23)[0]
    continuation = model.create_voice_clone_prompt('reference.wav',
        ref_text='The exact words spoken in the reference.', mode='continuation')
    continued = model.generate('The next sentence follows naturally.',
        mode='continuation', voice_clone_prompt=continuation, seed=23)[0]
    for piece in model.stream('Playback can begin with the first words.', seed=42):
        ...  # float32 pieces that join into generate's waveform
```

Output is mono 48 kHz. Reference files are decoded using Apple's audio codecs,
mixed to mono, and resampled to 16 kHz. Language is detected from text; omit the
`language` argument. Upstream's Chinese character tokenization is retained.

| Mode | Reference | Transcript | Instruction |
| --- | --- | --- | --- |
| `auto` | None | None | None |
| `design` | Optional | Not used for reference-only conditioning | Required voice description |
| `clone` | Required | Not used | Optional delivery/style description |
| `continuation` | Required | Required; transcribed if absent | Optional delivery/style description |
| `clone_continuation` | Required | Required; transcribed if absent | Optional delivery/style description |

Cloning places the reference in upstream's isolated audio prefix. Continuation
uses the transcript and acoustic prefix. Combined mode uses the same recording
for both native inputs, with their respective padding conventions. Prepared
voices retain this distinction, and loading a saved voice in an incompatible
mode returns an error. Omitted mode with a saved voice uses its prepared mode.
Instructions are translated to upstream's parenthesized prefix in the backend.
The app reads capabilities and never constructs model-specific prompt syntax.

`generate` accepts ordered batches and per-item mode, instruction, controls, and
voice prompts. Sampling uses a local CPU random generator; explicit seeds replay
without advancing another request's sampling stream. Numeric duration/speed
controls are rejected. No waveform time stretching is performed.

Native generation settings are `num_step` (10), `guidance_scale` (2),
`min_decode_steps` (2), `max_decode_steps` (4096), and `preprocess_prompt` (false).
Patch limits count 160 ms audio patches. The compiled context also bounds
generation. A limit reached before the stop predictor finishes returns an error
instead of silently returning a budget-truncated clip. Automatic bad-case
retries, external text normalization, ZipEnhancer denoising and LoRA are not
exposed. Upstream's internal CFG-Zero* sampler is retained.

## Export and compile

```sh
.venv/bin/python -m tinkerbox_ane.voxcpm2.export \
  --checkpoint artifacts/voxcpm2/checkpoint \
  --out artifacts/voxcpm2/fresh-export \
  --compile-to artifacts/voxcpm2/fresh-ane
```

Use empty export directories. Omit `--checkpoint` to download the pinned files,
or set `TINKERBOX_VOXCPM2_CHECKPOINT`. Bundles carry their CPU lookup tables,
tokenizer/configuration files, and AudioVAE weights, so installed loading is
offline. The existing compiler/watchdog retains full ANE-placement verification.

The 28-layer main model uses two 14-layer assets (`--base-group` changes the
split). The eight-layer residual model has its own asset. Both expose
128-position prefill and single-position decoding at cache tiers
512/1024/2048/4096/8192. Swift owns the KV buffers, grows them only when needed,
and copies valid rows across tiers. Each utterance resets its cache. Hidden
rows and RoPE inputs cross the Python/native pipe; the complete KV does not.
The local encoder and diffusion transformer reuse the same
MiniCPM blocks with bidirectional attention. The residual model omits RoPE as
specified by the checkpoint. The encoder reads one patch per decode step and
packs up to 25 reference patches per call. The diffusion transformer evaluates
its guided and unguided rows together, each attending only itself, and fuses
consecutive CFG-Zero* Euler steps, including the input/output projections, into
9-, 3- and 1-step functions. Any `num_step` decomposes greedily into these calls;
the default ten steps take one call, since upstream's first zero-velocity step
leaves the sample unchanged. Activations
stay FP16. The residual, encoder and diffusion stages store per-channel INT8
weights; the main model keeps FP16 weights by default, because per-channel INT8
there measurably changed pronunciation.

Every stage has distinct entry-point names, and cached decode names include the
cache capacity. Reusing generic function names across structurally identical
stages produced large numerical errors through
the native host, despite identical graph operations and correct weight payloads.
Renaming the affected stage's entry points fixed its direct FP16 comparison.
This is consistent with a specialization collision. Distinct names fixed the
complete-model numerical and speech checks as well.

The default export supports the upstream runtime's full 8,192-position context.
Text, reference features, and generated patches share that budget. Combined
conditioning stores the reference twice, once for cloning and once for
continuation. Each remaining position permits 160 ms of generated audio. The
upstream default decode budget remains 4,096 patches (655.36 seconds); callers
can raise `max_decode_steps` to 8,192, subject to the remaining context. No
additional duration cap or hidden sentence splitting is applied.

The main and residual KV together use 36 KiB per context position: 18 MiB at
512 and 288 MiB at 8,192, excluding model/runtime allocations. All functions in
a stage share its weights. AudioVAE uses upstream's causal streaming decoder in
16-frame chunks on a CPU worker thread, overlapped with generation, which also
bounds working activations. Weight normalization is folded at load and
depthwise convolutions are evaluated as tap sums, which PyTorch's grouped CPU
convolution runs far more slowly. `generate` returns the complete waveform;
`stream` yields each decoded chunk as soon as it is ready.

The default bundle was cross-compiled on M5 Pro for all 21 macOS targets, M1
through M6, and passed full ANE placement verification on each. Numerical,
speech and performance results were measured only on `h17s`; other chips need
on-device checks. Clean sequential compile times were 194–255 s from M2 on and
698–912 s for the M1 family. Stage the release with:

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.voxcpm2.publish \
  --source artifacts/voxcpm2/perf2-src --no-upload --continue-on-error
```

## Independent upstream FP16 validation

`scripts/voxcpm2/reference.py` imports the pinned upstream source directly and
does not import the port. It changes the runtime configuration to CPU FP16,
keeps AudioVAE in its upstream float32 dtype, disables optimization, normalization,
denoising, and retries, and records untouched forward results through observers.
No upstream model implementation files are patched. This is CPU FP16 comparison,
not a claim of CUDA equivalence or NVIDIA performance.

```sh
git clone https://github.com/OpenBMB/VoxCPM.git artifacts/voxcpm2/upstream
git -C artifacts/voxcpm2/upstream checkout cce58a5b59303c9bd63d12f23afe6a49a4a80c59
uv venv artifacts/voxcpm2/reference-env --python 3.12
uv pip install --python artifacts/voxcpm2/reference-env/bin/python \
  -r scripts/voxcpm2/reference-requirements.txt

.venv/bin/python -m tinkerbox_ane.guard artifacts/voxcpm2/reference-run -- \
  artifacts/voxcpm2/reference-env/bin/python -u scripts/voxcpm2/reference.py

.venv/bin/python -m tinkerbox_ane.guard artifacts/voxcpm2/validation-run -- \
  .venv/bin/python -u scripts/voxcpm2/validate.py --asset /path/to/verified-bundle
```

The reference driver needs the full pinned checkpoint, including `audiovae.pth`.
It creates a separate configuration directory with links to the unchanged
weights. The cases cover English automatic speech, voice design, Chinese,
reference-only cloning, continuation, and combined conditioning. The reference
recording is generated from the neutral English greeting and saved at 16 kHz.

Validation compares main and residual prefill, cached decode over the reference
trajectory, local encoding, and diffusion output on identical FP16 inputs.
Complete free-running clips are saved and transcribed separately. Numerical
agreement and end-to-end speech content are separate checks: placement alone
does not establish model correctness, and waveform samples need not be bitwise
identical across CPU and ANE arithmetic.

The practical gate accepts prefill/component cosine similarity of at least
0.95 with relative L2 error at most 0.35, and cached decode cosine of at least
0.90 with relative L2 at most 0.50. All outputs must be finite. Each free-running
clip must finish normally, remain within half to twice the upstream duration,
and have at most 20% normalized character error against upstream's separately
transcribed clip. Exact waveform matching is not required.

Public behavior and the actual app worker can be exercised with:

```sh
TINKERBOX_VOXCPM2_TEST_ASSET=/path/to/verified-bundle \
  .venv/bin/python -m tinkerbox_ane.guard artifacts/voxcpm2/public-tests -- \
  .venv/bin/python -m pytest tests/test_voxcpm2_speech.py -q

TINKERBOX_ASSET=/path/to/verified-bundle \
  .venv/bin/python -m tinkerbox_ane.guard artifacts/voxcpm2/swift-worker -- \
  .venv/bin/python app/test_voxcpm2_speech.py
```

The focused combined-mode context reproduction uses a 29.44-second neutral
reference, assembled from eight copies of the independently generated greeting:

```sh
.venv/bin/python -m tinkerbox_ane.guard artifacts/voxcpm2/long-reference-run -- \
  .venv/bin/python -u scripts/voxcpm2/long_reference.py --asset /path/to/verified-bundle
```

Build the wheel and app before the worker check. Native transcription checks
require Apple's English and Chinese speech assets. Validation results and the
tested bundle identities are recorded separately from these reproduction steps.

## Full-context checks

After preparing the independent upstream environment above, compare cache
expansion through all tier boundaries and position 8,192:

```sh
PYTHONPATH=src .venv/bin/python -m tinkerbox_ane.guard artifacts/voxcpm2/cache-run -- \
  artifacts/voxcpm2/reference-env/bin/python -u scripts/voxcpm2/validate_cache.py \
  --asset /path/to/verified-bundle

.venv/bin/python -m tinkerbox_ane.guard artifacts/voxcpm2/full-length-run -- \
  .venv/bin/python -u scripts/voxcpm2/full_length.py --asset /path/to/verified-bundle \
  --mode clone --paragraphs 4 --max-patches 2048
```

The cache comparison evaluates the first main-model asset (14 layers) against
the unchanged upstream FP16 layers, including single-position decoding around every capacity
transition. It complements the six complete-model FP16/content cases; it is
not a complete 8,192-position full-model or maximum-duration speech benchmark.
The long-speech gate requires normal stopping, finite audio longer than sixty
seconds, and at most 20% normalized character error against its input text.

## Faster main-model variant

`--main-weights lut8-group32` stores the main model as 8-bit k-means lookup
tables, one table per 32 output channels, with FP16 activations. Single-position
decode reads half the weight bytes, cutting the main model from about 19.6 to
12.6 ms per 160 ms patch on `h17s`, about 17% faster end to end. Compression
runs once on the architecture-independent export, roughly 15 minutes per
14-layer asset on CPU; each target architecture is then compiled and verified
as usual. Run the export directly rather than under the 25-minute watchdog,
and compile it under the watchdog:

```sh
.venv/bin/python -m tinkerbox_ane.voxcpm2.export \
  --checkpoint artifacts/voxcpm2/checkpoint --main-weights lut8-group32 \
  --out artifacts/voxcpm2/lut8-export
```

The bundle records each asset's weight format in `model.json` and loads like the
default bundle. Per-channel linear INT8 for the main model was rejected: mean
normalized character error rose from 0.028 to 0.069 over the 16-clip check.

## Recorded result

Results are from Apple M5 Pro, 24 GB, macOS 27.0.1 (`h17s`); only `h17s` was
compiled. Every compiled entry of the fresh export passed full ANE placement
verification. The default bundle is `artifacts/voxcpm2/perf2-ane`; the
lookup-table variant `artifacts/voxcpm2/perf2-lut8-ane` combines its main-model
assets with the same residual, encoder and diffusion assets.

| Check | Default | Lookup-table main model |
| --- | --- | --- |
| Six FP16/content cases | Pass; all transcripts match upstream | Pass; only a trailing period differs |
| Main prefill / decode cosine | ≥ 0.999994 / 0.999987 | ≥ 0.999953 / 0.999672 |
| Residual, encoder, diffusion-velocity cosine | ≥ 0.9999, 0.99993, 0.99988 | Same assets |
| First diffusion patch cosine | ≥ 0.999615 | ≥ 0.999615 |
| Growing cache through 8,192 (14 layers) | cosine 0.999990, rel. L2 0.0048 | cosine 0.999981, rel. L2 0.0066 |
| 16 clips, mean / max character error | 0.028 / 0.136 | 0.032 / 0.136 |
| Four-paragraph clone passage | 97.9 s audio in 27.0 s, 0.26% error | 94.9 s audio in 21.7 s, 0.26% error |

Warm end-to-end real-time factors (generation time divided by audio length);
v2.2.0 was measured with its own bundle and runtime:

| Request | v2.2.0 | Default | Lookup-table main model |
| --- | --- | --- | --- |
| Automatic voice | 0.78 | 0.27 | 0.23 |
| Saved clone | 0.79 | 0.28 | 0.23 |
| 21-second passage | 0.77 | 0.26 | 0.22 |

Streaming yields 16-frame (0.64-second) pieces from the same decoder, so the
joined pieces are bit-identical to the complete waveform; this held for all six
validation cases. Warm time to first audio through the worker protocol:

| Request | Default first / complete | Lookup-table first / complete |
| --- | --- | --- |
| 2.1-second greeting | 0.28 s / 0.60 s | 0.23 s / 0.52 s |
| 51–55-second passage, automatic voice | 0.32 s / 13.5 s | 0.28 s / 12.1 s |

On the default bundle, about 94% of a generation is ANE time. Each patch spends
roughly 19.6 ms in the main model, 15.4 ms in one fused diffusion call, 3.8 ms
in the residual model and 2.2 ms in the encoder, all near ANE weight bandwidth.
The 16-clip set differs from upstream only in transcribed numerals and
homophones, identically for both bundles except one homophone. Public API and
actual Swift worker checks passed for both bundles. The commands above write
local validation reports under `artifacts/voxcpm2/`; generated JSON is not committed.

An intentionally repeated 184-second reference in combined mode did not stop
before its remaining 4,877-patch context budget in the earlier per-layer bundle.
It returned an explicit error, not a truncated clip. That stress case was not
repeated for these bundles. Short reference recordings remain the practical
choice.

First use can take several minutes to populate persistent ANE specializations.
Other chips were compiled and placement-verified but not run; no CUDA reference
or other-device performance was tested.
