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
retries, external text normalization, ZipEnhancer denoising, LoRA, and streaming
delivery are not exposed. Upstream's internal CFG-Zero* sampler is retained.

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

The 28-layer main model uses one asset per layer. The eight-layer residual
model has its own asset. Both expose prefill and cached decoding; each request
owns its KV buffers. The local encoder and diffusion transformer reuse the same
MiniCPM blocks with bidirectional attention. The residual model omits RoPE as
specified by the checkpoint. Projection weights retain their original FP16
range.

Every stage has distinct entry-point names, and cached decode names include the
cache capacity. Reusing generic function names across structurally identical
stages produced large numerical errors through
the native host, despite identical graph operations and correct weight payloads.
Renaming the affected stage's entry points fixed its direct FP16 comparison.
This is consistent with a specialization collision. Distinct names fixed the
complete-model numerical and speech checks as well.

Export currently uses one prefill capacity per bundle. The default is 1024;
this is an implementation limit, while the upstream runtime is configured for
8192 positions. Text, reference features, and generated patches share the
compiled budget. Combined conditioning stores the reference twice, once for
cloning and once for continuation. Each remaining position permits 160 ms of
generated audio; a long reference and its transcript reduce that room. No
additional duration cap or hidden sentence splitting is applied.

Only the M5 Pro (`h17s`) bundle is being validated here. Other chips require
their own compilation and on-device correctness/performance checks; standard
M4 cross-compilation uses `--architecture h16g`.

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

## Recorded result

The fresh 1,024-position export passed all six practical FP16/content cases on
Apple M5 Pro, 24 GB, macOS 27.0.1 (`h17s`). All compiled entries passed full ANE
placement verification. Main-model prefill cosine was at least 0.999994 and
cached decode cosine at least 0.999986. All six clips had zero normalized
character error against separately transcribed upstream audio. Five clips had
the same duration; the greeting was one 160 ms patch shorter.

The long-reference reproduction exhausted the old 512-position context after
7.2 seconds. The 1,024-position bundle completed the 9.92-second passage with
zero transcription error. The actual Swift worker passed with the larger
bundle, including replay after unloading and all three saved reference modes.
The two public API regressions passed on the original validated 512-position
bundle. OmniVoice's existing bundle also passed its real Swift-worker check
after the shared API changes. SwiftUI and App Intent metadata builds passed.

First load can take several minutes to populate persistent ANE specialization
caches. The six short clips on the 1,024-position bundle generated in roughly
2–4 seconds each after loading. These measurements use ten diffusion steps.
No M4 bundle or CUDA reference was tested.

The full metrics, source identities, tested bundle hashes, and long-reference
result are in local validation reports under `artifacts/voxcpm2/`.
