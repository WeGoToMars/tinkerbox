"""Compare OmniVoice to independent FP16 captures and exercise long speech."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from tinkerbox_ane.host import TensorHost
from tinkerbox_ane.omnivoice.runtime import Speech
from tinkerbox_ane.transcription import transcribe
from speech_case import MAX_MEAN_ERROR, SEEDS, SETTINGS, TEXT, character_error


def metric(actual, expected):
    actual, expected = actual.double().flatten(), expected.double().flatten()
    return dict(cosine=float(F.cosine_similarity(actual, expected, dim=0)),
                relative_l2=float((actual-expected).norm()/expected.norm()),
                finite=bool(torch.isfinite(actual).all()))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--reference', type=Path, default=Path('artifacts/omnivoice/chunked-reference'))
parser.add_argument('--out', type=Path, default=Path('artifacts/omnivoice/full-context-validation'))
parser.add_argument('--speech', action='store_true', help='Also generate and transcribe a 60-second passage per seed')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
spec = json.loads((args.asset/'model.json').read_text())
assets = {stage['role']: stage for stage in spec.get('assets', [])}
long_asset = assets.get('diffusion', spec)
report = dict(device=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
              architecture=spec['architecture'], os=platform.mac_ver()[0], asset=str(args.asset),
              asset_hashes={role: row['source_hash'] for role, row in assets.items()} if assets else {'diffusion': spec['source_hash']},
              context_positions=max(spec['capacities']), cases={})


def save():
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


with Speech(args.asset) as model, torch.inference_mode():
    for path in sorted(args.reference.glob('*.pt')):
        row = torch.load(path, weights_only=True)
        started = time.monotonic()
        actual = model.logits(row['embeddings'], row['attention_mask'])
        expected = row['logits'].float()
        actual = actual-actual.mean(-1, keepdim=True)
        expected = expected-expected.mean(-1, keepdim=True)
        result = dict(metric(actual, expected), elapsed=time.monotonic()-started)
        report['cases'][path.stem] = result
        print(path.stem, result, flush=True)
        save()
    if args.speech:
        clips = []
        for seed in SEEDS:
            started = time.monotonic()
            audio = model.generate(TEXT, mode='design', seed=seed, postprocess_output=False, pad_duration=0,
                                   fade_duration=0, **SETTINGS)[0]
            path = args.out/f'long-speech-{seed}.wav'
            sf.write(path, audio, model.sampling_rate, subtype='FLOAT')
            transcript = transcribe(path, locale='en-US')
            clips.append(dict(seed=seed, transcript=transcript, character_error=character_error(transcript, TEXT),
                              seconds=len(audio)/model.sampling_rate, elapsed=time.monotonic()-started,
                              finite=bool(np.isfinite(audio).all())))
            print('speech', clips[-1], flush=True)
        mean = float(np.mean([clip['character_error'] for clip in clips]))
        report['speech'] = dict(text=TEXT, clips=clips, mean_character_error=mean, max_mean_error=MAX_MEAN_ERROR,
                                passed=mean <= MAX_MEAN_ERROR and all(c['finite'] and c['seconds'] == 60 for c in clips))
        save()

with torch.inference_mode():
    host = TensorHost(args.asset/long_asset['asset'], output='features')
    generator = torch.Generator().manual_seed(42)
    capacity, width, dim = max(spec['capacities']), spec['attention_width'], spec['head_dim']
    q = torch.randn(1, spec['heads'], width, dim, generator=generator, dtype=torch.float16)
    k = torch.randn(1, spec['kv_heads'], capacity, dim, generator=generator, dtype=torch.float16)
    v = torch.randn(1, spec['kv_heads'], capacity, dim, generator=generator, dtype=torch.float16)*16
    for visible in [capacity, 4096]:
        mask = torch.full((1, 1, 1, capacity), -30000, dtype=torch.float16)
        mask[..., :visible] = 0
        group = spec['heads']//spec['kv_heads']
        expected = F.scaled_dot_product_attention(q, k.repeat_interleave(group, dim=1),
                                                 v.repeat_interleave(group, dim=1), attn_mask=mask)
        function = next(row['name'] for row in spec['attention'] if row['capacity'] == capacity)
        out = host.run(function, dict(query=q.numpy(), key=k.numpy(), value=v.numpy(), mask=mask.numpy()))
        actual = torch.from_numpy(out.copy()).reshape(1, spec['heads'], dim, width).transpose(2, 3)
        result = metric(actual, expected)
        report['cases'][f'attention-{capacity}-visible-{visible}'] = result
        print('attention', visible, result, flush=True)
        save()
    host.close()
report['passed'] = (bool(report['cases']) and
                    all(r['finite'] and r['cosine'] >= .95 and r['relative_l2'] <= .35 for r in report['cases'].values()) and
                    report.get('speech', {}).get('passed', True))
save()
print('PASS' if report['passed'] else 'FAIL', flush=True)
raise SystemExit(0 if report['passed'] else 1)
