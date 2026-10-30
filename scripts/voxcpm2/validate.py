"""Compare ANE execution with separately captured upstream FP16 traces and audio."""

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

from tinkerbox_ane.voxcpm2.runtime import Speech
from tinkerbox_ane.transcription import transcribe
from gate import evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', type=Path, default=Path('artifacts/voxcpm2/ane'))
parser.add_argument('--reference', type=Path, default=Path('artifacts/voxcpm2/reference'))
parser.add_argument('--out', type=Path, default=Path('artifacts/voxcpm2/validation'))
parser.add_argument('--cases', nargs='+', default=['hello', 'design', 'chinese', 'clone', 'continuation', 'combined'])
parser.add_argument('--components-only', action='store_true')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

def metrics(actual, expected):
    actual, expected = actual.double().reshape(-1), expected.double().reshape(-1)
    return dict(cosine=float(F.cosine_similarity(actual, expected, dim=0)),
                relative_l2=float((actual-expected).norm()/expected.norm().clamp_min(1e-8)),
                max_abs=float((actual-expected).abs().max()), finite=bool(torch.isfinite(actual).all()))

spec = json.loads((args.asset/'model.json').read_text())
report = dict(device=subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
              architecture=spec['architecture'], os=platform.mac_ver()[0], asset=str(args.asset),
              source=spec['source'], asset_hashes={s['role']: s['source_hash'] for s in spec['assets']},
              scope='components' if args.components_only else 'end-to-end', requested_cases=args.cases,
              reference_dtype='float16', codec_dtype='float32', cases={}, passed=False)
with Speech(args.asset) as model, torch.inference_mode():
    for name in args.cases:
        saved = torch.load(args.reference/f'{name}.pt', map_location='cpu', weights_only=True)
        trace = saved['trace']
        result = dict(text=saved['text'], seed=saved['seed'], numerical={})
        for role, key in [('base', 'base_lm'), ('residual', 'residual_lm')]:
            prefill = trace[key][0]
            actual, states = model._lm(role, prefill['inputs_embeds'])
            result['numerical'][role+'_prefill'] = metrics(actual, prefill['output'])
            rows = []
            for row in trace[key+'.steps']:
                actual, states = model._lm(role, row['input'].unsqueeze(1), states)
                rows.append(metrics(actual[:, 0], row['output']))
            result['numerical'][role+'_decode'] = dict(
                steps=len(rows), min_cosine=min(x['cosine'] for x in rows),
                max_relative_l2=max(x['relative_l2'] for x in rows),
                finite=all(x['finite'] for x in rows))
        encoder = trace['feat_encoder'][0]
        encoded = model._encode(encoder['input'].reshape(-1, 4, 64))
        expected = model._linear(encoder['output'], 'enc_to_lm_proj')
        result['numerical']['encoder'] = metrics(encoded, expected)
        velocities = []
        for row in trace['feat_decoder.estimator']:
            if 'arguments' not in row:
                continue
            x, mu, clock, cond, _ = row['arguments']
            actual = torch.cat([model._estimate(x[i:i+2], mu[i:i+2], torch.cat([model._clock(t) for t in clock[i:i+2]]),
                                                cond[i:i+2]) for i in range(0, len(x), 2)])
            velocities.append(metrics(actual, row['output']))
        if velocities:
            result['numerical']['diffusion_velocity'] = dict(
                min_cosine=min(x['cosine'] for x in velocities),
                max_relative_l2=max(x['relative_l2'] for x in velocities),
                finite=all(x['finite'] for x in velocities))
        base = trace['base_lm'][0]['output'][:, -1:]
        if name in ('continuation', 'combined'):
            base = model._quantize(base)
        residual = trace['residual_lm'][0]['output'][:, -1:]
        mu = torch.cat((model._linear(base, 'lm_to_dit_proj'),
                        model._linear(residual, 'res_to_dit_proj')), dim=-1).squeeze(1)
        cond = trace['feat_encoder'][0]['input'][:, -1].transpose(1, 2)
        options = dict(num_step=10, guidance_scale=2)
        patch = model._sample(mu, cond, options, model._schedule(options), torch.Generator().manual_seed(saved['seed']))
        result['numerical']['first_diffusion_patch'] = metrics(patch.transpose(1, 2), trace['patches'][0])
        report['cases'][name] = result
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if not args.components_only and result['numerical']['base_prefill']['cosine'] >= 0.90:
            generation = dict(seed=saved['seed'])
            text = saved['text']
            if name == 'design':
                description, text = text[1:].split(')', 1)
                generation.update(mode='design', instruct=description)
            if name in ('clone', 'continuation', 'combined'):
                generation.update(mode='clone_continuation' if name == 'combined' else name,
                                  ref_audio=str(args.reference/'reference-16k.wav'),
                                  ref_text='Hello! Welcome to Tinkerbox.')
            start = time.monotonic()
            try:
                waveform = model.generate(text, **generation)[0]
                sf.write(args.out/f'{name}.wav', waveform, model.sampling_rate, subtype='FLOAT')
                result.update(seconds=len(waveform)/model.sampling_rate, elapsed=time.monotonic()-start,
                              ane_seconds=model.ane_seconds, finite=bool(np.isfinite(waveform).all()))
            except ValueError as error:
                result['generation_error'] = str(error)
            locale = 'zh-CN' if name == 'chinese' else 'en-US'
            if (args.out/f'{name}.wav').is_file() and not result.get('generation_error'):
                try:
                    result['transcript'] = transcribe(args.out/f'{name}.wav', locale=locale)
                except RuntimeError as error:
                    result['transcription_error'] = str(error)
            result['upstream_transcript'] = transcribe(args.reference/f'{name}.wav', locale=locale)
            reference, rate = sf.read(args.reference/f'{name}.wav')
            result['upstream_seconds'] = len(reference)/rate
        report['cases'][name] = result
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, json.dumps(result), flush=True)
evaluate(report)
(args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print('PASS' if report['passed'] else 'FAIL', flush=True)
raise SystemExit(0 if report['passed'] else 1)
