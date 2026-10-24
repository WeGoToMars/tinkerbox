"""Compare growing native KV against pinned upstream FP16 through position 8192."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors import safe_open
import torch
from torch.nn import functional as F

from tinkerbox_ane.host import TensorHost

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/voxcpm2/checkpoint'))
parser.add_argument('--upstream', type=Path, default=Path('artifacts/voxcpm2/upstream'))
parser.add_argument('--reference', type=Path, default=Path('artifacts/voxcpm2/reference/hello.pt'))
parser.add_argument('--out', type=Path, default=Path('artifacts/voxcpm2/cache-validation.json'))
args = parser.parse_args()
revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if revision != 'cce58a5b59303c9bd63d12f23afe6a49a4a80c59':
    raise ValueError('expected the pinned independent upstream source')
sys.path.insert(0, str((args.upstream/'src').resolve()))
from voxcpm.modules.minicpm4 import MiniCPMModel, MiniCPM4Config

torch.set_num_threads(4)
cfg = json.loads((args.checkpoint/'config.json').read_text())['lm_config']
cfg.update(num_hidden_layers=1, vocab_size=0)
reference = MiniCPMModel(MiniCPM4Config(**cfg)).half().eval().requires_grad_(False)
with safe_open(args.checkpoint/'model.safetensors', framework='pt') as weights:
    reference.layers[0].load_state_dict({k.removeprefix('base_lm.layers.0.'): weights.get_tensor(k)
                                        for k in weights.keys() if k.startswith('base_lm.layers.0.')})
spec = json.loads((args.asset/'model.json').read_text())
stage = next(row for row in spec['assets'] if row['role'] == 'base-0')
positions = max(spec['capacities'])
x = torch.load(args.reference, weights_only=True)['trace']['base_lm'][0]['inputs_embeds']
x = x.repeat(1, positions//x.shape[1]+1, 1)[:, :positions]
config = dict(hidden=stage['config']['hidden'],
              channels=2*stage['config']['kv_heads']*stage['config']['head_dim'],
              functions=stage['cache_functions'])
with torch.inference_mode():
    cos, sin = reference.rope_emb(torch.arange(positions))
    expected, _ = reference.layers[0](x, (cos, sin), True)
    host = TensorHost(args.asset/stage['asset'], output='features', cached=config)
    try:
        host.reset()
        parts, position = [], 0
        boundaries = sorted({p for tier in spec['capacities'] for p in (tier-1, tier) if p < positions})
        while position < positions:
            remaining = next((b-position for b in boundaries if b > position), positions-position)
            count = 1 if position in boundaries else min(spec['prefill_width'], remaining, positions-position)
            width = 1 if count == 1 else spec['prefill_width']
            capacity = next(c for c in spec['capacities'] if c >= position+count)
            pad = ((0, 0), (0, 0), (0, 0), (0, width-count))
            values = dict(hidden=np.pad(x[:, position:position+count].transpose(1, 2).unsqueeze(2).numpy(), pad),
                          cos=np.pad(cos[position:position+count].T[None, None].numpy(), pad),
                          sin=np.pad(sin[position:position+count].T[None, None].numpy(), pad))
            function = next(row['name'] for row in stage['cache_functions']
                            if row['capacity'] == capacity and row['width'] == width)
            output = host.step(function, values, count)
            parts.append(torch.from_numpy(output[:, :, 0].copy()).transpose(1, 2))
            position += count
            print('position', position, 'capacity', capacity, flush=True)
    finally:
        host.close()
    actual, expected = torch.cat(parts, dim=1).double().flatten(), expected.double().flatten()
    report = dict(asset=str(args.asset), architecture=spec['architecture'], source_hash=stage['source_hash'],
                  upstream_revision=revision, positions=positions,
                  cosine=float(F.cosine_similarity(actual, expected, dim=0)),
                  relative_l2=float((actual-expected).norm()/expected.norm()),
                  finite=bool(torch.isfinite(actual).all()))
report['passed'] = report['cosine'] >= .95 and report['relative_l2'] <= .35 and report['finite']
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
raise SystemExit(0 if report['passed'] else 1)
