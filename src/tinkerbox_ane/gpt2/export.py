"""Export any GPT-2 size with cached prefill and decode, using shared weights."""

import argparse
import gc
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from tinkerbox_ane.architectures.gpt2 import Config, Transformer
from tinkerbox_ane.bundle import write_checkpoint, checkpoint_digest
from tinkerbox_ane.coreai import compress, source_hash
from . import KIND, VARIANTS, checkpoint

PREFILL_WIDTH = 64
CAPACITIES = (256, 512, 1024)


def _export_stage(model, out, weights, capacities, batches, weight_format, start, end, workers):
    from coreai_torch import TorchConverter, get_decomp_table

    cfg = Config.from_dict(json.loads((weights/'config.json').read_text()))
    decompositions = get_decomp_table()
    decompositions.pop(torch.ops.aten.gelu.default, None)
    transformer = Transformer(cfg, start, end).half().eval().requires_grad_(False)
    with safe_open(weights/'model.safetensors', framework='pt') as tensors:
        transformer.load_hf(tensors)
    if weight_format == 'lut8-group32':
        from coreai_opt.palettization import (
            KMeansPalettizer, KMeansPalettizerConfig, ModuleKMeansPalettizerConfig, PalettizationSpec,
        )
        from coreai_opt.palettization.spec import PerGroupedChannelGranularity

        # Cluster each projection once, then reuse its LUT across all entry points.
        spec = PalettizationSpec(n_bits=8, granularity=PerGroupedChannelGranularity(axis=0, group_size=32))
        settings = ModuleKMeansPalettizerConfig(op_state_spec={'weight': spec}, enable_fast_kmeans_mode=False)
        palettizer = KMeansPalettizer(transformer, KMeansPalettizerConfig(
            global_config=None, module_type_configs={torch.nn.Conv2d: settings}))
        example = (torch.zeros(1, cfg.hidden, 1, 1, dtype=torch.float16),
                   torch.zeros(1, 1, 1, 65, dtype=torch.float16),
                   torch.zeros(1, 2*(end-start)*cfg.hidden, 1, 64, dtype=torch.float16))
        # Fast-mode decimal rounding can overflow the rescaled FP16 V weights.
        palettizer.prepare(example, num_workers=workers)
        transformer = palettizer.finalize().eval().requires_grad_(False)
    converter = TorchConverter()
    functions, names = [], []
    for batch, capacity in [(b, c) for b in batches for c in sorted({n*b for n in capacities} | set(capacities))]:
        for width in (PREFILL_WIDTH, batch):
            # Distinct stage entry points prevent equal-shaped middle stages from
            # sharing a specialization that contains another stage's weights.
            name = f'gpt2_l{start}c{capacity}w{width}'
            functions.append(dict(name=name, capacity=capacity, width=width, batch=batch))
            if name in names:
                continue
            names.append(name)
            args = (torch.zeros(1, cfg.hidden, 1, width, dtype=torch.float16),
                    torch.zeros(1, 1, width, capacity+width, dtype=torch.float16),
                    torch.empty(1, 2*(end-start)*cfg.hidden, 1, capacity, dtype=torch.float16))
            converter.add_pytorch_module(
                transformer,
                export_fn=lambda module, inputs=args: torch.export.export(module, inputs).run_decompositions(decompositions),
                input_names=['hidden', 'mask', 'cache'], output_names=['features'], entrypoint_name=name)
            print(f'{model} {start}:{end}: staged {name}', flush=True)
    asset = out/(f'decoder-{start}.aimodel' if end-start < cfg.layers else 'decoder.aimodel')
    program = converter.to_coreai()
    # LUT8 clusters the PyTorch weights above; INT8 quantizes the converted program.
    if weight_format == 'int8-per-channel':
        compress(program, weight_format)
    program.save_asset(asset)
    return dict(asset=asset.name, source_hash=source_hash(asset), start=start, end=end,
                functions=names, cache_functions=functions)


def export(model='gpt2', out=None, weights=None, capacities=CAPACITIES, batches=(1, 4), weight_format='float16', workers=4):
    out = Path(out or f'artifacts/{model}/source')
    weights = checkpoint(model) if weights is None else Path(weights)
    if out.exists() and any(out.iterdir()):
        raise ValueError('export destination must be empty')
    cfg = Config.from_dict(json.loads((weights/'config.json').read_text()))
    out.mkdir(parents=True, exist_ok=True)
    assets = []
    # Bound each cache input to 768 MiB: larger tensors can fail ANE compilation.
    # Balance contiguous stages while preserving the full logical context.
    cache_bytes_per_layer = 2*cfg.hidden*max(capacities)*max(batches)*2
    stage_layers = (768*1024**2)//cache_bytes_per_layer
    group = math.ceil(cfg.layers/math.ceil(cfg.layers/stage_layers))
    for start in range(0, cfg.layers, group):
        end = min(start+group, cfg.layers)
        assets.append(_export_stage(model, out, weights, capacities, batches, weight_format, start, end, workers))
        gc.collect()
    cpu = write_checkpoint(weights, out, ['wte.weight', 'wpe.weight'], dtype=torch.float16)
    repo, revision = VARIANTS[model]
    spec = dict(kind=KIND, version=2, model=model, config=asdict(cfg),
                capacities=list(capacities), prefill_width=PREFILL_WIDTH, batch_sizes=list(batches), weights=weight_format,
                assets=assets,
                checkpoint_hash=checkpoint_digest(cpu), source=dict(repo=repo, revision=revision))
    (out/'model.json').write_text(json.dumps(spec, indent=2)+'\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=VARIANTS)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--capacities', type=int, nargs='+', default=list(CAPACITIES))
    parser.add_argument('--batch-sizes', type=int, nargs='+', choices=(1, 4, 8), default=[1, 4])
    parser.add_argument('--weights', choices=('float16', 'int8-per-channel', 'lut8-group32'), default='float16')
    parser.add_argument('--workers', type=int, default=4, help='parallel LUT codebook workers')
    args = parser.parse_args()
    export(args.model, args.out, args.checkpoint, tuple(args.capacities), tuple(args.batch_sizes),
           args.weights, args.workers)


if __name__ == '__main__':
    main()
