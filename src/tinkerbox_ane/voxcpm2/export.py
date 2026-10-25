"""Export MiniCPM stages with prefill and cached decode, and local audio transformers."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open

from tinkerbox_ane.architectures.minicpm import Config, Transformer
from tinkerbox_ane.bundle import compile_bundle, write_checkpoint, checkpoint_digest
from tinkerbox_ane.coreai import source_hash
from . import KIND, REPO, REVISION, UPSTREAM_REVISION, SOURCE, checkpoint


def stage_configs(data, group=1):
    lm = data['lm_config']
    base = Config.from_dict(lm)
    stages = [(f'base-{i}', 'base_lm.', base, i, min(i+group, base.layers), True) for i in range(0, base.layers, group)]
    residual = Config.from_dict(dict(lm, num_hidden_layers=data['residual_lm_num_layers'],
                                     no_rope=data['residual_lm_no_rope']))
    stages.append(('residual', 'residual_lm.', residual, 0, residual.layers, True))
    for role, key, prefix in [('encoder', 'encoder_config', 'feat_encoder.encoder.'),
                              ('dit', 'dit_config', 'feat_decoder.estimator.decoder.')]:
        value = data[key]
        cfg = Config.from_dict(dict(lm, hidden_size=value['hidden_dim'], intermediate_size=value['ffn_dim'],
                                   num_attention_heads=value['num_heads'], num_hidden_layers=value['num_layers'],
                                   kv_channels=value['kv_channels']))
        stages.append((role, prefix, cfg, 0, cfg.layers, False))
    return stages


PREFILL_WIDTH = 128
CAPACITIES = (512, 1024, 2048, 4096, 8192)
BASE_GROUP = 14
# Per-channel INT8 weights for the bandwidth-bound residual and local stages:
# FP16 activations, unchanged speech content. The main model defaults to FP16,
# since INT8 there measurably changed pronunciation. Its optional 8-bit k-means
# lookup tables per 32 output channels are about 15% faster end to end, with
# near-FP16 speech content.
INT8_ROLES = ('residual', 'encoder', 'dit')
MAIN_WEIGHTS = ('float16', 'lut8-group32')


def compress(program, weights):
    from coreai_opt.coreai_utils import CompressionGranularity, DType, palettize_weights, quantize_weights
    if weights == 'int8-per-channel':
        quantize_weights(program, dtype=DType.INT8, granularity=CompressionGranularity.PER_CHANNEL,
                         weight_num_threshold=2048, in_place=True)
    elif weights == 'lut8-group32':
        palettize_weights(program, lut_dtype=None, n_bits=8, granularity=CompressionGranularity.PER_GROUPED_CHANNEL,
                          group_size=32, weight_num_threshold=2048, in_place=True)
    else:
        return program
    with program._module:  # Core AI optimizes the program when its module context exits.
        pass
    return program
# Uncached widths. The encoder reads one patch (special token plus four frames)
# while decoding and packs reference patches for prompts. The diffusion
# transformer evaluates its guided and unguided rows (two mu, time, four
# condition and four noisy frames each) together.
LOCAL_FUNCTIONS = dict(encoder=dict(main=5, batch=5*25), dit=dict(pair=22))
# Fused sampler lengths; any schedule decomposes greedily into these calls.
SAMPLER_STEPS = (9, 3, 1)


class Sampler(nn.Module):
    """Consecutive CFG-Zero* Euler steps of one patch in a single call.

    Packed rows are [mu (2), time, condition (4), noisy frames (4)], guided then
    unguided, matching the `pair` estimator layout."""

    def __init__(self, decoder, tensors, steps):
        super().__init__()
        self.decoder, self.steps = decoder, steps
        hidden = decoder.cfg.hidden
        self.in_proj, self.out_proj = nn.Conv2d(64, hidden, 1), nn.Conv2d(hidden, 64, 1)
        for name in ('in_proj', 'out_proj'):
            layer = getattr(self, name)
            layer.weight.data.copy_(tensors.get_tensor(f'feat_decoder.estimator.{name}.weight').reshape(layer.weight.shape))
            layer.bias.data.copy_(tensors.get_tensor(f'feat_decoder.estimator.{name}.bias'))

    def forward(self, x, context, clocks, dts, guidance, cos, sin, mask):
        for step in range(self.steps):
            clock, noisy = clocks[..., step:step+1], self.in_proj(x)
            hidden = torch.cat((context[..., :2], clock, context[..., 2:6], noisy,
                                context[..., 6:8], clock, context[..., 8:], noisy), dim=-1)
            out = self.out_proj(self.decoder(hidden, cos, sin, mask))
            positive, negative = out[..., 7:11], out[..., 18:]
            scale = ((positive*negative).sum(dim=(1, 3), keepdim=True)
                     / ((negative*negative).sum(dim=(1, 3), keepdim=True)+1e-8))
            negative = negative*scale
            x = x - dts[..., step:step+1]*(negative + guidance*(positive-negative))
        return x


def export(out=SOURCE, weights=None, capacities=CAPACITIES, stages=None, group=BASE_GROUP, main_weights='float16'):
    from coreai_torch import TorchConverter, get_decomp_table, ExternalizeSpec
    from coreai_torch.composite_ops import RMSNormImpl

    out, weights = Path(out), checkpoint() if weights is None else Path(weights)
    if out.exists() and any(out.iterdir()):
        raise ValueError('export destination must be empty')
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((weights / 'config.json').read_text())
    if data['architecture'] != 'voxcpm2' or data['patch_size'] != 4 or data['feat_dim'] != 64:
        raise ValueError('expected the pinned VoxCPM2 architecture')
    if (not capacities or tuple(sorted(set(capacities))) != tuple(capacities)
            or any(n < PREFILL_WIDTH or n > data['max_length'] for n in capacities)):
        raise ValueError('VoxCPM2 needs increasing cache capacities within its native context')
    if main_weights not in MAIN_WEIGHTS:
        raise ValueError(f'main-model weights must be one of {", ".join(MAIN_WEIGHTS)}')
    assets = []
    for role, prefix, cfg, start, end, cached in stage_configs(data, group):
        if stages and role not in stages:
            continue
        model = Transformer(cfg, start, end, cached=cached).half().eval().requires_grad_(False)
        with safe_open(weights / 'model.safetensors', framework='pt') as tensors:
            model.load_hf(tensors, prefix)
            samplers = ([(count, Sampler(model, tensors, count).half().eval().requires_grad_(False))
                         for count in SAMPLER_STEPS] if role == 'dit' else [])
        converter = TorchConverter()
        # Match D3: stage and capacity identify each specialization, while all
        # functions use the same Parameter objects and weight resources.
        function_prefix = 'voxcpm2_' + role.replace('-', '_') + '_'
        functions, cache_functions, sampler_functions = [], [], []
        def stage(module, args, inputs, name):
            converter.add_pytorch_module(
                module, export_fn=lambda value: torch.export.export(value, args).run_decompositions(get_decomp_table()),
                externalize_modules=[ExternalizeSpec(RMSNormImpl, composite_op_name='rms_norm',
                                                     composite_attrs=['eps', 'axes', 'version'])],
                input_names=inputs, output_names=['features'], entrypoint_name=name)
            functions.append(name)
            print(f'{role}: staged {name}', flush=True)
        variants = ([(f'c{n}w{width}', n, width) for n in capacities for width in (PREFILL_WIDTH, 1)] if cached
                    else [(name, 0, width) for name, width in LOCAL_FUNCTIONS[role].items()])
        local = {}
        for suffix, capacity, width in variants:
            args = (torch.zeros(1, cfg.hidden, 1, width, dtype=torch.float16),
                    torch.ones(1, 1, cfg.head_dim, width, dtype=torch.float16),
                    torch.zeros(1, 1, cfg.head_dim, width, dtype=torch.float16),
                    torch.zeros(1, 1, width, capacity+width, dtype=torch.float16))
            inputs = ['hidden', 'cos', 'sin', 'mask']
            name = function_prefix + suffix
            if cached:
                args += (torch.zeros(1, 2*(end-start)*cfg.kv_heads*cfg.head_dim, 1, capacity, dtype=torch.float16),)
                inputs.append('cache')
                cache_functions.append(dict(name=name, capacity=capacity, width=width))
            else:
                local[suffix] = dict(name=name, width=width)
            stage(model, args, inputs, name)
        width = LOCAL_FUNCTIONS['dit']['pair']
        for count, sampler in samplers:
            zeros = lambda *shape: torch.zeros(shape, dtype=torch.float16)
            args = (zeros(1, 64, 1, 4), zeros(1, cfg.hidden, 1, width-10), zeros(1, cfg.hidden, 1, count),
                    zeros(1, 1, 1, count), zeros(1, 1, 1, 1), torch.ones(1, 1, cfg.head_dim, width, dtype=torch.float16),
                    zeros(1, 1, cfg.head_dim, width), zeros(1, 1, width, width))
            name = function_prefix + f'steps{count}'
            stage(sampler, args, ['x', 'context', 'clocks', 'dts', 'guidance', 'cos', 'sin', 'mask'], name)
            sampler_functions.append(dict(name=name, steps=count))
        path = out / f'{role}.aimodel'
        stored = 'int8-per-channel' if role in INT8_ROLES else main_weights
        program = compress(converter.to_coreai(), stored)
        program.save_asset(path)
        assets.append(dict(role=role, asset=path.name, source_hash=source_hash(path), functions=functions,
                           config=asdict(cfg), start=start, end=end, cached=cached,
                           cache_functions=cache_functions, local_functions=local,
                           sampler_functions=sampler_functions, weights=stored))
        del converter, model, samplers
        print(f'{role}: exported', flush=True)
    cpu = write_checkpoint(weights, out, ['base_lm.embed_tokens.weight', 'fsq_layer.',
        'enc_to_lm_proj.', 'lm_to_dit_proj.', 'res_to_dit_proj.', 'fusion_concat_proj.',
        'stop_proj.', 'stop_head.', 'feat_encoder.in_proj.', 'feat_encoder.special_token',
        'feat_decoder.estimator.in_proj.', 'feat_decoder.estimator.cond_proj.',
        'feat_decoder.estimator.out_proj.', 'feat_decoder.estimator.time_mlp.',
        'feat_decoder.estimator.delta_time_mlp.'])
    spec = dict(kind=KIND, version=2, assets=assets, capacities=list(capacities), prefill_width=PREFILL_WIDTH,
                checkpoint_hash=checkpoint_digest(cpu),
                source=dict(repo=REPO, revision=REVISION, code_revision=UPSTREAM_REVISION))
    (out / 'model.json').write_text(json.dumps(spec, indent=2)+'\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=SOURCE)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--capacities', type=int, nargs='+', default=list(CAPACITIES))
    parser.add_argument('--stages', nargs='+')
    parser.add_argument('--base-group', type=int, default=BASE_GROUP, help='main-model layers per asset')
    parser.add_argument('--main-weights', choices=MAIN_WEIGHTS, default='float16')
    parser.add_argument('--compile-to', type=Path)
    parser.add_argument('--architecture')
    args = parser.parse_args()
    export(args.out, args.checkpoint, tuple(args.capacities), args.stages, args.base_group, args.main_weights)
    if args.compile_to:
        compile_bundle(args.out, args.compile_to, KIND, args.architecture)


if __name__ == '__main__':
    main()
