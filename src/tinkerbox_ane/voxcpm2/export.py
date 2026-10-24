"""Export MiniCPM stages with prefill and cached decode, and local audio transformers."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tinkerbox_ane.architectures.minicpm import Config, Transformer
from tinkerbox_ane.bundle import compile_bundle, write_checkpoint, checkpoint_digest
from tinkerbox_ane.coreai import source_hash
from . import KIND, REPO, REVISION, UPSTREAM_REVISION, SOURCE, checkpoint


def stage_configs(data):
    lm = data['lm_config']
    base = Config.from_dict(lm)
    stages = [(f'base-{i}', 'base_lm.', base, i, i+1, True) for i in range(base.layers)]
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


def export(out=SOURCE, weights=None, capacities=(1024,), stages=None):
    from coreai_torch import TorchConverter, get_decomp_table, ExternalizeSpec
    from coreai_torch.composite_ops import RMSNormImpl

    out, weights = Path(out), checkpoint() if weights is None else Path(weights)
    if out.exists() and any(out.iterdir()):
        raise ValueError('export destination must be empty')
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((weights / 'config.json').read_text())
    if data['architecture'] != 'voxcpm2' or data['patch_size'] != 4 or data['feat_dim'] != 64:
        raise ValueError('expected the pinned VoxCPM2 architecture')
    if len(capacities) != 1 or not 4 <= capacities[0] <= data['max_length']:
        raise ValueError('VoxCPM2 needs one valid prefill capacity per bundle')
    assets = []
    for role, prefix, cfg, start, end, cached in stage_configs(data):
        if stages and role not in stages:
            continue
        model = Transformer(cfg, start, end, cached=cached).half().eval().requires_grad_(False)
        with safe_open(weights / 'model.safetensors', framework='pt') as tensors:
            model.load_hf(tensors, prefix)
        converter = TorchConverter()
        lengths = capacities if cached else [5 if role == 'encoder' else 11]
        # Identical entry names on structurally identical stages can reuse a
        # specialization with incorrect constant bindings. Keep them distinct.
        function_prefix = 'voxcpm2_' + role.replace('-', '_') + '_'
        functions = []
        for n in lengths:
            args = (torch.zeros(1, cfg.hidden, 1, n, dtype=torch.float16),
                    torch.ones(1, 1, cfg.head_dim, n, dtype=torch.float16),
                    torch.zeros(1, 1, cfg.head_dim, n, dtype=torch.float16),
                    torch.zeros(1, 1, n, n, dtype=torch.float16))
            name = function_prefix + (f'p{n}' if cached else 'main')
            converter.add_pytorch_module(
                model, export_fn=lambda module, args=args: torch.export.export(module, args).run_decompositions(get_decomp_table()),
                externalize_modules=[ExternalizeSpec(RMSNormImpl, composite_op_name='rms_norm',
                                                     composite_attrs=['eps', 'axes', 'version'])],
                input_names=['hidden', 'cos', 'sin', 'mask'], output_names=['features'], entrypoint_name=name)
            functions.append(name)
            print(f'{role}: staged {name}', flush=True)
        if cached:
            n = capacities[-1]
            args = (torch.zeros(1, cfg.hidden, 1, 1, dtype=torch.float16),
                    torch.ones(1, 1, cfg.head_dim, 1, dtype=torch.float16),
                    torch.zeros(1, 1, cfg.head_dim, 1, dtype=torch.float16),
                    torch.zeros(1, 1, 1, n+1, dtype=torch.float16),
                    torch.zeros(1, 2*(end-start)*cfg.kv_heads*cfg.head_dim, 1, n, dtype=torch.float16))
            converter.add_pytorch_module(
                model, export_fn=lambda module, args=args: torch.export.export(module, args).run_decompositions(get_decomp_table()),
                externalize_modules=[ExternalizeSpec(RMSNormImpl, composite_op_name='rms_norm',
                                                     composite_attrs=['eps', 'axes', 'version'])],
                input_names=['hidden', 'cos', 'sin', 'mask', 'cache'],
                output_names=['features'], entrypoint_name=function_prefix+f'decode{n}')
            functions.append(function_prefix+f'decode{n}')
        path = out / f'{role}.aimodel'
        converter.to_coreai().save_asset(path)
        assets.append(dict(role=role, asset=path.name, source_hash=source_hash(path), functions=functions,
                           config=asdict(cfg), start=start, end=end, cached=cached))
        del converter, model
        print(f'{role}: exported', flush=True)
    cpu = write_checkpoint(weights, out, ['base_lm.embed_tokens.weight', 'fsq_layer.',
        'enc_to_lm_proj.', 'lm_to_dit_proj.', 'res_to_dit_proj.', 'fusion_concat_proj.',
        'stop_proj.', 'stop_head.', 'feat_encoder.in_proj.', 'feat_encoder.special_token',
        'feat_decoder.estimator.in_proj.', 'feat_decoder.estimator.cond_proj.',
        'feat_decoder.estimator.out_proj.', 'feat_decoder.estimator.time_mlp.',
        'feat_decoder.estimator.delta_time_mlp.'])
    spec = dict(kind=KIND, version=2, assets=assets, capacities=list(capacities),
                checkpoint_hash=checkpoint_digest(cpu),
                source=dict(repo=REPO, revision=REVISION, code_revision=UPSTREAM_REVISION))
    (out / 'model.json').write_text(json.dumps(spec, indent=2)+'\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=SOURCE)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--capacities', type=int, nargs='+', default=[1024])
    parser.add_argument('--stages', nargs='+')
    parser.add_argument('--compile-to', type=Path)
    parser.add_argument('--architecture')
    args = parser.parse_args()
    export(args.out, args.checkpoint, tuple(args.capacities), args.stages)
    if args.compile_to:
        compile_bundle(args.out, args.compile_to, KIND, args.architecture)


if __name__ == '__main__':
    main()
