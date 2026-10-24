"""Export the complete diffusion transformer and audio projection for ANE."""

import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
from torch import nn
from safetensors import safe_open

from tinkerbox_ane.blocks import conv1x1
from tinkerbox_ane.architectures.qwen3 import Backbone, Config
from tinkerbox_ane.architectures.attention import BlockAttention
from tinkerbox_ane.coreai import compile_asset, source_hash
from tinkerbox_ane.bundle import (CHECKPOINT, bundled_checkpoint, checkpoint_digest, clone,
                                  read_bundle, replace_directory, write_checkpoint, compile_bundle)
from . import KIND, REPO, REVISION, SOURCE, checkpoint


class Diffusion(nn.Module):
    def __init__(self, cfg, vocabulary, splits=4):
        super().__init__()
        self.backbone = Backbone(cfg, splits)
        self.head = conv1x1(cfg.hidden, vocabulary)

    def forward(self, hidden, cos, sin, mask):
        return self.head(self.backbone(hidden, cos, sin, mask))


WIDTH = 128
CAPACITIES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 40960)


class Project(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden, cos, sin):
        values = self.layer.project(hidden, cos, sin)
        return torch.cat([x.transpose(2, 3).reshape(1, -1, 1, hidden.shape[-1]) for x in values], dim=1)


class Finish(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden, attended):
        return self.layer.finish(hidden, attended)


class Attend(nn.Module):
    def __init__(self, cfg, capacity):
        super().__init__()
        self.attention = BlockAttention(cfg, capacity, scale=cfg.head_dim**-0.5)

    def forward(self, query, key, value, mask):
        result = self.attention(query, key, value, mask)
        return result.transpose(2, 3).reshape(1, -1, 1, query.shape[2])


class Output(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm, self.head = model.backbone.norm, model.head

    def forward(self, hidden):
        return self.head(self.norm(hidden))


def export(out=SOURCE, weights=None, capacities=CAPACITIES, splits=4):
    from coreai_torch import TorchConverter, get_decomp_table, ExternalizeSpec
    from coreai_torch.composite_ops import RMSNormImpl

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('export destination must be empty')
    weights = checkpoint() if weights is None else Path(weights)
    data = json.loads((weights / 'config.json').read_text())
    cfg = Config.from_dict(data['llm_config'])
    model = Diffusion(cfg, data['num_audio_codebook'] * data['audio_vocab_size'], splits).half().eval().requires_grad_(False)
    with safe_open(weights / 'model.safetensors', framework='pt', device='cpu') as tensors:
        model.backbone.load_hf(tensors)
        head = tensors.get_tensor('audio_heads.weight').reshape(
            data['num_audio_codebook'], data['audio_vocab_size'], cfg.hidden)
        # Softmax ignores a shared vocabulary offset. Remove it before rounding
        # weights so FP16 logits retain small differences between likely tokens.
        head = head - head.mean(dim=1, keepdim=True)
        model.head.weight.data.copy_(head.reshape(-1, cfg.hidden, 1, 1))
    if (not capacities or tuple(sorted(set(capacities))) != tuple(capacities)
            or any(n < WIDTH or n % WIDTH or n > data['llm_config']['max_position_embeddings'] for n in capacities)):
        raise ValueError('OmniVoice capacities must be increasing query-block multiples within upstream context')
    converter, started = TorchConverter(), time.monotonic()
    functions = []
    def add(module, inputs, names, function, output='features'):
        converter.add_pytorch_module(
            module, export_fn=lambda value: torch.export.export(value, inputs).run_decompositions(get_decomp_table()),
            externalize_modules=[ExternalizeSpec(RMSNormImpl, composite_op_name='rms_norm',
                                                 composite_attrs=['eps', 'axes', 'version'])],
            input_names=names, output_names=[output], entrypoint_name=function)
        functions.append(function)
        print(f'staged {function} ({time.monotonic() - started:.1f}s)', flush=True)
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float16)
    projections, finishes, attention = [], [], []
    with torch.no_grad():
        for index, layer in enumerate(model.backbone.layers):
            project, finish = f'omnivoice_project{index}', f'omnivoice_finish{index}'
            add(Project(layer), (zeros(1, cfg.hidden, 1, WIDTH), zeros(1, 1, cfg.head_dim, WIDTH),
                                zeros(1, 1, cfg.head_dim, WIDTH)), ['hidden', 'cos', 'sin'], project)
            add(Finish(layer), (zeros(1, cfg.hidden, 1, WIDTH), zeros(1, cfg.heads*cfg.head_dim, 1, WIDTH)),
                ['hidden', 'attended'], finish)
            projections.append(project)
            finishes.append(finish)
        for capacity in capacities:
            name = f'omnivoice_attend{capacity}'
            add(Attend(cfg, capacity), (zeros(1, cfg.heads, WIDTH, cfg.head_dim),
                zeros(1, cfg.kv_heads, capacity, cfg.head_dim), zeros(1, cfg.kv_heads, capacity, cfg.head_dim),
                zeros(1, 1, 1, capacity)), ['query', 'key', 'value', 'mask'], name)
            attention.append(dict(name=name, capacity=capacity))
        add(Output(model), (zeros(1, cfg.hidden, 1, WIDTH),), ['hidden'], 'omnivoice_output', output='logits')
    out.mkdir(parents=True)
    asset = out / 'diffusion.aimodel'
    converter.to_coreai().save_asset(asset)
    long_functions = list(functions)
    converter, functions = TorchConverter(), []
    with torch.no_grad():
        short_capacities = [96, 128, 256, 512, 1024]
        short_functions = []
        for capacity in short_capacities:
            name = f'omnivoice_s{capacity}'
            add(model, (zeros(1, cfg.hidden, 1, capacity), zeros(1, 1, cfg.head_dim, capacity),
                        zeros(1, 1, cfg.head_dim, capacity), zeros(1, 1, capacity, capacity)),
                ['hidden', 'cos', 'sin', 'mask'], name, output='logits')
            short_functions.append(name)
    short_asset = out / 'short.aimodel'
    converter.to_coreai().save_asset(short_asset)
    cpu = write_checkpoint(weights, out, ['llm.embed_tokens.weight', 'audio_embeddings.weight'])
    spec = dict(kind=KIND, version=2, assets=[
                    dict(role='diffusion', asset=asset.name, source_hash=source_hash(asset), functions=long_functions),
                    dict(role='short', asset=short_asset.name, source_hash=source_hash(short_asset), functions=functions)],
                checkpoint_hash=checkpoint_digest(cpu),
                source=dict(repo=REPO, revision=REVISION), capacities=list(capacities),
                hidden=cfg.hidden, head_dim=cfg.head_dim, rope_theta=cfg.rope_theta,
                heads=cfg.heads, kv_heads=cfg.kv_heads, attention_width=WIDTH,
                projections=projections, finishes=finishes, attention=attention, head_function='omnivoice_output',
                short_capacities=short_capacities, short_functions=short_functions,
                codebooks=data['num_audio_codebook'], audio_vocab=data['audio_vocab_size'], splits=splits,
                normalization='coreai-rms_norm', audio_head_centering='per-codebook-vocabulary-mean',
                attention_value_scale=model.backbone.layers[0].value_scale)
    (out / 'model.json').write_text(json.dumps(spec, indent=2) + '\n')
    return out


def compile(source, out, architecture=None, *, force=False):
    from tinkerbox_ane.host import device_architecture, validate_architecture

    architecture = architecture or device_architecture()
    validate_architecture(architecture)
    source = Path(source)
    spec = read_bundle(source, KIND)
    if 'assets' in spec:
        return compile_bundle(source, Path(out), KIND, architecture, force=force)
    cpu = bundled_checkpoint(source)
    if checkpoint_digest(cpu) != spec.get('checkpoint_hash'):
        raise ValueError('checkpoint identity mismatch; re-export the bundle')
    out = Path(out)
    target = out / f'diffusion.{architecture}.aimodelc'
    compile_asset(Path(source) / spec['asset'], target, architecture=architecture,
                  min_deployment_version='27.0', expected_functions=spec.get('functions', [f's{n}' for n in spec['capacities']]),
                  timeout=1500, force=force)
    spec.update(asset=target.name, architecture=architecture, min_os='27.0')
    with tempfile.TemporaryDirectory(prefix='.checkpoint-', dir=out) as temporary:
        staged = Path(temporary) / CHECKPOINT
        clone(cpu, staged)
        replace_directory(staged, out / CHECKPOINT)
    (out / 'model.json').write_text(json.dumps(spec, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=SOURCE)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--capacities', type=int, nargs='+', default=list(CAPACITIES))
    parser.add_argument('--splits', type=int, default=4)
    parser.add_argument('--compile-to', type=Path)
    parser.add_argument('--architecture')
    args = parser.parse_args()
    export(args.out, args.checkpoint, tuple(args.capacities), args.splits)
    if args.compile_to:
        compile(args.out, args.compile_to, args.architecture)


if __name__ == '__main__':
    main()
