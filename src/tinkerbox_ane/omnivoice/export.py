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
from tinkerbox_ane.coreai import compile_asset, source_hash
from tinkerbox_ane.bundle import (CHECKPOINT, bundled_checkpoint, checkpoint_digest, clone,
                                  read_bundle, replace_directory, write_checkpoint)
from . import KIND, REPO, REVISION, SOURCE, checkpoint


class Diffusion(nn.Module):
    def __init__(self, cfg, vocabulary, splits=4):
        super().__init__()
        self.backbone = Backbone(cfg, splits)
        self.head = conv1x1(cfg.hidden, vocabulary)

    def forward(self, hidden, cos, sin, mask):
        return self.head(self.backbone(hidden, cos, sin, mask))


def export(out=SOURCE, weights=None, capacities=(96, 128, 256, 512, 1024), splits=4):
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
    converter, started = TorchConverter(), time.monotonic()
    with torch.no_grad():
        for capacity in capacities:
            inputs = (torch.zeros(1, cfg.hidden, 1, capacity, dtype=torch.float16),
                      torch.ones(1, 1, cfg.head_dim, capacity, dtype=torch.float16),
                      torch.zeros(1, 1, cfg.head_dim, capacity, dtype=torch.float16),
                      torch.zeros(1, 1, capacity, capacity, dtype=torch.float16))
            converter.add_pytorch_module(
                model, export_fn=lambda module, inputs=inputs: torch.export.export(module, inputs).run_decompositions(get_decomp_table()),
                externalize_modules=[ExternalizeSpec(RMSNormImpl, composite_op_name='rms_norm',
                                                     composite_attrs=['eps', 'axes', 'version'])],
                input_names=['hidden', 'cos', 'sin', 'mask'], output_names=['logits'],
                entrypoint_name=f's{capacity}')
            print(f'staged s{capacity} ({time.monotonic() - started:.1f}s)', flush=True)
    out.mkdir(parents=True)
    asset = out / 'diffusion.aimodel'
    converter.to_coreai().save_asset(asset)
    cpu = write_checkpoint(weights, out, ['llm.embed_tokens.weight', 'audio_embeddings.weight'])
    spec = dict(kind=KIND, version=2, asset=asset.name, source_hash=source_hash(asset),
                checkpoint_hash=checkpoint_digest(cpu),
                source=dict(repo=REPO, revision=REVISION), capacities=list(capacities),
                hidden=cfg.hidden, head_dim=cfg.head_dim, rope_theta=cfg.rope_theta,
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
    cpu = bundled_checkpoint(source)
    if checkpoint_digest(cpu) != spec.get('checkpoint_hash'):
        raise ValueError('checkpoint identity mismatch; re-export the bundle')
    out = Path(out)
    target = out / f'diffusion.{architecture}.aimodelc'
    compile_asset(Path(source) / spec['asset'], target, architecture=architecture,
                  min_deployment_version='27.0', expected_functions=[f's{n}' for n in spec['capacities']],
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
    parser.add_argument('--capacities', type=int, nargs='+', default=[96, 128, 256, 512, 1024])
    parser.add_argument('--splits', type=int, default=4)
    parser.add_argument('--compile-to', type=Path)
    parser.add_argument('--architecture')
    args = parser.parse_args()
    export(args.out, args.checkpoint, tuple(args.capacities), args.splits)
    if args.compile_to:
        compile(args.out, args.compile_to, args.architecture)


if __name__ == '__main__':
    main()
