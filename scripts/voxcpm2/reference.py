"""Capture the untouched pinned upstream CPU FP16 pipeline for ANE validation.

Run with the upstream reference environment, not the product runtime. This driver
does not import tinkerbox_ane. Hooks record inputs/outputs without changing them.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import soundfile as sf
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path('artifacts/voxcpm2/upstream'))
parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/voxcpm2/checkpoint'))
parser.add_argument('--out', type=Path, default=Path('artifacts/voxcpm2/reference'))
parser.add_argument('--cases', nargs='+', default=['hello', 'design', 'chinese', 'clone', 'continuation', 'combined'])
args = parser.parse_args()
revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
if revision != 'cce58a5b59303c9bd63d12f23afe6a49a4a80c59':
    raise ValueError('Use the pinned upstream source revision')
sys.path.insert(0, str(args.source.resolve()/'src'))
from voxcpm.model.voxcpm2 import VoxCPM2Model
import transformers

torch.set_num_threads(4)
args.out.mkdir(parents=True, exist_ok=True)
weights = args.out/'fp16-checkpoint'
weights.mkdir(exist_ok=True)
for file in args.checkpoint.iterdir():
    if file.is_file() and file.name != 'config.json':
        target = weights/file.name
        if not target.exists():
            target.symlink_to(file.resolve())
config = json.loads((args.checkpoint/'config.json').read_text())
config.update(dtype='float16', device='cpu')
(weights/'config.json').write_text(json.dumps(config))
started = time.monotonic()
model = VoxCPM2Model.from_local(str(weights), device='cpu', optimize=False)
assert next(model.base_lm.parameters()).dtype == torch.float16
assert next(model.feat_decoder.parameters()).dtype == torch.float16
assert next(model.audio_vae.parameters()).dtype == torch.float32
print('LOADED', time.monotonic()-started, flush=True)
trace = {}

def capture_forward(name):
    def hook(_module, _args, kwargs, result):
        values = trace.setdefault(name, [])
        if len(values) < 4:
            saved = {key: value.detach().cpu().clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
            if _args:
                saved['input'] = _args[0].detach().cpu().clone()
                saved['arguments'] = [x.detach().cpu().clone() for x in _args if isinstance(x, torch.Tensor)]
            saved['output'] = (result[0] if isinstance(result, tuple) else result).detach().cpu().clone()
            values.append(saved)
    return hook

for name in ('base_lm', 'residual_lm', 'feat_encoder', 'feat_decoder.estimator'):
    model.get_submodule(name).register_forward_hook(capture_forward(name), with_kwargs=True)
for name in ('base_lm', 'residual_lm'):
    module = model.get_submodule(name)
    original = module.forward_step
    def step(*arguments, _original=original, _name=name, **kwargs):
        result = _original(*arguments, **kwargs)
        trace.setdefault(_name+'.steps', []).append(dict(
            input=arguments[0].detach().cpu().clone(), output=result.detach().cpu().clone()))
        return result
    module.forward_step = step

def latent_hook(_module, _arguments, _kwargs, result):
    trace.setdefault('patches', []).append(result.detach().cpu().clone())
model.feat_decoder.register_forward_hook(latent_hook, with_kwargs=True)
def stop_hook(_module, _arguments, result):
    trace.setdefault('stop', []).append(result.detach().cpu().clone())
model.stop_head.register_forward_hook(stop_hook)

cases = {
    'hello': dict(text='Hello! Welcome to Tinkerbox.', seed=42),
    'design': dict(text='(A warm, calm female voice)A familiar voice can tell a new story.', seed=7),
    'chinese': dict(text='你好，欢迎使用语音合成。', seed=42),
    'clone': dict(text='This is a short voice cloning demonstration.', seed=23),
    'continuation': dict(text='This sentence continues with the same familiar voice.', seed=23),
    'combined': dict(text='Every sentence should reach its final word.', seed=23),
}
report = dict(source_revision=revision, device='cpu', dtype='float16', codec_dtype='float32',
              torch=torch.__version__, transformers=transformers.__version__,
              normalization=False, denoising=False, retries=False, cases={})
if (args.out/'report.json').is_file():
    report['cases'].update(json.loads((args.out/'report.json').read_text())['cases'])
reference = args.out/'reference-16k.wav'
for name in args.cases:
    item = cases[name]
    trace.clear()
    generation = dict(target_text=item['text'], seed=item['seed'], inference_timesteps=10,
                      cfg_value=2.0, retry_badcase=False, min_len=2, max_len=256)
    if name in ('clone', 'combined'):
        generation['reference_wav_path'] = str(reference)
    if name in ('continuation', 'combined'):
        generation.update(prompt_wav_path=str(reference), prompt_text=cases['hello']['text'])
    start = time.monotonic()
    waveform = model.generate(**generation)
    sf.write(args.out/f'{name}.wav', waveform[0].numpy(), model.sample_rate, subtype='FLOAT')
    if name == 'hello':
        import librosa
        sf.write(reference, librosa.resample(waveform[0].numpy(), orig_sr=model.sample_rate, target_sr=16000), 16000, subtype='FLOAT')
    torch.save(dict(text=item['text'], seed=item['seed'], trace=trace), args.out/f'{name}.pt')
    report['cases'][name] = dict(text=item['text'], seed=item['seed'],
        seconds=waveform.numel()/model.sample_rate, patches=len(trace['patches']), elapsed=time.monotonic()-start,
        finite=bool(torch.isfinite(waveform).all()))
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(name, report['cases'][name], flush=True)
