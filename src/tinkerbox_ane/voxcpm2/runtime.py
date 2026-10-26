"""Cached ANE MiniCPM generation with native VoxCPM2 conditioning and CPU AudioVAE."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import secrets
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open

from tinkerbox_ane.audio import read_audio
from tinkerbox_ane.bundle import bundled_checkpoint, checkpoint_digest, read_bundle
from tinkerbox_ane.host import TensorHost, device_architecture
from tinkerbox_ane.speech import speech_controls
from tinkerbox_ane.transcription import transcribe
from . import KIND, REPO, REVISION, UPSTREAM_REVISION
from .codec import AudioVAE, AudioVAEConfig, fold_weight_norm, trim_reference
from .parameters import CAPABILITIES, PARAMETERS, validate
from .prompt import build_inputs


@dataclass
class VoiceClonePrompt:
    reference: torch.Tensor | None
    continuation: torch.Tensor | None
    ref_text: str
    mode: str

    def save(self, path):
        torch.save(dict(kind=KIND, revision=REVISION, reference=self.reference,
                        continuation=self.continuation, ref_text=self.ref_text, mode=self.mode), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(data, dict) or data.get('kind') != KIND or data.get('revision') != REVISION:
            raise ValueError('Choose a saved VoxCPM2 voice prompt for this checkpoint.')
        mode = data.get('mode')
        if mode not in ('clone', 'continuation', 'clone_continuation') or not isinstance(data.get('ref_text'), str):
            raise ValueError('Invalid VoxCPM2 voice prompt.')
        for key in ('reference', 'continuation'):
            value = data.get(key)
            needed = mode != ('continuation' if key == 'reference' else 'clone')
            if needed:
                if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[1:] != (4, 64)
                        or not 0 < len(value) < 8192 or not torch.isfinite(value).all()):
                    raise ValueError('Invalid VoxCPM2 reference features.')
            elif value is not None:
                raise ValueError('Voice conditioning and saved features differ.')
        if mode != 'clone' and not data['ref_text'].strip():
            raise ValueError('Continuation requires a matching reference transcript.')
        return cls(data.get('reference'), data.get('continuation'), data['ref_text'], mode)


class Speech:
    def __init__(self, asset):
        from transformers import LlamaTokenizerFast

        asset = Path(asset)
        self.spec = read_bundle(asset, KIND)
        if self.spec.get('architecture') != device_architecture():
            raise ValueError('VoxCPM2 bundle targets a different device architecture')
        if self.spec['source'] != dict(repo=REPO, revision=REVISION, code_revision=UPSTREAM_REVISION):
            raise ValueError('VoxCPM2 bundle and runtime revisions differ')
        weights = bundled_checkpoint(asset)
        if checkpoint_digest(weights) != self.spec.get('checkpoint_hash'):
            raise ValueError('VoxCPM2 CPU checkpoint identity mismatch; re-export the bundle')
        self.config = json.loads((weights / 'config.json').read_text())
        self.capacities = self.spec['capacities']
        self.prefill_width = self.spec.get('prefill_width')
        if (not self.capacities or type(self.prefill_width) is not int or self.prefill_width <= 0
                or self.capacities != sorted(set(self.capacities))
                or any(type(n) is not int or not self.prefill_width <= n <= 8192 for n in self.capacities)):
            raise ValueError('Invalid VoxCPM2 context capacities')
        self.stages = {stage['role']: stage for stage in self.spec['assets']}
        self.base_roles = [stage['role'] for stage in self.spec['assets'] if stage['role'].startswith('base-')]
        bounds = [0] + [self.stages[role]['end'] for role in self.base_roles]
        if (list(self.stages) != self.base_roles + ['residual', 'encoder', 'dit']
                or [self.stages[role]['start'] for role in self.base_roles] != bounds[:-1]
                or bounds[-1] != self.config['lm_config']['num_hidden_layers']):
            raise ValueError('VoxCPM2 bundle is missing transformer stages')
        if (set(self.stages['encoder'].get('local_functions', {})) != {'main', 'batch'}
                or set(self.stages['dit'].get('local_functions', {})) != {'pair'}
                or 1 not in {row['steps'] for row in self.stages['dit'].get('sampler_functions', [])}):
            raise ValueError('VoxCPM2 bundle predates fused local transformers; re-export the bundle')
        self._masks = {}
        self.tokenizer = LlamaTokenizerFast.from_pretrained(weights)
        # Bundles store these FP16 tables and the folded AudioVAE weights as the runtime uses them,
        # so both stay memory-mapped from disk rather than copied.
        with safe_open(weights / 'model.safetensors', framework='pt') as tensors:
            self.cpu = {key: tensors.get_tensor(key).half() for key in tensors.keys()}
        with torch.device('meta'):
            codec = fold_weight_norm(AudioVAE(AudioVAEConfig(**self.config['audio_vae_config'])))
        with safe_open(weights / 'audiovae.safetensors', framework='pt') as tensors:
            codec.load_state_dict({key: tensors.get_tensor(key) for key in tensors.keys()}, strict=True, assign=True)
        self.codec = codec.eval().requires_grad_(False)
        self.sampling_rate = self.codec.out_sample_rate
        self.capabilities = dict(CAPABILITIES, parameters=PARAMETERS, sampling_rate=self.sampling_rate,
                                 context_positions=max(self.capacities), streaming=True)
        cfg = self.config['lm_config']
        dim = cfg['kv_channels']
        inverse = 1 / (cfg['rope_theta'] ** (torch.arange(0, dim, 2).float()/dim))
        # The pinned source initializes its table at max_position_embeddings.
        scaling = cfg['rope_scaling']
        factors = scaling['long_factor'] if cfg['max_position_embeddings'] > scaling['original_max_position_embeddings'] else scaling['short_factor']
        angle = torch.outer(torch.arange(max(self.capacities)).float(), 1/torch.tensor(factors)) * inverse
        angle = torch.cat((angle, angle), dim=1)
        multiplier = math.sqrt(1 + math.log(cfg['max_position_embeddings']/scaling['original_max_position_embeddings'])
                               / math.log(scaling['original_max_position_embeddings']))
        self.cos = (angle.cos()*multiplier).half().numpy()
        self.sin = (angle.sin()*multiplier).half().numpy()
        self._lock = threading.RLock()
        self.ane_seconds = 0.0
        self.hosts = {}
        try:
            for stage in self.spec['assets']:
                cfg = stage['config']
                cached = None
                if stage['cached']:
                    cached = dict(hidden=cfg['hidden'], channels=2*(stage['end']-stage['start'])*cfg['kv_heads']*cfg['head_dim'],
                                  functions=stage['cache_functions'])
                    expected = {(capacity, width) for capacity in self.capacities for width in (self.prefill_width, 1)}
                    if {(row['capacity'], row['width']) for row in cached['functions']} != expected:
                        raise ValueError('VoxCPM2 bundle is missing cache tier functions')
                self.hosts[stage['role']] = TensorHost(asset/stage['asset'], output='features', cached=cached)
        except BaseException:
            self.close()
            raise

    def _linear(self, x, name):
        return F.linear(x, self.cpu[name+'.weight'], self.cpu.get(name+'.bias'))

    def _run(self, role, x, *, state=None):
        stage, host = self.stages[role], self.hosts[role]
        cfg = stage['config']
        if stage['cached']:
            if state is None:
                host.reset()
                state = dict(position=0)
            if state['position']+x.shape[1] > max(self.capacities):
                raise ValueError('VoxCPM2 context exceeds the compiled cache limit')
            parts = []
            for offset in range(0, x.shape[1], self.prefill_width):
                chunk = x[:, offset:offset+self.prefill_width]
                count = chunk.shape[1]
                width = 1 if count == 1 else self.prefill_width
                capacity = next(c for c in self.capacities if c >= state['position']+count)
                values = dict(hidden=np.pad(chunk.transpose(1, 2).unsqueeze(2).numpy(),
                    ((0, 0), (0, 0), (0, 0), (0, width-count))))
                values['cos'] = np.ones((1, 1, cfg['head_dim'], width), np.float16)
                values['sin'] = np.zeros_like(values['cos'])
                if cfg['rope']:
                    positions = slice(state['position'], state['position']+count)
                    values['cos'][..., :count] = self.cos[positions].T[None, None]
                    values['sin'][..., :count] = self.sin[positions].T[None, None]
                function = next(row['name'] for row in stage['cache_functions']
                                if row['capacity'] == capacity and row['width'] == width)
                started = time.perf_counter()
                output = host.step(function, values, count)
                self.ane_seconds += time.perf_counter()-started
                parts.append(torch.from_numpy(output[:, :, 0].copy()).transpose(1, 2))
                state['position'] += count
            return torch.cat(parts, dim=1), state
        raise ValueError(f'{role} is an uncached stage')

    def _rows(self, width, count, length):
        """Attention inputs for `count` independent rows of `length` packed into `width`."""
        key = width, count, length
        if key not in self._masks:
            # Each row attends only itself; padding attends itself and is discarded.
            mask = np.full((width, width), -np.inf, np.float16)
            for start in range(0, count*length, length):
                mask[start:start+length, start:start+length] = 0
            np.fill_diagonal(mask, 0)
            positions = np.arange(width) % length
            self._masks[key] = dict(cos=self.cos[positions].T[None, None], sin=self.sin[positions].T[None, None],
                                    mask=mask[None, None])
        return self._masks[key]

    def _local(self, role, function, rows):
        """Run independent equal-length sequences of an uncached stage in one call."""
        spec = self.stages[role]['local_functions'][function]
        count, length, hidden = rows.shape
        width = spec['width']
        if count*length > width:
            raise ValueError(f'{role} rows exceed the {function} width')
        values = dict(self._rows(width, count, length), hidden=np.zeros((1, hidden, 1, width), np.float16))
        values['hidden'][0, :, 0, :count*length] = rows.reshape(-1, hidden).T.numpy()
        started = time.perf_counter()
        output = self.hosts[role].run(spec['name'], values)
        self.ane_seconds += time.perf_counter()-started
        return torch.from_numpy(output[0, :, 0, :count*length].T.copy()).reshape(count, length, hidden)

    def _lm(self, role, x, states=None):
        roles = self.base_roles if role == 'base' else ['residual']
        new = []
        for i, name in enumerate(roles):
            x, state = self._run(name, x, state=states[i] if states else None)
            new.append(state)
        return x, new

    def _encode(self, patches):
        # Encode each distinct patch once; text positions all share the empty patch.
        unique, index = torch.unique(patches.reshape(len(patches), -1), dim=0, return_inverse=True)
        x = self._linear(unique.reshape(-1, *patches.shape[1:]), 'feat_encoder.in_proj')
        cls = self.cpu['feat_encoder.special_token'].reshape(1, 1, -1).expand(len(x), 1, -1)
        rows = torch.cat((cls, x), dim=1)
        width = self.stages['encoder']['local_functions']['batch']['width'] // rows.shape[1]
        function = 'main' if len(rows) == 1 else 'batch'
        out = torch.cat([self._local('encoder', function, rows[i:i+width])[:, 0] for i in range(0, len(rows), width)])
        return self._linear(out, 'enc_to_lm_proj')[index].unsqueeze(0)

    def _quantize(self, x):
        x = self._linear(x, 'fsq_layer.in_proj').tanh()
        x = (x*self.config['scalar_quantization_scale']).round()/self.config['scalar_quantization_scale']
        return self._linear(x, 'fsq_layer.out_proj')

    def _clock(self, t):
        """Diffusion time embedding of a scalar time, with upstream's zero step-size embedding."""
        prefix = 'feat_decoder.estimator.'
        width = self.config['dit_config']['hidden_dim']
        def embed(value, name):
            inverse = torch.exp(torch.arange(width//2, dtype=torch.float16) * (-math.log(10000)/(width//2-1)))
            phase = 1000*value.reshape(1, 1)*inverse.reshape(1, -1)
            embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
            return self._linear(F.silu(self._linear(embedding, prefix+name+'.linear_1')), prefix+name+'.linear_2')
        return embed(t, 'time_mlp') + embed(torch.zeros_like(t), 'delta_time_mlp')

    def _estimate(self, x, mu, clock, cond):
        """Velocities of up to two independent rows, evaluated in one packed call."""
        prefix = 'feat_decoder.estimator.'
        width = self.config['dit_config']['hidden_dim']
        rows = torch.cat((mu.reshape(-1, 2, width), clock.reshape(-1, 1, width),
                          self._linear(cond.transpose(1, 2), prefix+'cond_proj'),
                          self._linear(x.transpose(1, 2), prefix+'in_proj')), dim=1)
        out = self._local('dit', 'pair', rows)[:, -4:]
        return self._linear(out, prefix+'out_proj').transpose(1, 2).contiguous()

    def _schedule(self, options):
        """Fused Euler calls shared by every patch: (function, time embeddings, step sizes).

        Upstream's initial zero-velocity steps leave the sample unchanged and are skipped."""
        span = torch.linspace(1, 0, options['num_step']+1, dtype=torch.float16)
        span = span + (torch.cos(torch.pi/2*span)-1+span)
        t, dt = span[0], span[0]-span[1]
        zero_steps = max(1, int(len(span)*0.04))
        steps = []
        for step in range(1, len(span)):
            if step > zero_steps:
                steps.append((dt, self._clock(t)))
            t = t-dt
            if step < len(span)-1:
                dt = t-span[step+1]
        functions = sorted(self.stages['dit']['sampler_functions'], key=lambda row: -row['steps'])
        calls = []
        while steps:
            row = next(row for row in functions if row['steps'] <= len(steps))
            part, steps = steps[:row['steps']], steps[row['steps']:]
            calls.append((row['name'], torch.cat([clock for _, clock in part]).T[None, :, None].numpy(),
                          torch.stack([dt for dt, _ in part]).reshape(1, 1, 1, -1).numpy()))
        return calls

    def _sample(self, mu, cond, options, schedule, generator):
        x = torch.randn((1, 64, 4), dtype=torch.float16, generator=generator)
        width = self.config['dit_config']['hidden_dim']
        # Guided then unguided rows of [mu, time, condition, noisy frames]; they differ only in mu.
        mu, condition = mu.reshape(2, width), self._linear(cond.transpose(1, 2), 'feat_decoder.estimator.cond_proj')[0]
        context = torch.cat((mu, condition, torch.zeros_like(mu), condition)).T[None, :, None].numpy()
        values = dict(self._rows(self.stages['dit']['local_functions']['pair']['width'], 2, 11), context=context,
                      guidance=np.full((1, 1, 1, 1), options['guidance_scale'], np.float16), x=x[:, :, None].numpy())
        for function, clocks, dts in schedule:
            started = time.perf_counter()
            values['x'] = self.hosts['dit'].run(function, dict(values, clocks=clocks, dts=dts))
            self.ane_seconds += time.perf_counter()-started
        return torch.from_numpy(values['x'][:, :, 0].copy()).transpose(1, 2).contiguous()

    def _stream(self, text, voice, options, generator):
        """Yield the utterance's waveform in order as it is decoded; the pieces join into the whole."""
        # Upstream's stateful causal VAE decodes 16-frame (four-patch) chunks on
        # a worker thread, overlapped with generation. Continuation context is
        # decoded for continuity and then dropped.
        with self.codec.streaming_decode() as decoder, ThreadPoolExecutor(1) as worker:
            chunks, pending, skip = deque(), torch.zeros(1, 64, 0), 0
            def decode(chunk):
                with torch.inference_mode():
                    return decoder.decode_chunk(chunk)[0, 0]
            def decoded(wait):
                nonlocal skip
                while chunks and (wait or chunks[0].done()):
                    piece = chunks.popleft().result()
                    piece, skip = piece[skip:], max(0, skip-len(piece))
                    if not torch.isfinite(piece).all():
                        raise RuntimeError('VoxCPM2 generated non-finite audio')
                    if len(piece):
                        yield piece.numpy()
            for patches, audible, final in self._generate(text, voice, options, generator):
                if not audible:
                    skip += patches.shape[1]*4*self.codec.decode_chunk_size
                pending = torch.cat((pending, patches.reshape(1, -1, 64).transpose(1, 2).float()), dim=-1)
                while pending.shape[-1] >= 16 or (final and pending.shape[-1]):
                    chunks.append(worker.submit(decode, pending[..., :16]))
                    pending = pending[..., 16:]
                yield from decoded(wait=False)
            yield from decoded(wait=True)

    def _generate(self, text, voice, options, generator):
        """Yield (patches, audible, final): any continuation context, then each generated patch."""
        ids, features, audio = build_inputs(self.tokenizer, text, voice, context_positions=max(self.capacities))
        available = max(self.capacities)-len(ids)
        budget = min(options['max_decode_steps'], available)
        if budget <= options['min_decode_steps']+1:
            raise ValueError(f'VoxCPM2 prompt leaves too little audio context; bundle limit is {max(self.capacities)} positions')
        embedded = self._encode(features.half())
        text = F.embedding(ids, self.cpu['base_lm.embed_tokens.weight']).unsqueeze(0)
        mask = audio.reshape(1, -1, 1)
        combined = torch.where(mask, embedded, text)
        base, base_states = self._lm('base', combined)
        base = torch.where(mask, self._quantize(base), base)
        residual_input = self._linear(torch.cat((base, torch.where(mask, embedded, torch.zeros_like(embedded))), dim=-1), 'fusion_concat_proj')
        residual, residual_states = self._lm('residual', residual_input)
        hidden, residual = base[:, -1:], residual[:, -1:]
        previous = features[-1:].half().transpose(1, 2)
        if bool(audio[-1]):
            yield features[audio][-3:].unsqueeze(0).half(), False, False
        schedule = self._schedule(options)
        for step in range(budget):
            mu = torch.cat((self._linear(hidden, 'lm_to_dit_proj'), self._linear(residual, 'res_to_dit_proj')), dim=-1).squeeze(1)
            patch = self._sample(mu, previous, options, schedule, generator)
            stop = self._linear(F.silu(self._linear(hidden, 'stop_proj')), 'stop_head')
            if step > options['min_decode_steps'] and stop[0, 0].argmax().item() == 1:
                yield patch, True, True
                return
            if step+1 == budget:
                if available <= options['max_decode_steps']:
                    raise ValueError(f'VoxCPM2 ran out of audio context after {budget*0.16:.1f} seconds: '
                                     f'the text and reference use {len(ids)} of {max(self.capacities)} positions. '
                                     'Use a shorter reference, choose Voice clone instead of Clone and continue, '
                                     'or use a bundle with more context.')
                raise ValueError(f'VoxCPM2 reached Maximum audio patches ({budget}, {budget*0.16:.1f} seconds) '
                                 'before finishing. Increase that generation setting.')
            yield patch, True, False
            current = self._encode(patch)
            hidden, base_states = self._lm('base', current, base_states)
            hidden = self._quantize(hidden)
            residual_input = self._linear(torch.cat((hidden, current), dim=-1), 'fusion_concat_proj')
            residual, residual_states = self._lm('residual', residual_input, residual_states)
            previous = patch.transpose(1, 2)

    @torch.inference_mode()
    def generate(self, text, **arguments):
        with self._lock:
            utterances, options, generator = self._prepare(text, **arguments)
            self.ane_seconds = 0
            return [np.concatenate(list(self._stream(item, voice, options, generator))) for item, voice in utterances]

    @torch.inference_mode()
    def stream(self, text, **arguments):
        """Yield one text's waveform in order while it is generated; the pieces join into `generate`'s result."""
        if not isinstance(text, str):
            raise ValueError('Stream one text at a time.')
        with self._lock:
            utterances, options, generator = self._prepare(text, **arguments)
            self.ane_seconds = 0
            yield from self._stream(*utterances[0], options, generator)

    def _prepare(self, text, *, language=None, instruct=None, mode=None, controls=None,
                 voice_clone_prompt=None, ref_audio=None, ref_text=None, seed=None,
                 duration=None, speed=None, **parameters):
        """Validate a request into each text's (prompt, voice), the options and the shared noise generator."""
        if duration is not None or speed is not None:
            raise ValueError('VoxCPM2 supports descriptive delivery instructions, not numeric duration/speed.')
        options = {p['key']: p['default'] for p in PARAMETERS}
        options.update(validate(parameters))
        if options['max_decode_steps'] <= options['min_decode_steps']+1:
            raise ValueError('Maximum audio patches must exceed the minimum by at least two.')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError('Seed must be an integer between 0 and 2^63 - 1.')
        texts = text if isinstance(text, list) else [text]
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Speech text must not be empty.')
        def batch(value, name):
            if isinstance(value, list):
                if len(value) != len(texts):
                    raise ValueError(f'{name} must have one value per text')
                return value
            return [value]*len(texts)
        modes, instructions, fields, voices, languages = [batch(v, k) for v, k in (
            (mode, 'mode'), (instruct, 'instruct'), (controls, 'controls'),
            (voice_clone_prompt, 'voice_clone_prompt'), (language, 'language'))]
        if ref_audio is not None:
            if voice_clone_prompt is not None:
                raise ValueError('Choose reference audio or a saved voice prompt.')
            voices = [self.create_voice_clone_prompt(ref_audio, ref_text,
                       options['preprocess_prompt'], mode=m) for m in modes]
        utterances = []
        for item, instruction, selected, fields, voice, lang in zip(texts, instructions, modes, fields, voices, languages):
            if lang is not None:
                raise ValueError('VoxCPM2 detects language from the text; omit language.')
            if voice is not None and not isinstance(voice, VoiceClonePrompt):
                raise ValueError('Use a VoxCPM2 saved voice prompt.')
            if selected is None and voice is not None:
                selected = voice.mode
            selected, _ = speech_controls(self.capabilities, selected, instruction, fields, voice is not None)
            if voice is not None and selected['id'] in ('clone', 'continuation', 'clone_continuation') and voice.mode != selected['id']:
                raise ValueError('Saved voice conditioning differs from the selected mode; prepare the voice in that mode.')
            item = ' '.join(item.split())
            if instruction:
                item = f'({instruction.strip()})'+item
            utterances.append((item, voice))
        generator = torch.Generator().manual_seed(seed if seed is not None else secrets.randbits(63))
        return utterances, options, generator

    @torch.inference_mode()
    def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=False, *, mode=None):
        import torchaudio.functional as audio

        mode = 'clone' if mode in (None, 'design') else mode
        if mode not in ('clone', 'continuation', 'clone_continuation'):
            raise ValueError('Choose a reference-based VoxCPM2 mode.')
        with self._lock:
            if mode != 'clone' and not (ref_text and ref_text.strip()):
                ref_text = self.transcribe(ref_audio)
            waveform, rate = read_audio(ref_audio) if isinstance(ref_audio, (str, Path)) else ref_audio
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if (waveform.ndim != 2 or not waveform.numel() or not torch.isfinite(waveform).all()
                    or type(rate) is not int or rate <= 0):
                raise ValueError('Reference audio must contain finite samples.')
            waveform = audio.resample(waveform.mean(0, keepdim=True), rate, self.codec.sample_rate)
            if preprocess_prompt:
                waveform = trim_reference(waveform, self.codec.sample_rate)
            alignment = 4*self.codec.chunk_size
            count = math.ceil(waveform.shape[-1]/alignment)
            if count+4 >= max(self.capacities):
                raise ValueError('Reference audio exceeds the compiled VoxCPM2 context limit.')
            pad = (-waveform.shape[-1]) % alignment
            def encode(left):
                data = F.pad(waveform, (pad, 0) if left else (0, pad))
                return self.codec.encode(data, self.codec.sample_rate)[0].reshape(64, -1, 4).permute(1, 2, 0).contiguous()
            reference = encode(False) if mode != 'continuation' else None
            continuation = encode(True) if mode != 'clone' else None
            return VoiceClonePrompt(reference, continuation, ref_text.strip() if ref_text else '', mode)

    generation_parameters = staticmethod(validate)
    load_voice_clone_prompt = staticmethod(VoiceClonePrompt.load)

    def transcribe(self, audio, *, locale=None):
        return transcribe(audio, locale=locale)

    def close(self):
        for host in self.hosts.values():
            host.close()
        self.hosts.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
