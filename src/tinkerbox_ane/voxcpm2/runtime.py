"""Cached ANE MiniCPM generation with native VoxCPM2 conditioning and CPU AudioVAE."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import secrets
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
from .codec import AudioVAE, AudioVAEConfig, trim_reference
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
        if (len(self.capacities) != 1 or self.capacities != sorted(set(self.capacities))
                or any(type(n) is not int or not 4 <= n <= 8192 for n in self.capacities)):
            raise ValueError('Invalid VoxCPM2 context capacities')
        self.base_roles = [f'base-{i}' for i in range(self.config['lm_config']['num_hidden_layers'])]
        expected = self.base_roles + ['residual', 'encoder', 'dit']
        if [stage['role'] for stage in self.spec['assets']] != expected:
            raise ValueError('VoxCPM2 bundle is missing transformer stages')
        self.stages = {stage['role']: stage for stage in self.spec['assets']}
        self.tokenizer = LlamaTokenizerFast.from_pretrained(weights)
        with safe_open(weights / 'model.safetensors', framework='pt') as tensors:
            self.cpu = {key: tensors.get_tensor(key).half() for key in tensors.keys()}
        self.codec = AudioVAE(AudioVAEConfig(**self.config['audio_vae_config'])).eval().requires_grad_(False)
        state = torch.load(weights / 'audiovae.pth', map_location='cpu', weights_only=True)
        self.codec.load_state_dict(state.get('state_dict', state), strict=True)
        self.sampling_rate = self.codec.out_sample_rate
        self.capabilities = dict(CAPABILITIES, parameters=PARAMETERS, sampling_rate=self.sampling_rate,
                                 context_positions=max(self.capacities))
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
                self.hosts[stage['role']] = TensorHost(asset/stage['asset'], output='features')
        except BaseException:
            self.close()
            raise

    def _linear(self, x, name):
        return F.linear(x, self.cpu[name+'.weight'], self.cpu.get(name+'.bias'))

    def _run(self, role, x, *, state=None):
        stage = self.stages[role]
        cfg = stage['config']
        n = x.shape[1]
        if stage['cached']:
            capacity = next((c for c in self.capacities if c >= n), None) if state is None else 1
            if capacity is None:
                raise ValueError(f'VoxCPM2 context needs {n} positions; bundle limit is {max(self.capacities)}')
        else:
            capacity = n
        values = dict(hidden=x.transpose(1, 2).unsqueeze(2).numpy())
        if capacity != n:
            values['hidden'] = np.pad(values['hidden'], ((0, 0), (0, 0), (0, 0), (0, capacity-n)))
        positions = np.arange(capacity) if state is None else np.array([state['position']])
        values['cos'] = self.cos[positions].T[None, None] if cfg['rope'] else np.ones((1, 1, cfg['head_dim'], capacity), np.float16)
        values['sin'] = self.sin[positions].T[None, None] if cfg['rope'] else np.zeros((1, 1, cfg['head_dim'], capacity), np.float16)
        if state is not None:
            mask = np.arange(max(self.capacities)+1) < state['position']
            mask[-1] = True  # current key is appended after the cache capacity
            values.update(cache=state['cache'], mask=np.where(mask, 0, -np.inf).astype(np.float16)[None, None, None])
            function = stage['functions'][-1]
        else:
            allowed = np.arange(capacity)[None, :] <= np.arange(capacity)[:, None] if stage['cached'] else np.ones((capacity, capacity), bool)
            values['mask'] = np.where(allowed, 0, -np.inf).astype(np.float16)[None, None]
            function = stage['functions'][0]
        started = time.perf_counter()
        output = self.hosts[role].run(function, values)
        self.ane_seconds += time.perf_counter()-started
        hidden = torch.from_numpy(output[:, :cfg['hidden'], 0, :n].copy()).transpose(1, 2)
        if not stage['cached']:
            return hidden
        present = output[:, cfg['hidden']:, :, :n]
        if state is None:
            cache = np.zeros((*present.shape[:-1], max(self.capacities)), np.float16)
            cache[..., :n] = present
            state = dict(position=n, cache=cache)
        else:
            state['cache'][..., state['position']:state['position']+1] = present
            state['position'] += 1
        return hidden, state

    def _lm(self, role, x, states=None):
        roles = self.base_roles if role == 'base' else ['residual']
        new = []
        for i, name in enumerate(roles):
            x, state = self._run(name, x, state=states[i] if states else None)
            new.append(state)
        return x, new

    def _encode(self, patches):
        outputs = []
        zero = None
        for patch in patches:
            empty = not bool(patch.any())
            if empty and zero is not None:
                outputs.append(zero)
                continue
            x = self._linear(patch, 'feat_encoder.in_proj').unsqueeze(0)
            cls = self.cpu['feat_encoder.special_token'].reshape(1, 1, -1)
            out = self._run('encoder', torch.cat((cls, x), dim=1))[:, :1]
            out = self._linear(out, 'enc_to_lm_proj')
            outputs.append(out)
            if empty:
                zero = out
        return torch.cat(outputs, dim=1)

    def _quantize(self, x):
        x = self._linear(x, 'fsq_layer.in_proj').tanh()
        x = (x*self.config['scalar_quantization_scale']).round()/self.config['scalar_quantization_scale']
        return self._linear(x, 'fsq_layer.out_proj')

    def _velocity(self, x, mu, t, cond):
        prefix = 'feat_decoder.estimator.'
        width = self.config['dit_config']['hidden_dim']
        def time_embed(value, name):
            inverse = torch.exp(torch.arange(width//2, dtype=torch.float16) * (-math.log(10000)/(width//2-1)))
            phase = 1000*value.reshape(1, 1)*inverse.reshape(1, -1)
            embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
            return self._linear(F.silu(self._linear(embedding, prefix+name+'.linear_1')), prefix+name+'.linear_2')
        clock = time_embed(t, 'time_mlp') + time_embed(torch.zeros_like(t), 'delta_time_mlp')
        hidden = torch.cat((mu.reshape(1, 2, width), clock.unsqueeze(1),
                            self._linear(cond.transpose(1, 2), prefix+'cond_proj'),
                            self._linear(x.transpose(1, 2), prefix+'in_proj')), dim=1)
        out = self._run('dit', hidden)[:, -4:]
        return self._linear(out, prefix+'out_proj').transpose(1, 2).contiguous()

    def _sample(self, mu, cond, options, generator):
        x = torch.randn((1, 64, 4), dtype=torch.float16, generator=generator)
        span = torch.linspace(1, 0, options['num_step']+1, dtype=torch.float16)
        span = span + (torch.cos(torch.pi/2*span)-1+span)
        t, dt = span[0], span[0]-span[1]
        zero_steps = max(1, int(len(span)*0.04))
        for step in range(1, len(span)):
            if step <= zero_steps:
                velocity = torch.zeros_like(x)
            else:
                positive = self._velocity(x, mu, t, cond)
                negative = self._velocity(x, torch.zeros_like(mu), t, cond)
                flat, null = positive.reshape(1, -1), negative.reshape(1, -1)
                scale = (flat*null).sum(1, keepdim=True)/(null.square().sum(1, keepdim=True)+1e-8)
                negative = negative*scale.reshape(1, 1, 1)
                velocity = negative + options['guidance_scale']*(positive-negative)
            x = x-dt*velocity
            t = t-dt
            if step < len(span)-1:
                dt = t-span[step+1]
        return x.transpose(1, 2).contiguous()

    def _utterance(self, text, voice, options, generator):
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
        context = features[audio][-3:].unsqueeze(0).half() if bool(audio[-1]) else None
        patches = []
        for step in range(budget):
            mu = torch.cat((self._linear(hidden, 'lm_to_dit_proj'), self._linear(residual, 'res_to_dit_proj')), dim=-1).squeeze(1)
            patch = self._sample(mu, previous, options, generator)
            patches.append(patch)
            stop = self._linear(F.silu(self._linear(hidden, 'stop_proj')), 'stop_head')
            if step > options['min_decode_steps'] and stop[0, 0].argmax().item() == 1:
                break
            if step+1 == budget:
                if available <= options['max_decode_steps']:
                    raise ValueError(f'VoxCPM2 ran out of audio context after {budget*0.16:.1f} seconds: '
                                     f'the text and reference use {len(ids)} of {max(self.capacities)} positions. '
                                     'Use a shorter reference, choose Voice clone instead of Clone and continue, '
                                     'or use a bundle with more context.')
                raise ValueError(f'VoxCPM2 reached Maximum audio patches ({budget}, {budget*0.16:.1f} seconds) '
                                 'before finishing. Increase that generation setting.')
            current = self._encode(patch)
            hidden, base_states = self._lm('base', current, base_states)
            hidden = self._quantize(hidden)
            residual_input = self._linear(torch.cat((hidden, current), dim=-1), 'fusion_concat_proj')
            residual, residual_states = self._lm('residual', residual_input, residual_states)
            previous = patch.transpose(1, 2)
        generated = torch.stack(patches, dim=1)
        if context is not None:
            generated = torch.cat((context, generated), dim=1)
        latent = generated.reshape(1, -1, 64).transpose(1, 2).float()
        waveform = self.codec.decode(latent)[0, 0]
        if context is not None:
            waveform = waveform[context.shape[1]*4*self.codec.decode_chunk_size:]
        if not torch.isfinite(waveform).all():
            raise RuntimeError('VoxCPM2 generated non-finite audio')
        return waveform.numpy()

    @torch.inference_mode()
    def generate(self, text, *, language=None, instruct=None, mode=None, controls=None,
                 voice_clone_prompt=None, ref_audio=None, ref_text=None, seed=None,
                 duration=None, speed=None, **parameters):
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
        with self._lock:
            if ref_audio is not None:
                if voice_clone_prompt is not None:
                    raise ValueError('Choose reference audio or a saved voice prompt.')
                voices = [self.create_voice_clone_prompt(ref_audio, ref_text,
                           options['preprocess_prompt'], mode=m) for m in modes]
            self.ane_seconds = 0
            output = []
            generator = torch.Generator().manual_seed(seed if seed is not None else secrets.randbits(63))
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
                output.append(self._utterance(item, voice, options, generator))
            return output

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
