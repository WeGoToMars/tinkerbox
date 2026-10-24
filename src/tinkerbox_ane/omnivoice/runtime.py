"""Upstream speech behavior composed with a verified ANE diffusion export."""

from pathlib import Path
from contextlib import nullcontext
import math
import os
import threading
import time

import numpy as np
import torch
from torch import nn
from safetensors import safe_open

from tinkerbox_ane.bundle import bundled_checkpoint, read_bundle
from tinkerbox_ane.audio import read_audio
from tinkerbox_ane.host import TensorHost, device_architecture
from tinkerbox_ane.transcription import transcribe
from . import KIND, REVISION

_GENERATION_LOCK = threading.Lock()  # torch's CPU RNG is shared across model instances.


class Speech:
    def __init__(self, asset, weights=None):
        from omnivoice import OmniVoice, OmniVoiceConfig
        from omnivoice.models.omnivoice import OmniVoiceModelOutput
        from omnivoice.utils.duration import RuleDurationEstimator
        from omnivoice.utils.text import add_punctuation
        from transformers import AutoTokenizer, AutoFeatureExtractor, HiggsAudioV2TokenizerModel

        asset = Path(asset)
        self.spec = read_bundle(asset, KIND)
        if self.spec.get('architecture') != device_architecture():
            raise ValueError('OmniVoice bundle targets a different device architecture')
        if self.spec['source']['revision'] != REVISION:
            raise ValueError('OmniVoice bundle and CPU checkpoint revisions differ')
        if weights is None:
            local = os.environ.get('TINKERBOX_OMNIVOICE_CHECKPOINT')
            weights = Path(local) if local else bundled_checkpoint(asset)
        else:
            weights = Path(weights)
        config = OmniVoiceConfig.from_pretrained(weights)
        capacities = self.spec['capacities']
        if (not capacities or capacities != sorted(set(capacities))
                or any(not isinstance(n, int) or n <= 0 for n in capacities)
                or self.spec.get('batch_size', 1) != 1):
            raise ValueError('invalid OmniVoice bundle capacities or batch size')
        dimensions = dict(hidden=config.llm_config.hidden_size, head_dim=config.llm_config.head_dim,
                          codebooks=config.num_audio_codebook, audio_vocab=config.audio_vocab_size)
        if any(self.spec.get(key) != value for key, value in dimensions.items()):
            raise ValueError('OmniVoice bundle and checkpoint dimensions differ')
        owner = self

        class Embeddings(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config.llm_config
                self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

            def get_input_embeddings(self):
                return self.embed_tokens

        class ANEOmniVoice(OmniVoice):
            def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=True):
                if isinstance(ref_audio, (str, Path)):
                    ref_audio = read_audio(ref_audio)
                return super().create_voice_clone_prompt(ref_audio, ref_text, preprocess_prompt)

            def _preprocess_all(self, **kwargs):
                text = kwargs['text']
                kwargs['text'] = (add_punctuation(text) if isinstance(text, str)
                                  else [add_punctuation(item) for item in text])
                task = super()._preprocess_all(**kwargs)
                durations = self._ensure_list(kwargs.get('duration'), task.batch_size)
                for i in range(task.batch_size):
                    if durations[i] is None and task.ref_audio_tokens[i] is None:
                        # Upstream's short automatic estimate can collapse a
                        # voice to noise, including in F32. Allow three
                        # seconds at normal speed; explicit durations still win.
                        speed = task.speed[i] if task.speed else 1.0
                        minimum = math.ceil(3 * self.audio_tokenizer.config.frame_rate /
                                            (speed if speed and speed > 0 else 1.0))
                        task.target_lens[i] = max(task.target_lens[i], minimum)
                return task

            def forward(self, input_ids, audio_mask, attention_mask=None, **kwargs):
                embedded = self._prepare_embed_inputs(input_ids, audio_mask)
                return OmniVoiceModelOutput(logits=owner.logits(embedded, attention_mask))

        self.engine = ANEOmniVoice(config, llm=Embeddings()).eval()
        # These projections live in the ANE bundle. Retain only lookup tables on the CPU.
        self.engine.audio_heads = nn.Identity()
        with safe_open(weights / 'model.safetensors', framework='pt', device='cpu') as tensors:
            self.engine.llm.embed_tokens.weight.data.copy_(tensors.get_tensor('llm.embed_tokens.weight'))
            self.engine.audio_embeddings.weight.data.copy_(tensors.get_tensor('audio_embeddings.weight'))
        self.engine.text_tokenizer = AutoTokenizer.from_pretrained(weights)
        codec = weights / 'audio_tokenizer'
        self.engine.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(codec, device_map='cpu').eval()
        self.engine.feature_extractor = AutoFeatureExtractor.from_pretrained(codec)
        self.engine.sampling_rate = self.engine.feature_extractor.sampling_rate
        self.engine.duration_estimator = RuleDurationEstimator()
        # Supply the upstream pipeline's ASR contract with native transcription.
        # Keeping it installed also prevents upstream's lazy Whisper download.
        def native_asr(audio):
            source = audio if isinstance(audio, str) else (audio['array'], audio['sampling_rate'])
            return {'text': transcribe(source)}
        self.engine._asr_pipe = native_asr
        self.sampling_rate = self.engine.sampling_rate
        from .parameters import CAPABILITIES, PARAMETERS
        self.capabilities = dict(CAPABILITIES, parameters=PARAMETERS, sampling_rate=self.sampling_rate,
                                 context_positions=max(capacities))
        self._lock = threading.Lock()
        self.ane_seconds = 0.0
        self.host = TensorHost(asset / self.spec['asset'])

    @torch.inference_mode()
    def logits(self, embeddings, attention_mask):
        """Evaluate independent conditioned/unconditioned sequences in input order."""
        batch, length, _ = embeddings.shape
        try:
            capacity = next(n for n in self.spec['capacities'] if n >= length)
        except StopIteration:
            raise ValueError(f'speech context needs {length} positions; bundle limit is {max(self.spec["capacities"])}') from None
        dim = self.spec['head_dim']
        inverse = self.spec['rope_theta'] ** (-np.arange(0, dim, 2, dtype=np.float32) / dim)
        angles = np.arange(capacity, dtype=np.float32)[:, None] * inverse
        angles = np.concatenate([angles, angles], axis=1).T[None, None]
        cos, sin = np.cos(angles), np.sin(angles)
        results = []
        for i in range(batch):
            hidden = np.zeros((1, self.spec['hidden'], 1, capacity), np.float16)
            mask = np.full((1, 1, capacity, capacity), -30000, np.float16)
            # Pad queries attend themselves; real queries can never read padding.
            np.fill_diagonal(mask[0, 0], 0)
            hidden[0, :, 0, :length] = embeddings[i].float().numpy().T
            allowed = attention_mask[i, 0].numpy() if attention_mask is not None else np.ones((length, length), bool)
            mask[0, 0, :length, :length] = np.where(allowed, 0, -30000)
            started = time.perf_counter()
            output = self.host.run(f's{capacity}', dict(hidden=hidden, cos=cos, sin=sin, mask=mask))
            self.ane_seconds += time.perf_counter() - started
            row = output[0, :, 0, :length].T.copy()
            results.append(torch.from_numpy(row).float().reshape(length, self.spec['codebooks'], self.spec['audio_vocab']).permute(1, 0, 2))
        return torch.stack(results)

    def generate(self, text, *, seed=None, mode=None, controls=None, **kwargs):
        from tinkerbox_ane.speech import speech_controls

        size = len(text) if isinstance(text, list) else 1
        def batch(value, name):
            if isinstance(value, list):
                if len(value) != size:
                    raise ValueError(f'{name} must have one value per text')
                return value
            return [value] * size
        modes, fields = batch(mode, 'mode'), batch(controls, 'controls')
        instructions = batch(kwargs.get('instruct'), 'instruct')
        voices = batch(kwargs.get('voice_clone_prompt'), 'voice_clone_prompt')
        references = batch(kwargs.get('ref_audio'), 'ref_audio')
        native = []
        for i in range(size):
            if fields[i] is not None and not isinstance(fields[i], dict):
                raise ValueError('Speech controls must be an object.')
            supplied = dict(fields[i] or {})
            for key in ('duration', 'speed'):
                if key in kwargs:
                    value = batch(kwargs[key], key)[i]
                    if value is not None:
                        if key in supplied:
                            raise ValueError(f'Provide {key} once, either directly or in controls.')
                        supplied[key] = value
            _, values = speech_controls(self.capabilities, modes[i], instructions[i], supplied,
                                         voices[i] is not None or references[i] is not None)
            native.append(values)
        for key in ('duration', 'speed'):
            kwargs[key] = [values.get(key) for values in native]
        # A request's random sampling must not advance another caller's RNG state.
        with _GENERATION_LOCK, self._lock, torch.random.fork_rng(devices=[]) if seed is not None else nullcontext():
            if seed is not None:
                torch.manual_seed(seed)
            self.ane_seconds = 0.0
            return self.engine.generate(text=text, **kwargs)

    @staticmethod
    def generation_parameters(parameters):
        from .parameters import validate
        return validate(parameters)

    @staticmethod
    def load_voice_clone_prompt(path):
        from omnivoice import VoiceClonePrompt
        return VoiceClonePrompt.load(path)

    def create_voice_clone_prompt(self, *args, **kwargs):
        with self._lock:
            return self.engine.create_voice_clone_prompt(*args, **kwargs)

    def supported_language_ids(self):
        return self.engine.supported_language_ids()

    def supported_language_names(self):
        return self.engine.supported_language_names()

    def transcribe(self, audio, *, locale=None):
        with self._lock:
            return transcribe(audio, locale=locale)

    def close(self):
        self.host.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
