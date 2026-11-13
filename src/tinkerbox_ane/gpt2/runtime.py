"""GPT-2 embeddings and tied vocabulary readout around the native cached ANE host."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from tinkerbox_ane.architectures.gpt2 import Config
from tinkerbox_ane.bundle import bundled_checkpoint, read_bundle
from tinkerbox_ane.host import TensorHost, check_architecture
from . import KIND


class Decoder:
    def __init__(self, asset):
        from transformers import GPT2TokenizerFast

        asset = Path(asset)
        spec = read_bundle(asset, KIND)
        check_architecture(spec)
        self.model_id = (
            spec["model"] + {"float16": "", "int8-per-channel": "-int8", "lut8-group32": "-lut8"}[spec["weights"]]
        )
        weights = bundled_checkpoint(asset)
        cfg = Config.from_dict(json.loads((weights / "config.json").read_text()))
        self.width = spec["prefill_width"]
        self.context = max(spec["capacities"])
        self.batch_sizes = tuple(spec["batch_sizes"])
        functions = {}
        for stage in spec["assets"]:
            for function in stage["cache_functions"]:
                key = function["batch"], function["capacity"], function["width"]
                functions.setdefault(key, []).append(function["name"])
        self.functions = functions
        self.tokenizer = GPT2TokenizerFast.from_pretrained(weights, local_files_only=True)
        with safe_open(weights / "model.safetensors", framework="pt") as tensors:
            self.embedding = tensors.get_tensor("wte.weight")
            self.position_embedding = tensors.get_tensor("wpe.weight")
        # Keep the readout in FP32 so close vocabulary scores survive selection.
        self.readout = self.embedding.float()
        self.position = self._rows = self._batch = 0
        self._lengths = self._padding = None
        self.hosts = []
        try:
            for index, stage in enumerate(spec["assets"]):
                self.hosts.append(
                    TensorHost(
                        asset / stage["asset"],
                        output="features",
                        cached=dict(
                            hidden=cfg.hidden,
                            channels=2 * (stage["end"] - stage["start"]) * cfg.hidden,
                            functions=stage["cache_functions"],
                            last_only=index == len(spec["assets"]) - 1,
                        ),
                    )
                )
        except BaseException:
            self.close()
            raise

    def _run(self, functions, values, count):
        for index, (host, function) in enumerate(zip(self.hosts, functions)):
            output = host.step(function, values, count)
            if index + 1 < len(self.hosts):
                values["hidden"][..., :count] = output
        return output

    def _logits(self, hidden):
        hidden = hidden.float()
        # Accelerate's skinny GEMM costs more than a few GEMVs for partial
        # four-sequence groups. Keep GEMM's weight reuse for full groups.
        if len(hidden) < 4:
            return torch.stack([torch.mv(self.readout, row) for row in hidden]).numpy()
        return (hidden @ self.readout.T).numpy()

    def _packed(self, ids, positions):
        count = len(ids)
        width = self._batch if count == self._batch else self.width
        capacity = min(
            c for b, c, w in self.functions if b == self._batch and w == width and c >= self.position + count
        )
        hidden = self.embedding[ids] + self.position_embedding[positions]
        values = np.zeros((1, hidden.shape[1], 1, width), dtype=np.float16)
        values[0, :, 0, :count] = hidden.T.numpy()
        output = self._run(
            self.functions[self._batch, capacity, width], dict(hidden=values, _padding=self._padding), count
        )
        self.position += count
        return torch.from_numpy(output[0, :, 0].T.copy())

    def start(self, prompts):
        """Prefill left-padded prompts into fresh caches; returns next-token scores per prompt."""
        for host in self.hosts:
            host.reset()
        self.position = 0
        self._rows = len(prompts)
        self._batch = next(b for b in self.batch_sizes if b >= self._rows)
        prompts = prompts + [[self.tokenizer.bos_token_id]] * (self._batch - len(prompts))
        self._lengths = np.array([len(p) for p in prompts])
        longest = int(self._lengths.max())
        self._padding = longest - self._lengths
        ids = np.full((longest, self._batch), self.tokenizer.bos_token_id, dtype=np.int64)
        for row, prompt in enumerate(prompts):
            ids[self._padding[row] :, row] = prompt
        positions = np.maximum(0, np.arange(longest)[:, None] - self._padding)
        ids, positions = ids.reshape(-1), positions.reshape(-1)
        with torch.inference_mode():
            for offset in range(0, len(ids), self.width):
                hidden = self._packed(ids[offset : offset + self.width], positions[offset : offset + self.width])
            return self._logits(hidden[: self._rows])

    def step(self, tokens):
        tokens = tokens + [self.tokenizer.eos_token_id] * (self._batch - len(tokens))
        with torch.inference_mode():
            hidden = self._packed(tokens, self._lengths)
            self._lengths += 1
            return self._logits(hidden[: self._rows])

    def close(self):
        for host in self.hosts:
            host.close()
