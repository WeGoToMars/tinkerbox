"""Synchronous System One API backed by the native Core AI host."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import warnings

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from tinkerbox_ane.bundle import bundled_checkpoint, checkpoint_modules, read_bundle
from tinkerbox_ane.d3 import CHUNK, CODE, KIND, MAX_IMAGE_TOKENS, RESIDUAL_SCALE
from tinkerbox_ane.d3.model import CheckpointReader, Config
from tinkerbox_ane.d3.vision import IMAGE_TOKEN_ID, VisionPipeline
from tinkerbox_ane.host import Host, check_flops


class SequentialQuestionsWarning(UserWarning):
    """Several questions in one call run one after another, each reading its whole prompt."""


def check_layout(spec: dict) -> None:
    """The capacities, dimensions and stages the staged host allocates state for."""
    width, context, tiers = spec["width"], spec["context"], spec["tiers"]
    if (width <= 0 or context < width or context % width or not tiers
            or tiers != sorted(set(tiers)) or tiers[-1] != context
            or any(t < width or t > context or t % width for t in tiers)):
        raise ValueError("invalid attention capacities")
    prompts = spec.get("prompts", [])
    if prompts != sorted(set(prompts)) or any(type(w) is not int or w <= 0 or w % 32 or w > context
                                              for w in prompts):
        raise ValueError("invalid prompt widths")
    for name in ("hidden", "heads", "kv_heads", "head_dim", "conv_taps"):
        if not isinstance(spec[name], int) or spec[name] <= 0:
            raise ValueError(f"invalid {name}")
    if spec["heads"] % spec["kv_heads"] or spec["conv_taps"] < 2:
        raise ValueError("inconsistent model dimensions")
    if (not spec["stages"] or any(type(n) is not int or n <= 0 for n in spec["stages"])
            or len(spec["assets"]) != len(spec["stages"])
            or spec.get("attention_mode") != "noncausal_full_attention"
            or type(spec.get("rotary_dim")) is not int or not 0 < spec["rotary_dim"] <= spec["head_dim"]
            or spec["rotary_dim"] % 2):
        raise ValueError("invalid Qwen3.5 stages or attention configuration")
    counted = [(asset.get("flops"), asset["functions"]) for asset in spec["assets"]]
    if "vision_asset" in spec:
        counted.append((spec.get("vision_flops"), spec["vision_functions"]))
    check_flops(counted)


class Embeddings:
    def __init__(self, checkpoint: Path):
        self.table = CheckpointReader(checkpoint).get_tensor("language_model.embed_tokens.weight")

    @torch.inference_mode()
    def chunks(self, ids: list[int], image_features: list[torch.Tensor] | None = None):
        """Zero-padded (hidden, CHUNK) FP16 embeddings, one chunk at a time; image placeholder
        tokens take the image features in order."""
        features = torch.cat(image_features) if image_features else None
        used = 0
        for start in range(0, len(ids), CHUNK):
            part = ids[start:start + CHUNK]
            rows = torch.zeros(CHUNK, self.table.shape[1])
            rows[:len(part)] = self.table[part].float() * RESIDUAL_SCALE
            if features is not None:
                slots = (torch.tensor(part) == IMAGE_TOKEN_ID).nonzero().flatten()
                rows[slots] = features[used:used + len(slots)]
                used += len(slots)
            yield rows.T.half().numpy()


def rope_positions(ids: list[int], grids, merge: int) -> torch.Tensor:
    """Temporal, row and column positions of each token, as the official multimodal RoPE index:
    an image's tokens span its merged grid, and the text after it resumes past its larger side."""
    positions = torch.empty(3, len(ids), dtype=torch.long)
    grids, offset, i = iter(grids), 0, 0
    while i < len(ids):
        if ids[i] != IMAGE_TOKEN_ID:
            positions[:, i] = offset
            offset, i = offset + 1, i + 1
            continue
        _, rows, cols = (int(n) // f for n, f in zip(next(grids), (1, merge, merge)))
        count = rows * cols
        if ids[i:i + count] != [IMAGE_TOKEN_ID] * count:
            raise ValueError("image tokens do not match the image grid")
        positions[0, i:i + count] = offset
        positions[1, i:i + count] = torch.arange(rows).repeat_interleave(cols) + offset
        positions[2, i:i + count] = torch.arange(cols).repeat(rows) + offset
        offset, i = offset + max(rows, cols), i + count
    return positions


def load(asset: Path):
    spec = read_bundle(asset, KIND)
    source = spec["source"]
    path = bundled_checkpoint(asset, CODE)
    package = f"d3_remote_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]}"
    fmt, rt = checkpoint_modules(path, package, "d3_format", "d3_runtime")

    class Model:
        """Typed decisions over text and images. The full-attention layers see the whole prompt,
        so each question runs its own pass; several questions or requests run in order."""

        def __init__(self):
            self._lock = threading.Lock()
            self._closed = False
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._embeddings = Embeddings(path)
            with safe_open(path / "readout.safetensors", "pt") as f:
                self._readout = f.get_tensor("weight").float()  # [options, hidden]
            decision = json.loads((path / "decision_config.json").read_text())
            self._codes, self._temperature = decision["codes"], decision["temperature"]
            self._config = Config.from_hf(path / "config.json")
            self._vision = None
            self._max_image_tokens = MAX_IMAGE_TOKENS
            self._rope_custom = False
            self._host = Host(asset, KIND, check_layout)
            self._flops = {name: count for entry in spec["assets"] for name, count in entry["flops"].items()}
            self._flops.update(spec.get("vision_flops", {}))
            self.context, self.chunk_width = self._host.context, self._host.width
            try:
                self._set_rope(torch.arange(self.context).expand(3, -1))
            except BaseException:
                self.close()
                raise

        def _get_vision(self) -> VisionPipeline:
            if not spec.get("vision_asset"):
                raise ValueError("this bundle has no vision tower; image inputs are unavailable")
            if self._vision is None:
                self._vision = VisionPipeline(path, self._host, [int(name[1:]) for name in spec["vision_functions"]])
                self._vision.limit_tokens(self._max_image_tokens)
            return self._vision

        @property
        def max_image_tokens(self) -> int:
            """Most tokens per image, from 64 to 1,600; pictures keep their shape within this budget."""
            return self._max_image_tokens

        @max_image_tokens.setter
        def max_image_tokens(self, tokens: int):
            if isinstance(tokens, bool) or not isinstance(tokens, int) or not 64 <= tokens <= MAX_IMAGE_TOKENS:
                raise ValueError(f"max_image_tokens must be an integer from 64 to {MAX_IMAGE_TOKENS}, not {tokens!r}")
            with self._lock:
                self._max_image_tokens = tokens
                if self._vision is not None:
                    self._vision.limit_tokens(tokens)

        def _set_rope(self, positions: torch.Tensor):
            """Upload rotary tables for `positions` (temporal, row, column), padded to the context."""
            cfg = self._config
            if positions.shape[1] < self.context:
                tail = torch.arange(positions.shape[1], self.context).expand(3, -1)
                positions = torch.cat([positions, tail + positions.max() + 1 - positions.shape[1]], dim=1)
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2).float() / cfg.rotary_dim))
            angles = positions[:, :, None].float() * inv[None, None, :]
            # Interleaved multimodal sections: row and column angles replace every third frequency.
            mixed = angles[0].clone()
            for axis in (1, 2):
                indices = slice(axis, cfg.mrope_section[axis] * 3, 3)
                mixed[:, indices] = angles[axis, :, indices]
            self._host.set_rope(mixed.cos().T.half().numpy(), mixed.sin().T.half().numpy())

        def tokens(self, state, question) -> list[int]:
            """Prompt token ids of a text-only question."""
            text = fmt.render(self._tokenizer, state, rt.normalize_question(question).rendered, self._codes)
            return self._tokenizer.encode(text, add_special_tokens=False)

        def _plan(self, state, questions, images):
            """Each question's normalized form and prompt ids, and the processed images."""
            pictures = [rt.load_image(image) for image in images or ()]
            questions = [rt.normalize_question(question) for question in questions]
            media, rows = None, []
            if pictures and questions:
                vision = self._get_vision()
                for question in questions:
                    text = vision.processor.apply_chat_template(
                        rt.image_messages("d3", state, question.rendered, self._codes, len(pictures)),
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    if text.count("<|image_pad|>") != len(pictures) or "<|video_pad|>" in text:
                        raise ValueError("the state or question contains a literal image or video placeholder token")
                    encoded = vision.processor(text=[text], images=pictures, return_tensors="pt")
                    if media is None:
                        media = encoded
                    rows.append((encoded["input_ids"][0].tolist(), question))
            else:
                for question in questions:
                    text = fmt.render(self._tokenizer, state, question.rendered, self._codes)
                    rows.append((self._tokenizer.encode(text, add_special_tokens=False), question))
            for ids, _ in rows:
                if not 0 < len(ids) <= self.context:
                    raise ValueError(f"prompt needs {len(ids)} tokens; context limit is {self.context}")
            return rows, pictures, media

        def _rope(self, positions: torch.Tensor | None):
            """Use multimodal positions for one prompt, or plain positions when None."""
            if positions is not None or self._rope_custom:
                self._set_rope(torch.arange(self.context).expand(3, -1) if positions is None else positions)
                self._rope_custom = positions is not None

        @torch.inference_mode()
        def _probabilities(self, ids, question, media=None, features=None) -> list[float]:
            self._rope(None if media is None else rope_positions(ids, media["image_grid_thw"], self._vision.merge))
            hidden = self._host.hidden_chunks(self._embeddings.chunks(ids, features), len(ids))
            count = len(question.keys)
            logits = (self._readout[:count] @ torch.from_numpy(hidden).float() / self._temperature).tolist()
            top = max(logits)
            weights = [math.exp(value - top) for value in logits]
            return [weight / sum(weights) for weight in weights]

        def _run(self, requests):
            """Each request's (normalized questions, probabilities, input tokens, Neural Engine time and
            FLOPs), one pass per question."""
            with self._lock:
                if self._closed:
                    raise RuntimeError("the model is closed")
                # Validate every request before starting any native inference.
                plans = [self._plan(*request) for request in requests]
                passes = sum(len(rows) for rows, _, _ in plans)
                if passes > 1:
                    warnings.warn(f"{source['repo']} answers one question per pass; {passes} questions run "
                                  "sequentially, each reading its whole prompt", SequentialQuestionsWarning,
                                  stacklevel=3)
                done = []
                self._host.metrics()  # Start each request's Neural Engine count from zero.
                for rows, _, media in plans:
                    features = None
                    if media is not None:
                        features = self._vision.encode(media["pixel_values"], media["image_grid_thw"])
                    probabilities = [self._probabilities(ids, question, media, features) for ids, question in rows]
                    engine = self._host.metrics()
                    compute = {"neural_engine_ms": 1000 * engine["seconds"],
                               "flops": sum(self._flops[name] * calls for name, calls in engine["calls"].items())}
                    done.append(([question for _, question in rows], probabilities,
                                 sum(len(ids) for ids, _ in rows), compute))
                return done

        def probabilities(self, state, questions, images=None) -> list[list[float]]:
            """Each question's distribution over its options, in prompt order."""
            return self.probabilities_batch([(state, questions, images)])[0]

        def probabilities_batch(self, requests) -> list[list[list[float]]]:
            """probabilities for many (state, questions[, images]) requests."""
            requests = [(r[0], list(r[1]), r[2] if len(r) > 2 else None) for r in requests]
            return [probabilities for _, probabilities, _, _ in self._run(requests)]

        def system_one(self, state, questions, images=None) -> dict:
            """Named questions about one state and its pictures: `{"answers", "usage"}`. `usage` adds the
            request's Neural Engine time and the FLOPs of the functions it ran, counted at export."""
            return self.system_one_batch([(state, questions, images)])[0]

        def system_one_batch(self, requests) -> list[dict]:
            """Many (state, questions[, images]) requests, answered in order."""
            requests = [(r[0], dict(r[1]), r[2] if len(r) > 2 else None) for r in requests]
            done = self._run([(state, list(named.values()), images) for state, named, images in requests])
            return [{"answers": {name: rt.product_answer(question, probabilities)
                                 for name, question, probabilities in zip(named, questions, probs)},
                     "usage": {"input_tokens": read, "output_tokens": 0, **compute}}
                    for (_, named, _), (questions, probs, read, compute) in zip(requests, done)]

        def count_tokens(self, state, questions, images=None) -> dict:
            """Input tokens and vision passes `system_one` would use, without running the model, and how
            each picture is fed to it (see `VisionPipeline.layout`)."""
            with self._lock:
                if self._closed:
                    raise RuntimeError("the model is closed")
                rows, pictures, _ = self._plan(state, list(questions.values()), images)
                layouts = [self._get_vision().layout(picture) for picture in pictures]
                return {"input_tokens": sum(len(ids) for ids, _ in rows),
                        "image_tiles": len(pictures) if rows else 0, "images": layouts}

        def close(self):
            with self._lock:
                if not self._closed:
                    self._closed = True
                    self._host.close()

        def __enter__(self):
            if self._closed:
                raise RuntimeError("the model is closed")
            return self

        def __exit__(self, *_exc):
            self.close()

    return Model()
