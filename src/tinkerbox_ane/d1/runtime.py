"""Synchronous System One API backed by the native Core AI host."""

import json
from pathlib import Path
import threading

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from tinkerbox_ane.bundle import bundled_checkpoint, checkpoint_modules, read_bundle
from tinkerbox_ane.d1 import CHUNK, CODE, KIND, MAX_IMAGE_TILES, RESIDUAL_SCALE
from tinkerbox_ane.d1.vision import IMAGE_TOKEN_ID, VisionPipeline, as_rgb
from tinkerbox_ane.host import Host
from tinkerbox_ane.d1.tree import TreePlan, groups, shared_prefix


class Embeddings:
    def __init__(self, checkpoint: str | Path):
        with safe_open(Path(checkpoint) / "model.safetensors", "pt") as f:
            self.table = f.get_tensor("model.language_model.embed_tokens.weight")

    @torch.inference_mode()
    def chunks(self, ids: list[int], image_features: list[torch.Tensor] | None = None):
        """Zero-padded (hidden, CHUNK) FP16 embeddings, one chunk at a time; image placeholder
        tokens take the image features in order."""
        features = torch.cat(image_features).float() if image_features else None
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

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
        return self.table[token_ids].float() @ hidden.float()


def load(asset):
    path = bundled_checkpoint(asset, CODE)
    prompt, api = checkpoint_modules(path, "d1_remote", "prompt", "api")

    class Model(api.SystemOneApi):
        def __init__(self):
            self._lock = threading.Lock()
            self._closed = False
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._embeddings = Embeddings(path)
            self._vision = None
            self._max_image_tiles = MAX_IMAGE_TILES
            config = json.loads((path / "config.json").read_text())
            self._render_args = (self._tokenizer.bos_token or "", prompt.default_lead(config["model_type"]),
                                 prompt.DEFAULT_STATE_STYLE, prompt.DEFAULT_SYSTEM, "desc")
            self._host = Host(asset, KIND)
            spec = read_bundle(asset, KIND)
            self._flops = {**spec["flops"], **spec.get("vision_flops", {})}
            self.context, self.chunk_width = self._host.context, self._host.width
            try:
                cfg = config["text_config"]
                dim = cfg["hidden_size"] // cfg["num_attention_heads"]
                theta = cfg["rope_parameters"]["rope_theta"]
                inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                angles = torch.arange(self.context, dtype=torch.float32)[:, None] * inv[None]
                angles = torch.cat((angles, angles), dim=-1).T
                cos, sin = angles.cos(), angles.sin()
                self._host.set_rope(cos.half().numpy(), sin.half().numpy())
            except BaseException:
                self.close()
                raise

        def _get_vision(self) -> VisionPipeline:
            if self._vision is None:
                self._vision = VisionPipeline(path, self._host)
                self._vision.limit_tiles(self._max_image_tiles)
            return self._vision

        @property
        def max_image_tiles(self) -> int:
            """Most 512×512 tiles per image, from 1 (never split) to 10; split images add a thumbnail."""
            return self._max_image_tiles

        @max_image_tiles.setter
        def max_image_tiles(self, tiles: int):
            if isinstance(tiles, bool) or not isinstance(tiles, int) or not 1 <= tiles <= 10:
                raise ValueError(f"max_image_tiles must be an integer from 1 to 10, not {tiles!r}")
            with self._lock:
                self._max_image_tiles = tiles
                if self._vision is not None:
                    self._vision.limit_tiles(tiles)

        def tokens(self, state, question):
            text = prompt.render(self._tokenizer, state, prompt.as_question(question), *self._render_args)
            return self._tokenizer.encode(text, add_special_tokens=False)

        @torch.inference_mode()
        def _probabilities(self, ids, question, image_features=None):
            hidden = self._host.hidden_chunks(self._embeddings.chunks(ids, image_features), len(ids))
            # The vocabulary normalizer cancels in the option softmax.
            option_ids = sorted({i for group in prompt.readout_ids(self._tokenizer, question) for i in group})
            logits = self._embeddings.logits(torch.from_numpy(hidden), option_ids)
            return prompt.readout(self._tokenizer, question, dict(zip(option_ids, logits.tolist())))

        @torch.inference_mode()
        def _tree_probabilities(self, trunk, branches, questions, image_features=None):
            result = []
            for indices in groups(trunk, branches, self.context):
                if len(indices) == 1:
                    i = indices[0]
                    result.append(self._probabilities(trunk + branches[i], questions[i], image_features=image_features))
                    continue
                plan = TreePlan(trunk, [branches[i] for i in indices], self.chunk_width,
                                self._host.conv_taps, self.context)
                chunks = self._embeddings.chunks(plan.ids, image_features)
                hidden = self._host.hidden_chunks(chunks, len(plan.ids), plan=plan)
                for i, h in zip(indices, hidden):
                    question = questions[i]
                    option_ids = sorted({t for g in prompt.readout_ids(self._tokenizer, question) for t in g})
                    logits = self._embeddings.logits(torch.from_numpy(h), option_ids)
                    result.append(prompt.readout(self._tokenizer, question, dict(zip(option_ids, logits.tolist()))))
            return result

        def _plan(self, state, questions, images):
            """Token ids for one request: a single row, or a trunk with one branch per question.

            Returns `(ids, branches, questions, pics, tiles)` after checking the context limit."""
            pics = [as_rgb(im) for im in images or ()]
            questions = [prompt.as_question(q) for q in questions]
            if not questions:
                return [], [], questions, pics, 0
            bos, lead, style, system, option_style = self._render_args
            tiles = 0
            if len(questions) == 1:
                if pics:
                    vision = self._get_vision()
                    prefix = prompt.prefix_text(self._tokenizer, state, bos, style, system,
                                                images=vision.image_markup(len(pics)))
                    suffix = prompt.suffix_text(self._tokenizer, questions[0], lead, option_style)
                    inputs = vision.processor(text=[prefix + suffix], images=[list(pics)], return_tensors="pt", add_special_tokens=False)
                    ids, tiles = inputs["input_ids"][0].tolist(), len(inputs["pixel_values"])
                else:
                    ids = self.tokens(state, questions[0])
                if not 0 < len(ids) <= self.context:
                    raise ValueError(f"prompt needs {len(ids)} tokens; context limit is {self.context}")
                return ids, [], questions, pics, tiles
            if pics:
                vision = self._get_vision()
                prefix = prompt.prefix_text(self._tokenizer, state, bos, style, system,
                                            images=vision.image_markup(len(pics)))
                inputs = vision.processor(text=[prefix], images=[list(pics)], return_tensors="pt", add_special_tokens=False)
                trunk, tiles = inputs["input_ids"][0].tolist(), len(inputs["pixel_values"])
            else:
                trunk = self._tokenizer.encode(prompt.prefix_text(self._tokenizer, state, bos, style, system),
                                               add_special_tokens=False)
            branches = [self._tokenizer.encode(prompt.suffix_text(self._tokenizer, q, lead, option_style),
                                               add_special_tokens=False) for q in questions]
            list(groups(trunk, branches, self.context))  # raises when a group exceeds the context
            return trunk, branches, questions, pics, tiles

        def _compute(self):
            """Neural Engine time and counted FLOPs since the previous call."""
            return self._split(self._host.metrics(), 1)

        def _split(self, engine, share):
            """`share` of the Neural Engine time and FLOPs in a host metrics reading."""
            return {"neural_engine_ms": 1000 * engine["seconds"] * share,
                    "flops": round(share * sum(self._flops[name] * calls for name, calls in engine["calls"].items()))}

        def _run_trees(self, requests):
            """Each request's (probabilities, input tokens, Neural Engine time and FLOPs). Requests packed
            into one pass split its compute by their tokens."""
            prepared, singles = [], []
            # Validate every request before starting any native inference.
            for index, request in enumerate(requests):
                ids, branches, questions, pics, _ = self._plan(*request)
                if questions and not pics and not branches:
                    singles.append((index, ids, questions[0]))
                    prepared.append(None)
                else:
                    prepared.append((ids, branches, questions, len(ids) + sum(map(len, branches)), pics))
            done = [None] * len(prepared)
            self._host.metrics()  # Start each request's Neural Engine count from zero.
            if singles:
                # Pack independent single-question requests too, as the Torch API does.
                # Plan conservatively by full row length before sharing a common prefix.
                for indices in groups([], [r for _, r, _ in singles], self.context):
                    if len(indices) == 1:
                        index, ids, question = singles[indices[0]]
                        done[index] = ([self._probabilities(ids, question)], len(ids), self._compute())
                        continue
                    trunk, branches = shared_prefix([singles[i][1] for i in indices])
                    probabilities = self._tree_probabilities(trunk, branches, [singles[i][2] for i in indices])
                    engine = self._host.metrics()
                    tokens = sum(len(singles[i][1]) for i in indices)
                    for i, probs in zip(indices, probabilities):
                        index, ids, _ = singles[i]
                        done[index] = ([probs], len(ids), self._split(engine, len(ids) / tokens))
            for index, item in enumerate(prepared):
                if item is not None:
                    trunk, branches, questions, usage, pics = item
                    feats = [self._vision.encode_image(im) for im in pics] if pics and questions else None
                    if not questions:
                        probabilities = []
                    elif not branches:
                        probabilities = [self._probabilities(trunk, questions[0], image_features=feats)]
                    else:
                        probabilities = self._tree_probabilities(trunk, branches, questions, image_features=feats)
                    done[index] = (probabilities, usage if questions else 0, self._compute())
            return done

        def count_tokens(self, state, questions, images=None) -> dict:
            """Input tokens and vision tiles `system_one` would use, without running the model, and how
            each picture is fed to it (see `VisionPipeline.layout`)."""
            with self._lock:
                if self._closed:
                    raise RuntimeError("the model is closed")
                ids, branches, _, pics, tiles = self._plan(state, list(questions.values()), images)
                return {"input_tokens": len(ids) + sum(map(len, branches)), "image_tiles": tiles,
                        "images": [self._get_vision().layout(pic) for pic in pics]}

        def _run(self, requests):
            with self._lock:
                if self._closed:
                    raise RuntimeError("the model is closed")
                return self._run_trees(requests)

        def run(self, requests):
            return [(probabilities, read) for probabilities, read, _ in self._run(requests)]

        def system_one_batch(self, requests) -> list[dict]:
            """Many `(state, questions[, images])` requests, as upstream. `usage` adds each request's
            Neural Engine time and the FLOPs of the functions it ran, counted at export."""
            named = [(r[0], {n: prompt.as_question(q) for n, q in r[1].items()}, (r[2] if len(r) > 2 else None) or ())
                     for r in requests]
            done = self._run([(state, list(qs.values()), images) for state, qs, images in named])
            return [{"answers": {n: api.answer(q, p) for (n, q), p in zip(qs.items(), probs)},
                     "usage": {"input_tokens": read, "output_tokens": 0, **compute}}
                    for (_, qs, _), (probs, read, compute) in zip(named, done)]

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
