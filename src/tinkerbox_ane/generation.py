"""OpenAI Completions-shaped text completion: sampling, streaming, usage and ordered batches."""

import math
import threading
import time
import uuid

import numpy as np

from tinkerbox_ane.tokenization import token_spans


class TextGeneration:
    """A completion task composed with a causal decoder. Calls on one model are serialized."""

    def __init__(self, decoder):
        self._decoder = decoder
        self.model_id = decoder.model_id
        self.context = decoder.context
        self.batch_sizes = decoder.batch_sizes
        self._lock = threading.RLock()
        self._closed = False
        self._active = False

    def _tokens(self, prompt):
        return self._decoder.tokenizer.encode(prompt, add_special_tokens=False) or [
            self._decoder.tokenizer.bos_token_id
        ]

    def count_tokens(self, prompt):
        """Input length, including the BOS token used for an empty prompt."""
        return len(self._tokens(prompt))

    def token_spans(self, prompt):
        return token_spans(self._decoder.tokenizer, prompt)

    def complete(
        self,
        *,
        model=None,
        prompt="",
        max_tokens=16,
        temperature=1.0,
        top_p=1.0,
        seed=None,
        top_k=0,
        batch_size=None,
        n=1,
        stream=False,
        stream_options=None,
    ):
        """Return an OpenAI Completions-shaped response for one prompt or an ordered list.

        stream=True returns an iterator of chunks with indexed text deltas; close an
        unfinished one before starting another request. top_k, batch_size, token_ids
        and metrics are local extensions.
        """
        prompts = prompt if isinstance(prompt, list) else [prompt or ""]
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")
        if model is not None and model != self.model_id:
            raise ValueError("The requested model is not loaded.")
        if n != 1:
            raise ValueError("Only n=1 is supported; supply a list of prompts for independent completions.")
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer.")
        if not 0 <= temperature < math.inf:
            raise ValueError("temperature must be finite and nonnegative.")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer.")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer.")
        if batch_size is not None and batch_size not in self.batch_sizes:
            raise ValueError(f"batch_size must be one of {list(self.batch_sizes)}.")
        ids = [self._tokens(p) for p in prompts]
        if any(len(row) + max_tokens > self.context for row in ids):
            raise ValueError(f"Prompt and max_tokens exceed the {self.context}-token context limit.")
        header = dict(
            id=f"cmpl-{uuid.uuid4().hex}", object="text_completion", created=int(time.time()), model=self.model_id
        )
        events = self._events(ids, max_tokens, temperature, top_k, top_p, seed, batch_size)
        if stream:
            return self._stream(events, header, (stream_options or {}).get("include_usage", False))
        started = time.perf_counter()
        with self._lock:
            try:
                while True:
                    next(events)
            except StopIteration as done:
                return self._response(header, done.value, started)

    def _response(self, header, replies, started):
        prompt_tokens = sum(reply["metrics"]["input_tokens"] for reply in replies)
        completion_tokens = sum(reply["metrics"]["output_tokens"] for reply in replies)
        return dict(
            header,
            choices=[dict(index=index, logprobs=None, **reply) for index, reply in enumerate(replies)],
            usage=dict(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metrics=dict(elapsed_ms=(time.perf_counter() - started) * 1000),
        )

    def _stream(self, events, header, include_usage):
        started = time.perf_counter()
        tokens, emitted = {}, {}
        try:
            while True:
                try:
                    batch = next(events)
                except StopIteration as done:
                    response = self._response(header, done.value, started)
                    break
                choices = []
                for row, token in batch:
                    tokens.setdefault(row, []).append(token)
                    # A byte-level tokenizer can end a prefix inside a Unicode character.
                    # Hold incomplete trailing characters until later tokens resolve them.
                    text = self._decoder.tokenizer.decode(
                        tokens[row], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).rstrip("�")
                    if len(text) > len(emitted.get(row, "")):
                        choices.append(
                            dict(index=row, text=text[len(emitted.get(row, "")) :], logprobs=None, finish_reason=None)
                        )
                        emitted[row] = text
                if choices:
                    yield dict(header, choices=choices, usage=None)
            # Finish events carry each prompt's remaining text, token IDs and metrics.
            yield dict(
                header,
                choices=[
                    dict(choice, text=choice["text"][len(emitted.get(choice["index"], "")) :])
                    for choice in response["choices"]
                ],
                usage=None,
            )
            yield dict(
                header, choices=[], usage=response["usage"] if include_usage else None, metrics=response["metrics"]
            )
            return response
        finally:
            events.close()

    def _events(self, ids, *options):
        # The lock is held per step, never across a yield, so a stream may resume on any thread.
        with self._lock:
            if self._closed:
                raise RuntimeError("text generation model is closed")
            if self._active:
                raise RuntimeError("finish or close the active completion stream first")
            self._active = True
        try:
            events = self._batch(ids, *options)
            while True:
                with self._lock:
                    if self._closed:
                        raise RuntimeError("text generation model is closed")
                    try:
                        batch = next(events)
                    except StopIteration as done:
                        return done.value
                yield batch
        finally:
            self._active = False

    def _batch(self, ids, budget, temperature, top_k, top_p, seed, batch_size):
        replies = [None] * len(ids)
        for rows in self._groups(ids, budget, batch_size, seeded=temperature > 0 and seed is not None):
            group = yield from self._generate([ids[row] for row in rows], rows, budget, temperature, top_k, top_p, seed)
            for row, reply in zip(rows, group):
                replies[row] = reply
        return replies

    def _groups(self, ids, budget, batch_size, seeded):
        """Rows to decode together, shortest prompts first.

        Different FP16 execution widths can perturb sampling thresholds, so seeded
        sampling decodes each prompt alone for replay. Groups that would be mostly
        padding also run row by row.
        """
        order = sorted(range(len(ids)), key=lambda row: len(ids[row]))
        limit = 1 if seeded else batch_size or max(self.batch_sizes)
        start = 0
        while start < len(order):
            rows = order[start : start + limit]
            start += len(rows)
            size = next(b for b in self.batch_sizes if b >= len(rows))
            width = self._decoder.width
            # Eight rows pay off only when decode steps amortize the packed prefill calls:
            # packed attention grows with the physical cache, even for masked sequence pairs.
            if batch_size is None and size == 8 and (len(ids[rows[-1]]) * 8 + width - 1) // width > max(1, budget - 1):
                start -= len(rows) - 4
                rows, size = rows[:4], 4
            if size * (len(ids[rows[-1]]) + budget) > 2 * sum(len(ids[row]) + budget for row in rows):
                yield from ([row] for row in rows)
            else:
                yield rows

    def _generate(self, prompts, rows, budget, temperature, top_k, top_p, seed):
        started = time.perf_counter()
        size = next(b for b in self.batch_sizes if b >= len(prompts))
        tokenizer = self._decoder.tokenizer
        generated = [[] for _ in prompts]
        reasons, finished = ["length"] * len(prompts), [None] * len(prompts)
        rngs = [np.random.default_rng(seed) for _ in prompts]
        scores = self._decoder.start(prompts) if budget else None
        for step in range(budget):
            best = scores.argmax(1) if temperature == 0 else None
            next_tokens, events = [tokenizer.eos_token_id] * len(prompts), []
            for index in range(len(prompts)):
                if finished[index] is not None:
                    continue
                token = (
                    int(best[index])
                    if best is not None
                    else self._sample(scores[index], temperature, top_k, top_p, rngs[index])
                )
                generated[index].append(token)
                events.append((rows[index], token))
                next_tokens[index] = token
                if token == tokenizer.eos_token_id:
                    reasons[index] = "stop"
                    finished[index] = (time.perf_counter() - started) * 1000
            yield events
            if all(value is not None for value in finished):
                break
            if step + 1 < budget:
                scores = self._decoder.step(next_tokens)
        elapsed = (time.perf_counter() - started) * 1000
        return [
            dict(
                text=tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                finish_reason=reason,
                token_ids=tokens,
                metrics=dict(
                    input_tokens=len(prompt),
                    output_tokens=len(tokens),
                    batch_size=size,
                    elapsed_ms=latency if latency is not None else elapsed,
                ),
            )
            for prompt, tokens, reason, latency in zip(prompts, generated, reasons, finished)
        ]

    @staticmethod
    def _sample(scores, temperature, top_k, top_p, rng):
        if 0 < top_k < len(scores):
            threshold = np.partition(scores, len(scores) - top_k)[len(scores) - top_k]
            above = np.flatnonzero(scores > threshold)
            # Keep the original stable ordering at the cutoff: lowest token ID
            # wins equal scores, including ties that straddle the kth entry.
            ties = np.flatnonzero(scores == threshold)[: top_k - len(above)]
            order = np.concatenate((above, ties))
            order = order[np.lexsort((order, -scores[order]))]
        else:
            order = np.argsort(-scores, kind="stable")
        values = scores[order].astype(np.float64)
        values = (values - values[0]) / temperature
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        if top_p < 1:
            keep = min(len(order), int(np.searchsorted(np.cumsum(probabilities), top_p)) + 1)
            order, probabilities = order[:keep], probabilities[:keep]
            probabilities /= probabilities.sum()
        return int(rng.choice(order, p=probabilities))

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._decoder.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
