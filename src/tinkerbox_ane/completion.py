"""Worker requests for text-completion models."""

import time


def complete(model, entry, request, emit):
    """Count or complete prompts; streamed text is emitted at most 30 times a second, with the number of
    steps that produced it."""
    if request.pop("count", False):
        prompt = request.get("prompt")
        prompts = prompt if isinstance(prompt, list) else [prompt or ""]
        return dict(counts=[dict(tokens=model.count_tokens(p), tiles=0, images=[]) for p in prompts])
    result = model.complete(**request)
    if not request.get("stream"):
        return result
    pending, last = {}, 0.0

    def flush():
        emit(
            dict(
                completionChunk=dict(
                    choices=[dict(index=index, text=text, tokens=tokens) for index, (text, tokens) in pending.items()]
                )
            )
        )
        pending.clear()

    try:
        while True:
            try:
                chunk = next(result)
            except StopIteration as done:
                if pending:
                    flush()
                return done.value
            for choice in chunk["choices"]:
                # A finishing choice carries held-back text, not a new token.
                text, tokens = pending.get(choice["index"], ("", 0))
                pending[choice["index"]] = (text + choice["text"], tokens + (choice["finish_reason"] is None))
            if pending and time.perf_counter() - last >= 1 / 30:
                flush()
                last = time.perf_counter()
    finally:
        result.close()
