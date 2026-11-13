"""Public text-completion behavior against real compiled GPT-2 bundles."""

import pytest

import tinkerbox_ane
from tinkerbox_ane.completion import complete as worker_complete
from tinkerbox_ane.models import entry

# Every size and format shares one architecture and runtime. API behavior runs on GPT-2 Small; batch and cache
# numerics run on one variant per weight format, across three sizes.
VARIANTS = ["gpt2", "gpt2-medium-int8", "gpt2-xl-lut8"]
# GPT-2 Small's pinned upstream greedy continuation of 'The capital of France is' catches a corrupted export
# even if every local execution width agrees with the same incorrect model.
CAPITAL = [262, 3139, 286, 262]
PROMPTS = ["The capital of France is", "Once upon a time,"]


def load(bundle, model_id, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return tinkerbox_ane.load(model_id, bundle(model_id, model_id.upper().replace("-", "_") + "_ASSET"))


@pytest.fixture(scope="module")
def model(bundle):
    with pytest.MonkeyPatch.context() as monkeypatch, load(bundle, "gpt2", monkeypatch) as model:
        yield model


@pytest.fixture(scope="module", params=VARIANTS)
def variant(bundle, request):
    with pytest.MonkeyPatch.context() as monkeypatch, load(bundle, request.param, monkeypatch) as model:
        yield model


def texts(chunks, count):
    text = [""] * count
    for chunk in chunks:
        for choice in chunk["choices"]:
            text[choice["index"]] += choice["text"]
    return text


def test_openai_completion_shape(model):
    reply = model.complete(model=model.model_id, prompt=PROMPTS, max_tokens=4, temperature=0, batch_size=4)
    assert reply["id"].startswith("cmpl-") and reply["created"] > 0
    assert reply["object"] == "text_completion" and reply["model"] == model.model_id
    assert [choice["index"] for choice in reply["choices"]] == [0, 1]
    assert all(
        choice["text"]
        and choice["finish_reason"] == "length"
        and choice["logprobs"] is None
        and len(choice["token_ids"]) == 4
        for choice in reply["choices"]
    )
    assert reply["usage"] == dict(prompt_tokens=10, completion_tokens=8, total_tokens=18)
    assert reply["metrics"]["elapsed_ms"] > 0
    single = model.complete(prompt=PROMPTS[0], max_tokens=4, temperature=0)
    assert single["choices"][0]["text"] == reply["choices"][0]["text"] and single["id"] != reply["id"]
    bos = model.complete(prompt=None, max_tokens=0)
    assert bos["choices"][0]["text"] == "" and bos["usage"] == dict(
        prompt_tokens=1, completion_tokens=0, total_tokens=1
    )
    with pytest.raises(ValueError, match="n=1"):
        model.complete(prompt="Hello", n=2)
    with pytest.raises(ValueError, match="context limit"):
        model.complete(prompt=[PROMPTS[0], " hello" * model.context], max_tokens=1, stream=True)


def test_token_spans_cover_the_text(model):
    assert model.token_spans("Hello world!") == [
        dict(start_utf16=0, end_utf16=5, token_ids=[15496]),
        dict(start_utf16=5, end_utf16=11, token_ids=[995]),
        dict(start_utf16=11, end_utf16=12, token_ids=[0]),
    ]
    text = "Hi 👋🏽! 日本語 café\r\nend"
    encoded = text.encode("utf-16-le")
    spans = model.token_spans(text)
    assert b"".join(encoded[2 * s["start_utf16"] : 2 * s["end_utf16"]] for s in spans).decode("utf-16-le") == text
    assert sum(len(s["token_ids"]) for s in spans) == model.count_tokens(text)
    assert all(encoded[2 * s["start_utf16"] : 2 * s["end_utf16"]].decode("utf-16-le") for s in spans)
    assert model.token_spans("") == [] and model.count_tokens("") == 1


def test_streaming_joins_into_the_complete_reply(model):
    prompts = ["The capital of France is", "こんにちは、私は", "Once upon a time,", "The capital of France is"]
    options = dict(prompt=prompts, max_tokens=12, temperature=0, batch_size=4)
    expected = model.complete(**options)
    chunks = list(model.complete(**options, stream=True, stream_options=dict(include_usage=True)))
    assert all(chunk["id"] == chunks[0]["id"] for chunk in chunks)
    assert texts(chunks, len(prompts)) == [choice["text"] for choice in expected["choices"]]
    assert any(ord(char) > 127 for char in expected["choices"][1]["text"])
    finishes = {c["index"]: c for chunk in chunks for c in chunk["choices"] if c["finish_reason"] is not None}
    assert [finishes[i]["token_ids"] for i in range(len(prompts))] == [c["token_ids"] for c in expected["choices"]]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == expected["usage"]
    assert all(chunk["usage"] is None for chunk in chunks[:-1])
    seeded = dict(prompt=prompts[:2], max_tokens=12, temperature=0.8, top_k=40, seed=42)
    assert texts(model.complete(**seeded, stream=True), 2) == [c["text"] for c in model.complete(**seeded)["choices"]]
    stream = model.complete(prompt=prompts[0], max_tokens=8, temperature=0, stream=True)
    assert next(stream)["choices"][0]["text"]
    with pytest.raises(RuntimeError, match="active completion stream"):
        model.complete(prompt=prompts[0], max_tokens=1)
    stream.close()
    assert (
        model.complete(prompt=prompts[0], max_tokens=12, temperature=0)["choices"][0]["text"]
        == expected["choices"][0]["text"]
    )


def test_worker_streams_and_counts_against_the_context(model):
    entry_ = entry(model.model_id)
    counts = worker_complete(model, entry_, dict(count=True, prompt=PROMPTS), emit=None)
    assert [count["tokens"] for count in counts["counts"]] == [model.count_tokens(p) for p in PROMPTS]
    pieces = []
    reply = worker_complete(
        model, entry_, dict(prompt=PROMPTS, max_tokens=16, temperature=0, stream=True), emit=pieces.append
    )
    assert reply["choices"] and pieces and all("completionChunk" in piece for piece in pieces)
    streamed = texts((piece["completionChunk"] for piece in pieces), len(PROMPTS))
    assert all(choice["text"].startswith(text) for choice, text in zip(reply["choices"], streamed))
    assert [c["text"] for c in reply["choices"]] == [
        c["text"] for c in model.complete(prompt=PROMPTS, max_tokens=16, temperature=0)["choices"]
    ]
    with pytest.raises(ValueError, match="context limit"):
        worker_complete(model, entry_, dict(prompt=" hello" * model.context, max_tokens=1), emit=pieces.append)


def test_batches_match_single_requests_and_seeded_replay(variant):
    model = variant
    options = dict(max_tokens=8, temperature=0)
    singles = [model.complete(prompt=p, **options)["choices"][0] for p in PROMPTS]
    if model.model_id == "gpt2":
        assert singles[0]["token_ids"][:4] == CAPITAL
    full = model.complete(prompt=[PROMPTS[0]] * max(model.batch_sizes), **options)["choices"]
    assert all(
        c["metrics"]["batch_size"] == max(model.batch_sizes) and c["token_ids"] == singles[0]["token_ids"] for c in full
    )
    # Partial groups preserve independent continuations, including repeated prompts in another order.
    for order in ([0, 1], [1, 0, 1]):
        batch = model.complete(prompt=[PROMPTS[i] for i in order], **options)["choices"]
        for index, together in zip(order, batch):
            alone = singles[index]
            assert together["token_ids"] == alone["token_ids"] and alone["text"]
            assert alone["metrics"]["input_tokens"] == model.count_tokens(PROMPTS[index])
            assert alone["metrics"]["output_tokens"] == len(alone["token_ids"]) <= 8
    sampled = dict(max_tokens=12, temperature=0.8, top_k=30, top_p=0.9, seed=42)
    first = model.complete(prompt=PROMPTS[0], **sampled)["choices"][0]["token_ids"]
    model.complete(prompt=PROMPTS[1], **sampled)
    assert first == model.complete(prompt=PROMPTS[0], **sampled)["choices"][0]["token_ids"]
    assert [c["token_ids"] for c in model.complete(prompt=PROMPTS, **sampled)["choices"]] == [
        model.complete(prompt=p, **sampled)["choices"][0]["token_ids"] for p in PROMPTS
    ]


def test_context_growth_and_mixed_length_batches(variant):
    model = variant
    # One BPE token per repetition: decode crosses the cache growth boundaries.
    for count in (63, 127, 511):
        choice = model.complete(prompt=" hello" * count, max_tokens=4, temperature=0)["choices"][0]
        assert choice["metrics"]["input_tokens"] == count and choice["metrics"]["output_tokens"] > 0
    prompts = [
        "The capital of France is",
        "",
        " hello" * 255,
        "A short story about an astronaut",
        "Hello",
        "The ocean is",
        "The musician played",
        "The library contains",
        "The scientist discovered",
    ]
    options = dict(max_tokens=8, temperature=0)
    expected = [model.complete(prompt=p, **options)["choices"][0]["token_ids"] for p in prompts]
    actual = model.complete(prompt=prompts, **options)["choices"]
    assert [c["token_ids"] for c in actual] == expected
    assert [c["metrics"]["input_tokens"] for c in actual] == [model.count_tokens(p) for p in prompts]
    limited = model.complete(prompt=prompts, batch_size=4, **options)["choices"]
    assert [c["token_ids"] for c in limited] == expected
    # Reordering requests cannot share context, including partial final groups.
    reversed_ = model.complete(prompt=list(reversed(prompts)), **options)["choices"]
    assert [c["token_ids"] for c in reversed_] == list(reversed(expected))
    edge = [
        word * (model.context - 1)
        for word in (" hello", " world", " book", " music", " red", " green", " blue", " yellow")
    ]
    alone = [model.complete(prompt=p, max_tokens=1, temperature=0)["choices"][0]["token_ids"] for p in edge]
    assert [
        c["token_ids"]
        for c in model.complete(prompt=edge, max_tokens=1, temperature=0, batch_size=max(model.batch_sizes))["choices"]
    ] == alone


def test_close_ends_streams_and_requests(bundle, monkeypatch):
    model = load(bundle, "gpt2", monkeypatch)
    stream = model.complete(prompt=PROMPTS[0], max_tokens=8, temperature=0, stream=True)
    next(stream)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(stream)
    with pytest.raises(RuntimeError, match="closed"):
        model.complete(prompt="Hello")
