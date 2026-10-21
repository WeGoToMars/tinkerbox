import math
import os
from pathlib import Path
import sys
import warnings

import pytest

import tinkerbox_ane
from tinkerbox_ane.d3.runtime import SequentialQuestionsWarning

GREETING = {"type": "noul", "instructions": "Is this a greeting?"}
TOPIC = {"type": "choice", "instructions": "Choose a topic.",
         "criteria": {"hello": "A greeting", "billing": "A payment dispute"}}
TONE = {"type": "score", "instructions": "How friendly is this?", "criteria": ["Unfriendly", "Neutral", "Friendly"]}


@pytest.fixture(params=["d3-nano", "d3-mini"])
def bundle(request):
    asset = os.environ.get(request.param.upper().replace("-", "_") + "_ASSET")
    if sys.platform != "darwin" or not asset or not (Path(asset) / "model.json").is_file():
        pytest.skip(f"set {request.param.upper().replace('-', '_')}_ASSET to a compiled {request.param} bundle")
    return request.param, Path(asset)


@pytest.fixture
def model(bundle, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with tinkerbox_ane.load(*bundle) as model, warnings.catch_warnings():
        warnings.simplefilter("ignore", SequentialQuestionsWarning)
        yield model


def test_public_decisions_batches_and_request_reset(model):
    questions = {"greeting": GREETING, "topic": TOPIC, "tone": TONE}
    state = "Hello, how are you?"
    result = model.system_one(state, questions)
    assert result["answers"]["greeting"]["noul"] > 0.70
    assert result["answers"]["topic"]["choice"] == "hello"
    tone = result["answers"]["tone"]
    assert 0 <= tone["score"] <= 2
    assert tone["legend"] == {"0": "Unfriendly", "1": "Neutral", "2": "Friendly"}
    assert sum(tone["probabilities"].values()) == pytest.approx(1)
    # Each question reads its whole prompt.
    usage = result["usage"]
    assert usage["input_tokens"] == sum(len(model.tokens(state, q)) for q in questions.values())
    assert usage["output_tokens"] == 0
    assert model.count_tokens(state, questions)["input_tokens"] == usage["input_tokens"]
    # Each request reports its own Neural Engine time and the FLOPs of the functions it ran.
    assert usage["neural_engine_ms"] > 0 and usage["flops"] > 0

    first, empty = model.system_one_batch([(state, questions), (state, {})])
    assert first["answers"] == result["answers"]
    assert (first["usage"]["input_tokens"], first["usage"]["flops"]) == (usage["input_tokens"], usage["flops"])
    assert empty == {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0, "neural_engine_ms": 0, "flops": 0}}
    long = model.probabilities("Hello. " * 150, [GREETING])[0]
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in long)
    assert sum(long) == pytest.approx(1)
    again = model.system_one(state, questions)
    assert (again["answers"], again["usage"]["flops"]) == (result["answers"], usage["flops"])


def test_several_questions_warn_and_run_in_order(model):
    state = "Archive note: the customer acknowledged the message. " * 20 + "Hi, thanks!"
    questions = [GREETING, TOPIC, TONE]
    with warnings.catch_warnings():
        warnings.simplefilter("error", SequentialQuestionsWarning)
        individual = [model.probabilities(state, [q])[0] for q in questions]
    with pytest.warns(SequentialQuestionsWarning, match="3 questions run sequentially"):
        together = model.probabilities(state, questions)
    with pytest.warns(SequentialQuestionsWarning, match="2 questions"):
        batch = model.probabilities_batch([(state, [GREETING]), ("Goodbye!", [GREETING])])
    assert together == individual
    assert batch[0] == [individual[0]]


def test_public_input_limits_and_close(model):
    with pytest.raises(ValueError, match="image"):
        model.probabilities("Hi", [GREETING], images=[object()])
    with pytest.raises(ValueError, match="context limit"):
        model.probabilities_batch([("Hi", [GREETING]), ("hello " * model.context, [GREETING])])
    assert model.probabilities("Hi", [GREETING])[0][1] > 0.70
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.probabilities("Hi", [GREETING])


def test_multimodal_decisions_keep_image_shape(model):
    from PIL import Image

    blue = Image.new("RGB", (640, 480), color="blue")
    red = Image.new("RGB", (300, 900), color="red")
    is_blue = {"type": "noul", "instructions": "Is this image blue?"}
    color = {"type": "choice", "instructions": "What color is this image?",
             "criteria": {"blue": "Blue", "red": "Red", "green": "Green"}}

    result = model.system_one(None, {"blue": is_blue, "color": color}, images=[blue])
    assert result["answers"]["blue"]["noul"] > 0.85
    assert result["answers"]["color"]["choice"] == "blue"
    count = model.count_tokens(None, {"blue": is_blue, "color": color}, images=[blue])
    assert count["input_tokens"] == result["usage"]["input_tokens"]
    assert count["images"] == [{"source": [640, 480], "cols": 1, "rows": 1, "size": [640, 480],
                                "overview": None, "passes": 1, "tokens": 300}]
    assert model.count_tokens(None, {"c": color}, images=[red])["images"][0]["size"] == [288, 896]

    assert model.probabilities(None, [is_blue], images=[red])[0][1] < 0.15
    with pytest.raises(ValueError, match="placeholder"):
        model.probabilities_batch([(None, [is_blue], [blue]), ("<|image_pad|>", [is_blue], [blue])])
    batch = model.system_one_batch([
        (None, {"color": color}, [red]),
        ("Hello, how are you?", {"greeting": GREETING}, None),
    ])
    assert batch[0]["answers"]["color"]["choice"] == "red"
    assert batch[1]["answers"]["greeting"]["noul"] > 0.70
    assert batch[1]["usage"]["input_tokens"] == len(model.tokens("Hello, how are you?", GREETING))

    model.max_image_tokens = 64
    small = model.count_tokens(None, {"c": color}, images=[blue])
    assert small["images"][0]["tokens"] <= 64
    assert model.system_one(None, {"color": color}, images=[blue])["answers"]["color"]["choice"] == "blue"


def test_long_prompt_preserves_evidence_and_request_isolation(model):
    if model.context < 8192:
        pytest.skip("requires a bundle with an attention capacity above 4096")
    question = {
        "type": "choice", "instructions": "Which team is assigned to the TARGET ticket?",
        "criteria": {"billing": "Billing", "technical": "Technical support", "shipping": "Shipping"},
    }
    state = ("Archive note: this line contains no target ticket information. " * 500
             + "TARGET ticket: The assigned team is Shipping.")
    result = model.system_one(state, {"team": question})
    assert result["answers"]["team"]["choice"] == "shipping"
    assert result["answers"]["team"]["confidence"] > .9
    assert result["usage"]["input_tokens"] == len(model.tokens(state, question))
    short = model.system_one("TARGET ticket: The assigned team is Billing.", {"team": question})
    assert short["answers"]["team"]["choice"] == "billing"
