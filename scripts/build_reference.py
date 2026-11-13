"""Build 204 official FP32 CPU decisions covering public benchmarks and the fixed 32k runtime."""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from tinkerbox_ane.d1 import REPO, REVISION
from tinkerbox_ane.d1.validation import distribution, load_reference, reference_metadata

PER_SOURCE = 32
PUBLIC_MAX_TOKENS = 512
EVIDENCE_POSITIONS = ("beginning", "middle", "end")
QUESTION_TYPES = ("noul", "choice", "score")
LETTERS = "ABCDEFGH"


def rows(dataset: str, config: str, split: str, total: int) -> list[tuple[int, dict]]:
    result = []
    for offset in (k * (total // 8) for k in range(8)):
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 12}
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    fetched = json.load(response)["rows"]
                if len(fetched) != 12:
                    raise ValueError(f"expected 12 rows at offset {offset}, received {len(fetched)}")
                block = [(row["row_idx"], row["row"]) for row in fetched]
                if [index for index, _ in block] != list(range(offset, offset + 12)):
                    raise ValueError("dataset response does not match the requested row positions")
                result.extend(block)
                break
            except Exception as error:
                if attempt == 2:
                    raise RuntimeError(f"failed to fetch {dataset}/{config}/{split} at {offset}") from error
                time.sleep(2 * (attempt + 1))
    return result


def arc(row):
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    return (
        row["question"],
        {
            "type": "choice",
            "instructions": "Which option answers the question correctly?",
            "criteria": dict(zip(labels, texts)),
        },
        labels.index(row["answerKey"]),
    )


def mmlu(row):
    return (
        row["question"],
        {
            "type": "choice",
            "instructions": "Which option answers the question correctly?",
            "criteria": dict(zip(LETTERS, row["choices"])),
        },
        row["answer"],
    )


def boolq(row):
    question = row["question"].strip().rstrip("?") + "?"
    return (
        row["passage"],
        {"type": "noul", "instructions": question[0].upper() + question[1:]},
        0 if row["answer"] else 1,
    )


def ag_news(row):
    topics = {
        "world": "World news and politics",
        "sports": "Sports",
        "business": "Business and economy",
        "sci_tech": "Science and technology",
    }
    return (
        row["text"],
        {"type": "choice", "instructions": "What is the topic of this news article?", "criteria": topics},
        row["label"],
    )


def yelp(row):
    return (
        row["text"],
        {
            "type": "score",
            "instructions": "How many stars did the reviewer give?",
            "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        },
        row["label"],
    )


SOURCES = [
    ("arc_challenge", ("allenai/ai2_arc", "ARC-Challenge", "test", 1172), arc),
    ("mmlu", ("cais/mmlu", "all", "test", 14042), mmlu),
    ("boolq", ("google/boolq", "default", "validation", 3270), boolq),
    ("ag_news", ("fancyzhx/ag_news", "default", "test", 7600), ag_news),
    ("yelp", ("Yelp/yelp_review_full", "yelp_review_full", "test", 50000), yelp),
]


def long_prompt(engine, api, bucket: int, question_type: str, position: str, variant: int) -> dict:
    if question_type == "noul":
        label = variant % 2
        question = {"type": "noul", "instructions": "According to the TARGET ticket, was the refund approved?"}
        evidence = "TARGET ticket: The refund was " + ("approved." if label == 0 else "denied.")
    elif question_type == "choice":
        label = variant % 3
        teams = ("Billing", "Technical support", "Shipping")
        question = {
            "type": "choice",
            "instructions": "Which team is assigned to the TARGET ticket?",
            "criteria": {"billing": "Billing", "technical": "Technical support", "shipping": "Shipping"},
        }
        evidence = f"TARGET ticket: The assigned team is {teams[label]}."
    elif question_type == "score":
        label = variant % 5
        question = {
            "type": "score",
            "instructions": "What priority level is explicitly assigned to the TARGET ticket?",
            "criteria": ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5"],
        }
        evidence = f"TARGET ticket: The assigned priority level is {label + 1} out of 5."
    else:
        raise ValueError(f"unknown question type: {question_type}")
    q = api.as_question(question)
    prefix = "Support archive. Answer only from the record marked TARGET; other archive notes are unrelated.\n"
    note = "Archive note: A previous customer received the receipt and acknowledged the message. The case is closed.\n"

    def candidate(characters):
        filler = (note * (characters // len(note) + 1))[:characters]
        cut = {"beginning": 0, "middle": len(filler) // 2, "end": len(filler)}[position]
        if position == "middle":
            cut = filler.rfind("\n", 0, cut) + 1
        state = prefix + filler[:cut] + "\n" + evidence + "\n" + filler[cut:]
        ids = engine.tokenizer.encode(engine.render(state, q), add_special_tokens=False)
        return state, ids

    if len(candidate(0)[1]) > bucket:
        raise ValueError(f"bucket {bucket} cannot hold the long-prompt question and evidence")
    low, high = 0, bucket * 8
    while len(candidate(high)[1]) <= bucket:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if len(candidate(middle)[1]) <= bucket:
            low = middle
        else:
            high = middle
    state, ids = candidate(low)
    if not bucket - 8 <= len(ids) <= bucket:
        raise ValueError(f"could not fill bucket {bucket}: got {len(ids)} tokens")
    return {
        "source": f"long_{bucket}",
        "evidence_position": position,
        "state": state,
        "question": question,
        "label": label,
        "ids": ids,
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    output = Path("artifacts/reference.json")
    import torch
    from transformers import AutoModel

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True, dtype=torch.float32).eval().cpu()
    engine = model.engine
    api = sys.modules[type(engine).__mro__[1].__module__]

    def decision(state, question, label, source):
        ids = engine.tokenizer.encode(engine.render(state, api.as_question(question)), add_special_tokens=False)
        return {"source": source, "state": state, "question": question, "label": label, "ids": ids}

    def score(item):
        item["reference"] = distribution(engine.probabilities(item["state"], [item["question"]])[0])
        print(f"{item['source']}: {len(item['ids'])} tokens", flush=True)
        return item

    items = []
    for name, spec, cast in SOURCES:
        selected = []
        for row_index, row in rows(*spec):
            state, question, label = cast(row)
            item = decision(state, question, label, name)
            if len(item["ids"]) <= PUBLIC_MAX_TOKENS:
                item["source_row_index"] = row_index
                selected.append(score(item))
            if len(selected) == PER_SOURCE:
                break
        if len(selected) != PER_SOURCE:
            raise RuntimeError(f"{name}: expected {PER_SOURCE} prompts, found {len(selected)}")
        items.extend(selected)
    items.append(score(decision("Hi", {"type": "noul", "instructions": "Is this a greeting?"}, 0, "short_prompt")))
    for bucket_index, length in enumerate((1024, 2048, 3072, 4096)):
        for type_index, kind in enumerate(QUESTION_TYPES):
            for position_index, position in enumerate(EVIDENCE_POSITIONS):
                items.append(
                    score(long_prompt(engine, api, length, kind, position, bucket_index + type_index + position_index))
                )
    for length in (6144, 16384, 32768, 10000, 18000, 22000, 26000):
        items.append(score(long_prompt(engine, api, length, "noul", "middle", 0)))
    document = {"reference": reference_metadata(), "items": items}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        load_reference(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {len(items)} decisions to {output}")


if __name__ == "__main__":
    main()
