"""Build official FP32 CPU references or validate a d3 ANE bundle against them."""

import argparse
import json
import math
from pathlib import Path
import time

import tinkerbox_ane
from tinkerbox_ane.bundle import read_bundle
from tinkerbox_ane.d3 import KIND, VARIANTS, checkpoint
from tinkerbox_ane.d3.runtime import d3_code


def picture(spec):
    """A deterministic test image: a solid color, or color stripes across its width."""
    from PIL import Image, ImageDraw

    width, height = spec["size"]
    image = Image.new("RGB", (width, height), spec["color"])
    if spec.get("stripes"):
        draw = ImageDraw.Draw(image)
        for i, color in enumerate(spec["stripes"]):
            draw.rectangle([i * width // len(spec["stripes"]), 0, (i + 1) * width // len(spec["stripes"]), height],
                           fill=color)
    return image


def cases():
    greeting = {"type": "noul", "instructions": "Is this a greeting?"}
    topic = {"type": "choice", "instructions": "Choose a topic.",
             "criteria": {"hello": "A greeting", "billing": "A payment dispute", "weather": "A weather report"}}
    score = {"type": "score", "instructions": "How friendly is this?",
             "criteria": ["Unfriendly", "Neutral", "Friendly"]}
    color = {"type": "choice", "instructions": "What is the main color of the picture?",
             "criteria": {"red": "Red", "green": "Green", "blue": "Blue"}}
    order = {"type": "noul", "instructions": "Do the stripes go red, green, then blue from left to right?"}
    same = {"type": "noul", "instructions": "Are both pictures the same color?"}
    rgb = ["red", "green", "blue"]
    return [("Hello, how are you?", greeting, []),
            ("Archive note: the customer acknowledged the message. " * 20 + "Hi, thanks!", greeting, []),
            ("Archive note: the customer acknowledged the message. " * 80 + "Hi, thanks!", greeting, []),
            ("The invoice charged me twice. Please refund the duplicate payment.", topic, []),
            ("Warm sunshine is expected throughout the weekend.", topic, []),
            ("Thank you so much for your help! Have a wonderful day!", score, []),
            ("This is unacceptable. Fix your mistake immediately.", score, []),
            (None, color, [{"size": [640, 480], "color": "blue"}]),
            ("A tall photo.", color, [{"size": [300, 900], "color": "green"}]),
            (None, order, [{"size": [1280, 960], "color": "white", "stripes": rgb}]),
            (None, order, [{"size": [1200, 400], "color": "white", "stripes": list(reversed(rgb))}]),
            ("Two photos.", same, [{"size": [512, 512], "color": "red"}, {"size": [800, 600], "color": "red"}])]


def distribution(values, count):
    if len(values) != count or not math.isclose(sum(values), 1, abs_tol=1e-4):
        raise ValueError("expected one probability per option, summing to one")
    return list(values)


def build_reference(model, out):
    import torch

    torch.set_num_threads(6)
    repo, revision, _ = VARIANTS[model]
    path = checkpoint(repo, revision)
    _, rt = d3_code(path)
    official = rt.D3.from_pretrained(str(path), device="cpu", permutation_average=False)
    official.fast = None
    # The official runtime defaults to BF16; validation explicitly uses FP32 CPU.
    official.backbone.float()
    items = []
    for state, question, images in cases():
        prepared = official.prepare(state, {"q": question}, [picture(spec) for spec in images] or None)
        probabilities, tokens = official.run(prepared)
        count = len(prepared.questions["q"].keys)
        item = {"state": state, "question": question, "images": images, "tokens": tokens,
                "reference": distribution(probabilities["q"], count)}
        if not images:
            item["ids"] = prepared.sequences["q"]
        items.append(item)
        print(f"reference: {tokens} tokens {item['reference']}", flush=True)
    document = {"reference": {"repository": repo, "revision": revision, "implementation": "official",
                              "dtype": "float32", "device": "cpu"}, "items": items}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")


def validate(asset, reference):
    spec = read_bundle(asset, KIND)
    model = next(name for name, (repo, _, _) in VARIANTS.items() if repo == spec["source"]["repo"])
    document = json.loads(Path(reference).read_text())
    if document["reference"]["revision"] != spec["source"]["revision"]:
        raise ValueError("reference and bundle use different checkpoints")
    records = []
    with tinkerbox_ane.load(model, asset) as runtime:
        for item in document["items"]:
            images = [picture(image) for image in item["images"]]
            started = time.monotonic()
            got = runtime.probabilities(item["state"], [item["question"]], images)[0]
            tokens = runtime.count_tokens(item["state"], {"q": item["question"]}, images)["input_tokens"]
            ref = item["reference"]
            drift = [abs(a - b) for a, b in zip(got, ref)]
            records.append({"tokens": tokens, "images": len(images),
                            "probabilities": got, "reference": ref, "max_probability_drift": max(drift),
                            "latency_s": time.monotonic() - started,
                            "token_match": tokens == item["tokens"],
                            "agreement": got.index(max(got)) == ref.index(max(ref))})
    errors = [abs(a - b) for r in records for a, b in zip(r["probabilities"], r["reference"])]
    report = {"asset": str(Path(asset).resolve()), "architecture": spec["architecture"],
              "source_hashes": [a["source_hash"] for a in spec["assets"]], "records": records,
              "max_probability_drift": max(errors), "mean_probability_drift": sum(errors) / len(errors)}
    report["passed"] = (all(r["token_match"] and r["agreement"] for r in records)
                        and max(errors) <= .05 and sum(errors) / len(errors) <= .005)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-reference", choices=list(VARIANTS), metavar="MODEL",
                        help="Write the official FP32 reference of d3-nano or d3-mini")
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.write_reference:
        build_reference(args.write_reference, args.reference)
    else:
        if not args.asset or not args.out:
            parser.error("validation requires --asset and --out")
        report = validate(args.asset, args.reference)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
