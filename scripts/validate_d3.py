"""Build official FP32 CPU references or validate a d3 ANE bundle against them.

Without --questions, references cover a small built-in set of text and image cases. With --questions, they cover a
saved decision set such as d1's 204 (scripts/build_reference.py writes it to artifacts/reference.json):
public benchmarks, a short prompt and long needle prompts, keeping those within the model's context."""

import argparse
import json
import hashlib
from pathlib import Path
import time

import tinkerbox_ane
from tinkerbox_ane.bundle import read_bundle
from tinkerbox_ane.d3 import KIND, VARIANTS, checkpoint
from tinkerbox_ane.d3.runtime import d3_code
from tinkerbox_ane.validation import distribution


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


def decision_set(path):
    """Text decisions with labels from a saved reference of another model."""
    return [{"source": item["source"], "label": item["label"], "state": item["state"],
             "question": item["question"], "images": []} for item in json.loads(Path(path).read_text())["items"]]


def float32_by_layer(backbone):
    """Run a BF16 backbone in FP32 while keeping each decoder layer's weights in BF16 between its calls.

    Every BF16 value is exact in FP32, so this computes what a fully FP32 model computes, but holds one
    FP32 layer at a time: d3-mini's reference fits in 24 GB."""
    import torch

    for name, parameter in backbone.named_parameters():
        if not name.startswith("language_model.layers."):
            parameter.data = parameter.data.float()

    def cast(dtype):
        def hook(layer, *_):
            for parameter in layer.parameters():
                parameter.data = parameter.data.to(dtype)
        return hook

    for layer in backbone.language_model.layers:
        layer.register_forward_pre_hook(cast(torch.float32))
        layer.register_forward_hook(cast(torch.bfloat16))


def build_reference(model, out, questions=None, dtype="float32"):
    import torch

    torch.set_num_threads(10)
    repo, revision, tiers = VARIANTS[model]
    path = checkpoint(repo, revision)
    _, rt = d3_code(path)
    official = rt.D3.from_pretrained(str(path), device="cpu", permutation_average=False, max_length=tiers[-1])
    official.fast = None
    # The official runtime defaults to BF16; validation uses FP32 CPU, and BF16 shows the upstream drift.
    if any(p.dtype != torch.bfloat16 for p in official.backbone.parameters()):
        raise ValueError("expected a BF16 checkpoint")
    if dtype == "float32":
        float32_by_layer(official.backbone)
    planned = decision_set(questions) if questions else [
        {"state": state, "question": question, "images": images} for state, question, images in cases()]
    items = []
    for item in planned:
        prepared = official.prepare(item["state"], {"q": item["question"]},
                                    [picture(spec) for spec in item["images"]] or None)
        if "q" in prepared.errors:
            print(f"skipped {item.get('source', 'case')}: {prepared.errors['q']['message']}", flush=True)
            continue
        probabilities, tokens = official.run(prepared)
        count = len(prepared.questions["q"].keys)
        item = {**item, "tokens": tokens, "reference": distribution(probabilities["q"], count)}
        if not item["images"]:
            item["ids"] = prepared.sequences["q"]
        items.append(item)
        print(f"reference {len(items)}/{len(planned)}: {tokens} tokens {item['reference']}", flush=True)
    document = {"reference": {"repository": repo, "revision": revision, "implementation": "official",
                              "dtype": dtype, "device": "cpu"}, "items": items}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")


def validate(asset, reference, baseline=None, *, smoke=False):
    spec = read_bundle(asset, KIND)
    source = spec["source"]
    model = next(name for name, (repo, _, _) in VARIANTS.items() if repo == source["repo"])
    reference_bytes = Path(reference).read_bytes()
    document = json.loads(reference_bytes)
    repo, revision, _ = VARIANTS[model]
    if source != {"repo": repo, "revision": revision}:
        raise ValueError("bundle does not use a pinned d3 checkpoint")
    expected = {"repository": repo, "revision": revision,
                "implementation": "official", "dtype": "float32", "device": "cpu"}
    if document.get("reference") != expected:
        raise ValueError("reference must use the pinned official FP32 CPU model")
    if not document.get("items"):
        raise ValueError("reference has no decisions")
    for item in document["items"]:
        count = 2 if item["question"]["type"] == "noul" else len(item["question"]["criteria"])
        distribution(item["reference"], count)
        if not item["images"] and not item.get("ids"):
            raise ValueError("text references need exact prompt token IDs")

    records = []
    tiers, prompts = spec["tiers"], spec.get("prompts", [])

    def path(tokens):
        width = next((w for w in prompts if w >= tokens), None)
        return f"prompt{width}" if width else f"attention{next(t for t in tiers if t >= tokens)}"

    # Short prompts use whole-prompt functions, making smaller attention functions unreachable.
    required = {f"prompt{w}" for w in prompts} | {f"attention{t}" for t in tiers if t > max(prompts, default=0)}
    covered = set()
    with tinkerbox_ane.load(model, asset) as runtime:
        warmed = set()
        patch_tiers = [int(name[1:]) for name in spec.get("vision_functions", [])]
        for item in document["items"]:
            images = [picture(image) for image in item["images"]]
            counted = runtime.count_tokens(item["state"], {"q": item["question"]}, images)
            tokens = counted["input_tokens"]
            execution = path(tokens)
            vision = tuple(next(p for p in patch_tiers if p >= image["tokens"] * 4)
                           for image in counted["images"])
            key = (execution, vision)
            if key not in warmed:
                runtime.probabilities(item["state"], [item["question"]], images)
                warmed.add(key)
            started = time.perf_counter()
            got = runtime.probabilities(item["state"], [item["question"]], images)[0]
            latency = time.perf_counter() - started
            ref = distribution(item["reference"], len(got))
            distribution(got, len(ref))
            token_match = tokens == item["tokens"] and (bool(images) or runtime.tokens(
                item["state"], item["question"]) == item["ids"])
            covered.add(execution)
            records.append({"source": item.get("source", "image" if images else "text"), "label": item.get("label"),
                            "tokens": tokens, "tier": next(t for t in tiers if t >= tokens), "images": len(images),
                            "path": execution, "vision_capacities": list(vision),
                            "probabilities": got, "reference": ref,
                            "max_probability_drift": max(abs(a - b) for a, b in zip(got, ref)),
                            "latency_s": latency, "token_match": token_match,
                            "agreement": got.index(max(got)) == ref.index(max(ref))})
    report = {"asset": str(Path(asset).resolve()), "architecture": spec["architecture"],
              "reference": expected, "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
              "source_hashes": [a["source_hash"] for a in spec["assets"]],
              "vision_source_hash": spec.get("vision_source_hash"), "checkpoint_hash": spec.get("checkpoint_hash"),
              "mode": "smoke" if smoke else "acceptance", "missing_paths": sorted(required - covered),
              **summary(records),
              "tiers": {str(tier): summary([r for r in records if r["tier"] == tier])
                        for tier in sorted({r["tier"] for r in records})}, "records": records}
    if baseline:
        # The same drift measures for another official precision, such as BF16, on the same items.
        others = json.loads(Path(baseline).read_text())
        identity = lambda item: (item["state"], item["question"], item["images"], item["tokens"])
        if ((others["reference"].get("repository"), others["reference"].get("revision")) != (repo, revision)
                or list(map(identity, others["items"])) != list(map(identity, document["items"]))):
            raise ValueError("baseline and reference cover different items")
        compared = [{**r, "probabilities": got, "latency_s": 0,
                     "agreement": got.index(max(got)) == r["reference"].index(max(r["reference"]))}
                    for r, got in zip(records, (distribution(item["reference"]) for item in others["items"]))]

        def drift(rows):
            return {k: v for k, v in summary(rows).items() if k not in ("p50_ms", "label_accuracy")}
        report["baseline"] = {"reference": others["reference"], **drift(compared),
                              "tiers": {tier: drift([r for r in compared if str(r["tier"]) == tier])
                                        for tier in report["tiers"]}}
    # d1's gates, except a maximum drift of 0.1 and 96% agreement per tier: the Neural Engine
    # runs the Gated DeltaNet recurrence in FP16, which upstream keeps in FP32.
    report["smoke_passed"] = (all(r["token_match"] for r in records) and report["agreement"] >= .97
                              and report["max_probability_drift"] <= .1 and report["mean_probability_drift"] <= .005
                              and all(t["agreement"] >= .96 and t["mean_probability_drift"] <= .01
                                      for t in report["tiers"].values()))
    report["passed"] = report["smoke_passed"] and not report["missing_paths"]
    return report


def summary(records):
    errors = [abs(a - b) for r in records for a, b in zip(r["probabilities"], r["reference"])]
    latencies = sorted(r["latency_s"] for r in records)
    labeled = [r for r in records if r["label"] is not None]
    result = {"decisions": len(records), "agreement": sum(r["agreement"] for r in records) / len(records),
              "mean_probability_drift": sum(errors) / len(errors), "max_probability_drift": max(errors),
              "p50_ms": 1000 * latencies[len(latencies) // 2]}
    if labeled:
        def correct(r, key):
            return r[key].index(max(r[key])) == r["label"]
        result["label_accuracy"] = {"ane": sum(correct(r, "probabilities") for r in labeled) / len(labeled),
                                    "reference": sum(correct(r, "reference") for r in labeled) / len(labeled)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-reference", choices=list(VARIANTS), metavar="MODEL",
                        help="Write the official FP32 reference of d3-nano or d3-mini")
    parser.add_argument("--questions", type=Path,
                        help="With --write-reference: a saved decision set to use instead of the built-in cases")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="With --write-reference: official precision; validation gates use float32")
    parser.add_argument("--baseline", type=Path,
                        help="With --asset: a reference of the same items at another precision to report beside")
    parser.add_argument("--smoke", action="store_true", help="Allow partial coverage; the report still distinguishes acceptance")
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.write_reference:
        build_reference(args.write_reference, args.reference, args.questions, args.dtype)
    else:
        if not args.asset or not args.out:
            parser.error("validation requires --asset and --out")
        report = validate(args.asset, args.reference, args.baseline, smoke=args.smoke)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
        raise SystemExit(0 if report["smoke_passed" if args.smoke else "passed"] else 1)


if __name__ == "__main__":
    main()
