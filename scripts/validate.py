"""Compare all saved FP32 decisions with the native runtime."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tinkerbox_ane.bundle import read_bundle
from tinkerbox_ane.d1 import DEFAULT_ASSET, KIND, load
from tinkerbox_ane.d1.validation import evaluate, load_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", nargs="?", type=Path, default=DEFAULT_ASSET)
    args = parser.parse_args()
    reference_path = Path("artifacts/reference.json")
    reference = load_reference(reference_path)
    spec = read_bundle(args.asset, KIND)
    tiers, width = spec["tiers"], spec["width"]

    def tier_for(tokens):
        padded = -(-tokens // width) * width
        return next(tier for tier in tiers if padded <= tier)

    records = []
    started_at = datetime.now(timezone.utc)
    with load(args.asset) as model:
        warmed = set()
        for item in reference["items"]:
            tier = tier_for(len(item["ids"]))
            if tier not in warmed:
                print(f"warmup: c{tier}", flush=True)
                model.probabilities(item["state"], [item["question"]])
                warmed.add(tier)
        for index, item in enumerate(reference["items"]):
            ids = model.tokens(item["state"], item["question"])
            start = time.perf_counter()
            probabilities = model.probabilities(item["state"], [item["question"]])[0]
            records.append({"index": index, "tokens": len(ids), "source": item["source"],
                            "label": item["label"], "reference": item["reference"],
                            "probabilities": probabilities, "token_match": ids == item["ids"],
                            "tier": tier_for(len(ids)), "latency_s": time.perf_counter() - start})
            if (index + 1) % 32 == 0 or index + 1 == len(reference["items"]):
                print(f"validated {index + 1}/{len(reference['items'])}", flush=True)
    report = evaluate(records, tiers)
    report.update(reference=reference["reference"], source_hash=spec["source_hash"],
                  reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  started_at=started_at.isoformat(), records=records)
    output = Path("artifacts/validation") / f"{started_at:%Y%m%dT%H%M%S%fZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['decisions']} decisions, "
          f"{report['agreement']:.1%} agreement, drift mean {report['mean_probability_drift']:.6f}, "
          f"max {report['max_probability_drift']:.6f}")
    for tier, row in report["tiers"].items():
        print(f"  {tier}: {row['p50_ms']:.1f} ms median, {row['decisions']} decisions")
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
