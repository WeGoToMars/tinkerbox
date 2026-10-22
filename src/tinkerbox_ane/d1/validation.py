"""Official FP32 references and decision-level regression checks."""

import json
import math
from collections import defaultdict
from pathlib import Path

from tinkerbox_ane.d1 import REPO, REVISION
from tinkerbox_ane.validation import distribution


def reference_metadata() -> dict:
    return {"repository": REPO, "revision": REVISION, "implementation": "official", "dtype": "float32", "device": "cpu"}


def load_reference(path: Path) -> dict:
    document = json.loads(Path(path).read_text())
    if document.get("reference") != reference_metadata():
        raise ValueError("reference must use the pinned official FP32 CPU model")
    if not document.get("items"):
        raise ValueError("reference has no decisions")
    for item in document["items"]:
        if not item["ids"] or any(type(token) is not int or token < 0 for token in item["ids"]):
            raise ValueError("reference needs nonempty prompt token IDs")
        count = 2 if item["question"]["type"] == "noul" else len(item["question"]["criteria"])
        distribution(item["reference"], count)
    return document


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate(records: list[dict], tiers) -> dict:
    if not records:
        raise ValueError("cannot validate an empty run")
    groups = defaultdict(list)
    drifts, agreement, token_mismatches = [], 0, 0
    for record in records:
        reference = distribution(record["reference"])
        probabilities = distribution(record["probabilities"], len(reference))
        if not math.isfinite(record["latency_s"]) or record["latency_s"] <= 0:
            raise ValueError("decision latency must be positive and finite")
        if type(record["token_match"]) is not bool:
            raise ValueError("missing prompt token comparison")
        drift = [abs(a - b) for a, b in zip(reference, probabilities)]
        match = max(range(len(reference)), key=reference.__getitem__) == max(range(len(probabilities)), key=probabilities.__getitem__)
        agreement += match
        token_mismatches += not record["token_match"]
        drifts.extend(drift)
        groups[record["tier"]].append((record["latency_s"] * 1000, drift, match))
    by_tier = {}
    for tier, rows in sorted(groups.items()):
        errors = [error for _, drift, _ in rows for error in drift]
        times = [elapsed for elapsed, _, _ in rows]
        ratio = sum(match for _, _, match in rows) / len(rows)
        mean, worst = sum(errors) / len(errors), max(errors)
        by_tier[str(tier)] = {"decisions": len(rows), "agreement": ratio,
                              "mean_probability_drift": mean, "max_probability_drift": worst,
                              "p50_ms": percentile(times, .5), "p95_ms": percentile(times, .95),
                              "passed": ratio >= .97 and mean <= .01 and worst <= .05}
    ratio, mean, worst = agreement / len(records), sum(drifts) / len(drifts), max(drifts)
    missing = sorted(set(tiers) - groups.keys())
    return {"passed": token_mismatches == 0 and ratio >= .97 and mean <= .005 and worst <= .05
                      and not missing and all(row["passed"] for row in by_tier.values()),
            "decisions": len(records), "agreement": ratio, "mean_probability_drift": mean,
            "max_probability_drift": worst, "token_mismatches": token_mismatches,
            "missing_tiers": missing, "tiers": by_tier}
