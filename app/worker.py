"""One loaded model, one JSON reply per request."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

output = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)  # Native code and child processes print to fd 1 too; keep the reply channel clean.


def send(reply):
    output.write(json.dumps(reply, allow_nan=False) + "\n")
    output.flush()


def question(request):
    kind = request["type"]
    result = {"type": kind, "instructions": request["instructions"]}
    if kind == "choice":
        criteria = request["criteria"]
        if len(criteria) < 2 or len(set(criteria)) != len(criteria):
            raise ValueError("Choice needs at least two unique options.")
        result["criteria"] = dict.fromkeys(criteria)
    elif kind == "score":
        criteria = request["criteria"]
        if not 2 <= len(criteria) <= 10:
            raise ValueError("Score needs 2–10 ordered levels.")
        result["criteria"] = criteria
    elif kind != "noul":
        raise ValueError(f"Unknown question type: {kind}")
    return result


def labeled(answer):
    """The answer with its probabilities as an ordered list of labels and values."""
    kind = answer["type"]
    if kind == "noul":
        probabilities = [("Yes", answer["noul"]), ("No", 1 - answer["noul"])]
    else:
        probabilities = [(f"{key} · {answer['legend'][key]}" if kind == "score" else key, value)
                         for key, value in answer["probabilities"].items()]
    answer = {key: value for key, value in answer.items() if key != "legend"}
    answer["probabilities"] = [{"label": label, "value": value} for label, value in probabilities]
    return answer


def decide(model, entry, request):
    """Every question about every item: one input and several questions share a single tree pass,
    and one question over many inputs is packed into one batch."""
    from PIL import Image, ImageOps

    questions = {str(index): question(item) for index, item in enumerate(request["questions"])}
    if not questions or not request["items"]:
        raise ValueError("Ask at least one question about at least one input.")
    batch = []
    for item in request["items"]:
        images = [ImageOps.exif_transpose(Image.open(path)) for path in item.get("images", [])]
        if images and "image" not in entry["inputs"]:
            raise ValueError(f"{entry['name']} does not accept images.")
        state = item["state"] if item["state"].strip() or not images else None
        batch.append((state, questions, images))
    if request.get("count"):
        # What a run would read, without running the model.
        counts = [model.count_tokens(*item) for item in batch]
        return {"counts": [{"tokens": count["input_tokens"], "tiles": count["image_tiles"],
                            "images": count["images"]} for count in counts]}
    started = time.perf_counter()
    results = model.system_one_batch(batch)
    elapsed = (time.perf_counter() - started) * 1000
    return {"results": [{"answers": [labeled(result["answers"][name]) for name in questions],
                         "tokens": result["usage"]["input_tokens"],
                         # Models that count their compute report Neural Engine time and FLOPs.
                         "flops": result["usage"].get("flops"),
                         "neuralEngineMs": result["usage"].get("neural_engine_ms")} for result in results],
            "elapsedMs": elapsed}


# One request handler per catalog task; a new task adds its own instead of branching inside another.
HANDLERS = {"decision": decide}


def main():
    started = time.perf_counter()
    from tinkerbox_ane import models
    from tinkerbox_ane.host import device_architecture

    model_id = os.environ.get("TINKERBOX_MODEL") or models.CATALOG[0]["id"]
    entry = models.entry(model_id)
    if entry["task"] not in HANDLERS:
        raise ValueError(f"This app cannot run {entry['task']} models.")
    handle = HANDLERS[entry["task"]]
    package = importlib.import_module(entry["module"])
    local = os.environ.get("TINKERBOX_ASSET")
    if local:
        asset = Path(local)
    else:
        send({"status": "Fetching model files…"})
        asset = models.download(model_id, lambda done, total: send(
            {"status": "Downloading model…", "progress": min(done / total, 1.0) if total else None}))
    send({"status": "Loading model…"})
    cpu = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    with models.load(model_id, asset) as model:
        send({"ready": True, "modelId": model_id, "modelName": f"{entry['publisher']} {entry['name']}",
              "cpuModel": cpu, "architecture": device_architecture(),
              "questionTrees": entry.get("question_trees", False),
              "settings": package.SETTINGS, "info": package.INFO,
              "setupMs": (time.perf_counter() - started) * 1000})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                # Every request carries its settings; missing ones fall back to their defaults.
                for setting in package.SETTINGS:
                    setattr(model, setting["key"], request.get("settings", {}).get(setting["key"], setting["default"]))
                send(handle(model, entry, request))
            except Exception as error:
                send({"error": str(error)})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        send({"error": str(error)})
        sys.exit(1)
