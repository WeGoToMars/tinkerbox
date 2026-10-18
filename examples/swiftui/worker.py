"""One loaded model, one JSON reply per decision."""

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


def main():
    started = time.perf_counter()
    from PIL import Image, ImageOps

    from tinkerbox_ane import models
    from tinkerbox_ane.host import device_architecture

    model_id = os.environ.get("TINKERBOX_MODEL") or models.CATALOG[0]["id"]
    entry = models.entry(model_id)
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
              "setupMs": (time.perf_counter() - started) * 1000})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                kind = request["type"]
                images = [ImageOps.exif_transpose(Image.open(path)) for path in request.get("images", [])]
                if images and "image" not in entry["inputs"]:
                    raise ValueError(f"{entry['name']} does not accept images.")
                state = request["state"] if request["state"].strip() or not images else None
                started = time.perf_counter()
                result = model.system_one(state, {"decision": question(request)}, images=images)
                elapsed = (time.perf_counter() - started) * 1000
                answer = result["answers"]["decision"]
                if kind == "noul":
                    probabilities = [("Yes", answer["noul"]), ("No", 1 - answer["noul"])]
                else:
                    probabilities = [
                        (f"{key} · {answer['legend'][key]}" if kind == "score" else key, value)
                        for key, value in answer["probabilities"].items()
                    ]
                send({"answer": answer,
                      "probabilities": [{"label": label, "value": value} for label, value in probabilities],
                      "tokens": result["usage"]["input_tokens"], "elapsedMs": elapsed})
            except Exception as error:
                send({"error": str(error)})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        send({"error": str(error)})
        sys.exit(1)
