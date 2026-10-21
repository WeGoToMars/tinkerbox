"""Model catalog shared by the Python API and the app.

`models.json` lists every model this package can run. The app fetches the same file from the
repository to show models it does not ship yet, disabled; its bundled entries always describe what it runs.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("models.json").read_text())["models"]


def entry(model: str) -> dict:
    for item in CATALOG:
        if item["id"] == model:
            return item
    raise ValueError(f"unknown model {model!r}; available: {', '.join(item['id'] for item in CATALOG)}")


def download(model: str, progress=None) -> Path:
    """Fetch the AOT bundle for this Mac; `progress(done_bytes, total_bytes)` reports the download."""
    from tinkerbox_ane.bundle import download_bundle

    item = entry(model)
    return download_bundle(item["id"], item["repo"], item["revision"], progress)


def install(model: str, bundle: Path) -> Path:
    """Use a locally compiled bundle as the catalog revision, e.g. before it is published."""
    from tinkerbox_ane.bundle import install_bundle

    item = entry(model)
    kind = importlib.import_module(item["module"]).KIND
    return install_bundle(item["id"], item["revision"], bundle, kind)


def load(model: str, asset=None):
    """Load a model's System One API from a local bundle, or download it; use as a context manager."""
    return importlib.import_module(entry(model)["module"]).load(download(model) if asset is None else asset)
