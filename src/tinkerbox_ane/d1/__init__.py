"""LiquidAI d1-3B on the Apple Neural Engine."""

from pathlib import Path

from tinkerbox_ane.models import entry

MODEL = "d1-3b"
KIND = "d1ane-fused"
REPO = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
ARTIFACT_REPO, ARTIFACT_REVISION = entry(MODEL)["repo"], entry(MODEL)["revision"]
RESIDUAL_SCALE = 64.0  # Keep ANE residual products above the subnormal threshold.
CHUNK = 128
CONTEXT = 32768
TIERS = (256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768)
SOURCE = Path("artifacts/build/d1_fp16_fused128_ctx32768")
DEFAULT_ASSET = Path("artifacts/d1_fp16_fused128_ctx32768")
INPUT, OUTPUT = "hidden_in", "hidden_out"
# Bounds vision passes per image; 6 keeps the tile counts of the former one-megapixel cap.
MAX_IMAGE_TILES = 6

# Settings the app shows and sends with each request; each key is a writable attribute of the model.
SETTINGS = [
    {"key": "max_image_tiles", "title": "Image tiles", "type": "int", "min": 1, "max": 10,
     "default": MAX_IMAGE_TILES,
     "help": "Most 512×512 tiles per image. Larger pictures get the grid closest to their shape within "
             "this limit, so a 4:3 photo stops at 3×2, plus a thumbnail; each tile adds a vision pass and "
             "about 256 tokens. 1 never splits."},
]
# Fixed properties the app shows beside the settings.
INFO = [
    {"label": "Image tile", "value": "512 × 512, 256 tokens"},
    {"label": "Thumbnail", "value": "Added to split images"},
    {"label": "Chunk width", "value": f"{CHUNK} tokens"},
    {"label": "Context", "value": f"{CONTEXT // 1024}K tokens"},
]


def checkpoint() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, revision=REVISION))


def load(asset=None):
    """Load the synchronous System One API; use as a context manager."""
    from tinkerbox_ane.d1.runtime import load as load_model
    from tinkerbox_ane.models import download

    return load_model(download(MODEL) if asset is None else Path(asset))
