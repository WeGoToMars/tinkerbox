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


def checkpoint() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, revision=REVISION))


def load(asset=None):
    """Load the synchronous System One API; use as a context manager."""
    from tinkerbox_ane.d1.runtime import load as load_model
    from tinkerbox_ane.models import download

    return load_model(download(MODEL) if asset is None else Path(asset))
