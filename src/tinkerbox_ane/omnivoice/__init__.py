"""OmniVoice speech synthesis: ANE diffusion and codec decoder, CPU codec encoder and preprocessing."""

from pathlib import Path

from .parameters import CAPABILITIES as SPEECH_CAPABILITIES, PARAMETERS as SPEECH_PARAMETERS  # noqa: F401

MODEL = "omnivoice"
KIND = "omnivoice-qwen3"
REPO = "k2-fsa/OmniVoice"
REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
UPSTREAM_REVISION = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
SOURCE = Path("artifacts/omnivoice/export")
SETTINGS = []  # Speech generation settings travel on each request, including floats and booleans.
INFO = [
    {"label": "Synthesis", "value": "ANE · Qwen3 diffusion"},
    {"label": "Audio codec", "value": "ANE decoder · 24 kHz"},
]


# OmniVoice's CC BY-NC 4.0 LICENSE must already be in the staging or --metadata directory.
LICENSE_FILE = "LICENSE"


def upstream(model):
    """The upstream checkpoint an export of `model` records; publishing checks it."""
    return dict(repo=REPO, revision=REVISION)


def checkpoint():
    import os
    from huggingface_hub import snapshot_download

    if local := os.environ.get("TINKERBOX_OMNIVOICE_CHECKPOINT"):
        return Path(local)
    return Path(snapshot_download(REPO, revision=REVISION))


def load(asset):
    """Load the speech API; use as a context manager."""
    from .runtime import Speech

    return Speech(Path(asset))
