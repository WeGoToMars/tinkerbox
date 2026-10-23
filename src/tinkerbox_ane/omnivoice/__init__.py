"""OmniVoice speech synthesis: ANE diffusion, CPU codec and preprocessing."""

from pathlib import Path

MODEL = 'omnivoice'
KIND = 'omnivoice-qwen3'
REPO = 'k2-fsa/OmniVoice'
REVISION = 'c5fdb5ccb189668d56333f77ba2629f4cd7535f4'
UPSTREAM_REVISION = '08be0b4ccbac3e13e374e86fbfead4b4cac343e2'
SOURCE = Path('artifacts/omnivoice/export')
SETTINGS = []  # Speech generation settings travel on each request, including floats and booleans.
INFO = [
    {'label': 'Synthesis', 'value': 'ANE · Qwen3 diffusion'},
    {'label': 'Audio codec', 'value': 'CPU · 24 kHz'},
]
from .parameters import PARAMETERS as SPEECH_PARAMETERS


def checkpoint():
    import os
    from huggingface_hub import snapshot_download
    if local := os.environ.get('TINKERBOX_OMNIVOICE_CHECKPOINT'):
        return Path(local)
    return Path(snapshot_download(REPO, revision=REVISION))


def load(asset=None, *, asr_model_name=None):
    from .runtime import Speech
    from tinkerbox_ane.models import download
    return Speech(download(MODEL) if asset is None else Path(asset), asr_model_name=asr_model_name)
