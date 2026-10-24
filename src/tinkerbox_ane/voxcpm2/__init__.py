"""VoxCPM2 speech synthesis with MiniCPM transformers on ANE."""

import os
from pathlib import Path

MODEL = 'voxcpm2'
KIND = 'voxcpm2-minicpm'
REPO = 'openbmb/VoxCPM2'
REVISION = '32279effe8c19989596f05d353d1447f51d9e915'
UPSTREAM_REVISION = 'cce58a5b59303c9bd63d12f23afe6a49a4a80c59'
SOURCE = Path('artifacts/voxcpm2/export')
from .parameters import PARAMETERS as SPEECH_PARAMETERS
SETTINGS = []
INFO = [dict(label='Synthesis', value='ANE · MiniCPM + local diffusion'),
        dict(label='Audio', value='48 kHz · CPU AudioVAE V2')]


def checkpoint():
    if local := os.environ.get('TINKERBOX_VOXCPM2_CHECKPOINT'):
        return Path(local)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPO, revision=REVISION,
                                  allow_patterns=['*.json', '*.safetensors', 'audiovae.pth']))


def load(asset):
    from .runtime import Speech
    return Speech(asset)
