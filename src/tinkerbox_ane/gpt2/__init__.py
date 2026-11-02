"""The four original GPT-2 text-completion variants on Apple Neural Engine."""

from pathlib import Path

KIND = 'gpt2'
VARIANTS = {
    'gpt2': ('openai-community/gpt2', '607a30d783dfa663caf39e06633721c8d4cfcd7e'),
    'gpt2-medium': ('openai-community/gpt2-medium', '6dcaa7a952f72f9298047fd5137cd6e4f05f41da'),
    'gpt2-large': ('openai-community/gpt2-large', '32b71b12589c2f8d625668d2335a01cac3249519'),
    'gpt2-xl': ('openai-community/gpt2-xl', '15ea56dee5df4983c59b2538573817e1667135e2'),
}
SETTINGS = []
INFO = [dict(label='Context', value='1,024 tokens'), dict(label='Batch sizes', value='1, 4, 8')]


def checkpoint(model='gpt2'):
    from huggingface_hub import snapshot_download

    repo, revision = VARIANTS[model]
    return Path(snapshot_download(repo, revision=revision, allow_patterns=[
        'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
        'vocab.json', 'merges.txt', 'model.safetensors', 'LICENSE', 'README.md']))


def load(asset):
    """Load the synchronous text-completion API; use as a context manager."""
    from tinkerbox_ane.generation import TextGeneration
    from tinkerbox_ane.gpt2.runtime import Decoder

    return TextGeneration(Decoder(asset))
