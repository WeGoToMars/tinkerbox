"""vllm-sr d3 decision models (Qwen3.5) on the Apple Neural Engine."""

from pathlib import Path

KIND = "qwen3.5-vl"
# Catalog model: checkpoint repository, pinned revision and attention capacities.
VARIANTS = {
    "d3-nano": ("vllm-sr/d3-nano", "462efc366229ea268cf4eed3738b3fc4415e4817",
                (256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768)),
    "d3-mini": ("vllm-sr/d3-mini", "bd0090dd758d32ba59ef5c749910a8e87fc4391f",
                (256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768)),
}
# Loading imports the checkpoint's Python code, so only these sources are trusted.
REVISIONS = {repo: revision for repo, revision, _ in VARIANTS.values()}
# Bundles carry that code; both revisions ship the same files (bundle.code_digest).
CODE = "b7fc4617c548a51ec19fc21d3bafc68721429d06a1f463fef625f3f6b9400e7f"
# What the runtime reads from the checkpoint's weights; bundles keep only these.
RUNTIME_TENSORS = ("language_model.embed_tokens.weight", "visual.pos_embed.weight")
RESIDUAL_SCALE = 64.0  # Keep ANE residual products above the subnormal threshold.
CHUNK = 128
# Short prompts run each stage in one call at the smallest of these widths that holds them: the DeltaNet
# scan over the whole prompt, then full attention. Longer prompts take the chunked path. The Neural
# Engine keeps one weight copy per width group in an asset (up to 64, up to 128, wider), so wider
# prompt functions, about a third faster for 129–320 tokens, would double each stage's resident weights.
PROMPT_WIDTHS = (128,)
INPUT, OUTPUT = "hidden_in", "hidden_out"
# Vision tower patch capacities; an image runs at the smallest that holds its patches.
PATCH_TIERS = (1024, 1536, 2048, 3072, 4096, 5120, 6400)
# The official processor's 1.6 MP budget: 1,600 tokens of 32 × 32 pixels.
MAX_IMAGE_TOKENS = PATCH_TIERS[-1] // 4

# Settings the app shows and sends with each request; each key is a writable attribute of the model.
SETTINGS = [
    {"key": "max_image_tokens", "title": "Maximum tokens per image", "type": "int", "min": 64, "max": MAX_IMAGE_TOKENS,
     "step": 64, "default": MAX_IMAGE_TOKENS,
     "help": "Images keep their aspect ratio and are resized to multiples of 32 pixels within this budget. "
             "Each 32×32 block uses one token; 1,600 tokens is the official 1.6 MP limit."},
]
# Fixed properties the app shows beside the settings.
INFO = [
    {"label": "Image input", "value": "Aspect-preserving, up to 1.6 MP"},
    {"label": "Questions", "value": "One pass each, in order"},
    {"label": "Chunk width", "value": f"{CHUNK} tokens"},
]


def checkpoint(repo: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    if REVISIONS.get(repo) != revision:
        raise ValueError(f"{repo}@{revision} is not a supported d3 checkpoint")
    return Path(snapshot_download(repo, revision=revision))


def load(asset=None, model: str = "d3-nano"):
    """Load the synchronous System One API; use as a context manager."""
    from tinkerbox_ane.d3.runtime import load as load_model
    from tinkerbox_ane.models import download

    return load_model(download(model) if asset is None else Path(asset))
