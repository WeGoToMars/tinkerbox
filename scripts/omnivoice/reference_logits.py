"""Capture upstream OmniVoice CPU FP16 logits without importing the ANE port."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from omnivoice import OmniVoice
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/omnivoice/checkpoint'))
parser.add_argument('--out', type=Path, default=Path('artifacts/omnivoice/chunked-reference'))
parser.add_argument('--lengths', type=int, nargs='+', default=[96, 1100])
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
model = OmniVoice.from_pretrained(args.checkpoint, train=True, dtype=torch.float16,
                                  attn_implementation='sdpa').eval().requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
report = dict(package=importlib.metadata.version('omnivoice'), torch=torch.__version__,
              device='cpu', dtype='float16', cases={})
with torch.inference_mode():
    for length in args.lengths:
        generator = torch.Generator().manual_seed(42)
        ids = torch.randint(0, model.config.audio_vocab_size, (1, 8, length), generator=generator)
        text = tokenizer('Hello! Welcome to Tinkerbox. A familiar voice can tell a new story.',
                         return_tensors='pt').input_ids
        size = min(text.shape[1], length)
        ids[:, :, :size] = text[:, :size].unsqueeze(1)
        mask = torch.ones((1, length), dtype=torch.bool)
        mask[:, :size] = False
        attention = torch.ones((1, 1, length, length), dtype=torch.bool)
        embeddings = model._prepare_embed_inputs(ids, mask)
        started = time.monotonic()
        output = model(input_ids=ids, audio_mask=mask, attention_mask=attention).logits
        torch.save(dict(embeddings=embeddings, logits=output, attention_mask=attention), args.out/f'{length}.pt')
        report['cases'][str(length)] = dict(seconds=time.monotonic()-started,
                                          finite=bool(torch.isfinite(output).all()))
        print(length, report['cases'][str(length)], flush=True)
(args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
