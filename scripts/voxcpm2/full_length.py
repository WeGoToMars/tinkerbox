"""Long public-API generation with reference conditioning and content validation."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from tinkerbox_ane import load
from tinkerbox_ane.transcription import transcribe
from gate import character_error

TEXT = '''The morning train arrived just as the first shops began to open. Mara stepped onto the platform with a small canvas bag and a notebook full of questions. She had come to the valley to record the stories of people who repaired old things. Her first appointment was with a clockmaker whose workshop stood beside the river. She followed the sound of running water through a narrow street, passing a bakery, a flower stall, and a bicycle leaning against a blue door.

The clockmaker welcomed her with a cup of tea. His workbench held tiny wheels, glass faces, wooden boxes, and several clocks waiting patiently for attention. He explained that every repair began with listening. A clock might run too quickly, stop at the same hour each day, or lose its voice altogether. Before he touched a tool, he watched its movement and learned its habits. Mara wrote this down, then closed her notebook so she could listen more carefully.

At noon they walked to the bakery. The baker was repairing a wooden chair that had belonged to her grandmother. One leg had split, but the seat was still strong. She showed Mara the marks left by years of use and explained why she wanted to keep them. A repaired object did not need to look new. It needed to be useful again, while keeping the parts of its history that mattered. Outside, a delivery bicycle rattled over the stones toward the market.

In the afternoon Mara visited the bicycle shop. The mechanic had arranged old frames along the wall and sorted spare parts into shallow drawers. A young customer arrived with a loose wheel and a worried expression. Together they checked the tire, tightened the axle, and tested the brakes. The mechanic asked the customer to explain each step before trying it. By the time the bicycle was ready, the worried expression had become a smile. Mara asked whether teaching always took longer. The mechanic said it usually saved time in the end.

Later, clouds gathered above the hills. Mara took shelter in a small library where a volunteer was mending the torn pages of a picture book. The volunteer used thin paper, a soft brush, and very little paste. Children had borrowed the book so often that its corners had become round. They still asked for it by name. The volunteer said that a careful repair could make room for another hundred readers. Rain tapped against the windows while Mara helped carry the finished books back to their shelves.

When the rain stopped, she returned to the river. The clockmaker was closing his workshop, the baker was sweeping her doorway, and the mechanic was wheeling a bicycle inside. Mara sat on a bench and read her notes. Each person had spoken about a different kind of work, yet the same ideas appeared again and again: pay attention, understand what came before, and leave something ready for the next person. She opened to a fresh page and wrote the final words of her story. Tomorrow, she would begin by listening.'''

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--reference', type=Path, default=Path('artifacts/voxcpm2/reference/reference-16k.wav'))
parser.add_argument('--out', type=Path, default=Path('artifacts/voxcpm2/full-length'))
parser.add_argument('--reference-repeats', type=int, default=1)
parser.add_argument('--mode', choices=['clone', 'continuation', 'clone_continuation'], default='clone')
parser.add_argument('--paragraphs', type=int, default=6)
parser.add_argument('--max-patches', type=int, default=4096)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
text = '\n\n'.join(TEXT.split('\n\n')[:args.paragraphs])
torch.set_num_threads(4)
waveform, rate = sf.read(args.reference)
reference = args.out/'reference.wav'
sf.write(reference, np.tile(waveform, args.reference_repeats), rate, subtype='FLOAT')
report = dict(asset=str(args.asset), text=text, reference_seconds=len(waveform)*args.reference_repeats/rate,
              seed=23, diffusion_steps=10, mode=args.mode, max_patches=args.max_patches,
              minimum_seconds=60, passed=False)
with load('voxcpm2', args.asset) as model:
    report['context_positions'] = model.capabilities['context_positions']
    started = time.monotonic()
    try:
        speech = model.generate(text, mode=args.mode, ref_audio=str(reference),
            ref_text=None if args.mode == 'clone' else 'Hello! Welcome to Tinkerbox. '*args.reference_repeats, seed=23,
            max_decode_steps=args.max_patches)[0]
        report.update(seconds=len(speech)/model.sampling_rate, elapsed=time.monotonic()-started,
                      finite=bool(np.isfinite(speech).all()))
        path = args.out/'speech.wav'
        sf.write(path, speech, model.sampling_rate, subtype='FLOAT')
        report['transcript'] = transcribe(path, locale='en-US')
        report['character_error'] = character_error(report['transcript'], text)
        report['passed'] = report['context_positions'] == 8192 and report['seconds'] > 60 and report['finite'] and report['character_error'] <= 0.20
    except (ValueError, RuntimeError) as error:
        report.update(error=str(error), elapsed=time.monotonic()-started)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report['passed'] else 1)
