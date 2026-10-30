"""The 60-second single-chunk speech case of the OmniVoice ANE gate."""

TEXT = ('The morning train arrived just as the first shops began to open. '
        'Mara stepped onto the platform with a small canvas bag and a notebook full of questions. '
        'She had come to the valley to record the stories of people who repaired old things. '
        'Her first appointment was with a clockmaker whose workshop stood beside the river. '
        'She followed the sound of running water through a narrow street, passing a bakery, '
        'a flower stall, and a bicycle leaning against a blue door. '
        'The clockmaker welcomed her with a cup of tea. His workbench held tiny wheels, '
        'glass faces, wooden boxes, and several clocks waiting patiently for attention. '
        'He explained that every repair began with listening. A clock might run too quickly, '
        'stop at the same hour each day, or lose its voice altogether. Before he touched a tool, '
        'he watched its movement and learned its habits. Mara wrote this down, then closed '
        'her notebook so she could listen more carefully.')
SETTINGS = dict(instruct='female, low pitch, british accent', duration=60, audio_chunk_threshold=120, num_step=32)
# A single seed is not a usable gate: upstream CPU FP16 itself exceeded 0.20 on
# two of five seeds. The mean over these seeds is gated instead.
SEEDS = (1, 2, 3, 5, 42)
MAX_MEAN_ERROR = .20


def character_error(actual, reference):
    actual = ''.join(c for c in actual.casefold() if c.isalnum())
    reference = ''.join(c for c in reference.casefold() if c.isalnum())
    previous = list(range(len(reference)+1))
    for i, a in enumerate(actual, 1):
        current = [i]
        for j, b in enumerate(reference, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]/max(len(reference), 1)
