"""VoxCPM2's text and isolated-reference/continuation sequence contract."""

import torch


def tokenize(tokenizer, text):
    # Match upstream mask_multichar_chinese_tokens, including its vocabulary
    # membership check, rather than enabling checkpoint-supplied Python code.
    words = tokenizer.tokenize(text)
    split = {token for token in tokenizer.get_vocab() if len(token) >= 2
             and all('\u4e00' <= c <= '\u9fff' for c in token)}
    tokens = []
    for token in words:
        clean = token.replace('▁', '')
        tokens.extend(list(clean) if clean in split else [token])
    return tokenizer.convert_tokens_to_ids(tokens)


def build_inputs(tokenizer, text, voice=None, *, context_positions):
    reference = voice.reference if voice is not None else None
    prompt = voice.continuation if voice is not None else None
    text = (voice.ref_text if prompt is not None else '') + text
    ids = tokenize(tokenizer, text) + [101]
    positions = len(ids) + (len(reference)+2 if reference is not None else 0) + (len(prompt) if prompt is not None else 0)
    if positions+4 > context_positions:
        raise ValueError(f'VoxCPM2 prompt needs {positions} positions plus audio; bundle limit is {context_positions}')
    features = torch.zeros(len(ids), 4, 64)
    audio = torch.zeros(len(ids), dtype=torch.bool)
    if reference is not None:
        n = len(reference)
        ids = [103] + [0]*n + [104] + ids
        features = torch.cat((torch.zeros(1, 4, 64), reference, torch.zeros(1, 4, 64), features))
        audio = torch.cat((torch.tensor([False]), torch.ones(n, dtype=torch.bool),
                           torch.tensor([False]), audio))
    if prompt is not None:
        ids += [0]*len(prompt)
        features = torch.cat((features, prompt))
        audio = torch.cat((audio, torch.ones(len(prompt), dtype=torch.bool)))
    return torch.tensor(ids), features, audio
