"""Visible input token spans for native text editors, independent of inference tasks."""

from bisect import bisect_left, bisect_right

import regex


def token_spans(tokenizer, text):
    """UTF-16 spans for raw input; tokens sharing a displayed character are combined."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + 1 + (ord(char) > 0xFFFF))
    characters = list(regex.finditer(r"\X", text))
    starts = [character.start() for character in characters]
    ends = [character.end() for character in characters]
    spans = []
    for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        if start == end:
            continue
        start = offsets[starts[bisect_right(starts, start) - 1]]
        end = offsets[ends[bisect_left(ends, end)]]
        if spans and start < spans[-1]["end_utf16"]:
            spans[-1]["end_utf16"] = max(end, spans[-1]["end_utf16"])
            spans[-1]["token_ids"].append(token)
        else:
            spans.append(dict(start_utf16=start, end_utf16=end, token_ids=[token]))
    return spans
