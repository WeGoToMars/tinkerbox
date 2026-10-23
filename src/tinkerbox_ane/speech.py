"""Task-level speech requests, independent of any model architecture."""

import base64
import io
import math
import secrets
import time


def synthesize(model, entry, request):
    """Generate an ordered batch of PCM WAVs for the app/automation protocol."""
    import soundfile as sf

    if request.get('operation') == 'create_voice':
        if not request.get('ref_audio'):
            raise ValueError('Choose reference audio to save a voice.')
        prompt = model.create_voice_clone_prompt(
            ref_audio=request['ref_audio'], ref_text=request.get('ref_text') or None,
            preprocess_prompt=request.get('preprocess_prompt', True))
        output = io.BytesIO()
        prompt.save(output)
        return dict(voicePrompt=base64.b64encode(output.getvalue()).decode())
    if request.get('operation', 'generate') != 'generate':
        raise ValueError('Unknown speech operation.')

    items = request.get('items', [])
    if not items:
        raise ValueError('Provide at least one text to speak.')
    options = model.generation_parameters(request.get('parameters', {}))
    texts = [item.get('text', '') for item in items]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('Speech text must not be empty.')
    for key in ('duration', 'speed'):
        for item in items:
            value = item.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be positive and finite.')
    seed = request.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63):
        raise ValueError('Seed must be an integer between 0 and 2^63 - 1.')
    if seed is None:
        seed = secrets.randbits(63)
    started = time.perf_counter()
    if any(item.get('ref_audio') or item.get('voice_clone_prompt') for item in items):
        # Upstream cannot mix cloned and automatic voices in one batch. Preserve
        # each item's voice and ordering through independent complete requests.
        audio, ane_seconds = [], 0.0
        for index, item in enumerate(items):
            arguments = {key: item[key] for key in ('language', 'instruct', 'ref_audio', 'ref_text', 'duration', 'speed')
                         if item.get(key) is not None}
            if path := item.get('voice_clone_prompt'):
                arguments['voice_clone_prompt'] = model.load_voice_clone_prompt(path)
            audio.extend(model.generate(text=item['text'], seed=seed + index,
                                        **arguments, **options))
            ane_seconds += model.ane_seconds
    else:
        arguments = {key: [item.get(key) for item in items]
                     for key in ('language', 'instruct', 'duration') if any(item.get(key) for item in items)}
        arguments['speed'] = [item.get('speed') or 1.0 for item in items]
        audio = model.generate(text=texts, seed=seed, **arguments, **options)
        ane_seconds = model.ane_seconds
    results = []
    for waveform in audio:
        output = io.BytesIO()
        sf.write(output, waveform, model.sampling_rate, format='WAV', subtype='PCM_16')
        results.append(dict(wav=base64.b64encode(output.getvalue()).decode(),
                            sampleRate=model.sampling_rate, duration=len(waveform) / model.sampling_rate))
    return dict(audio=results, elapsedMs=(time.perf_counter() - started) * 1000,
                aneMs=ane_seconds * 1000, seed=seed)
