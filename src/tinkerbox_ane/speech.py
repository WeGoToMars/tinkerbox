"""Task-level speech requests, independent of any model architecture."""

import base64
import io
import math
import secrets
import time


def validate_fields(definitions, values):
    """Validate advertised controls without coercing one model's semantics into another's."""
    if not isinstance(values, dict):
        raise ValueError('Speech controls must be an object.')
    fields = {field['key']: field for field in definitions}
    for key, value in values.items():
        if key not in fields:
            raise ValueError(f'unknown speech control: {key}')
        field = fields[key]
        if value is None and field.get('optional'):
            continue
        kind = field['type']
        if kind == 'bool':
            valid = isinstance(value, bool)
        elif kind in ('enum', 'text'):
            valid = isinstance(value, str)
            if valid and kind == 'enum':
                valid = value in [choice['value'] for choice in field['choices']]
            if valid and field.get('required'):
                valid = bool(value.strip())
        else:
            valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(value) and (kind != 'int' or isinstance(value, int))
                     and ('min' not in field or value >= field['min'])
                     and ('max' not in field or value <= field['max']))
        if not valid:
            raise ValueError(f'invalid {key}: {value!r}')
    for key, field in fields.items():
        if field.get('required') and values.get(key, field.get('default')) in (None, ''):
            raise ValueError(f'{field["title"]} is required.')
    return dict(values)


def speech_controls(capabilities, mode, instruct, controls, has_reference):
    """Resolve one item's advertised synthesis mode and its native controls."""
    if mode is None:
        mode = (capabilities['instruction_mode'] if instruct else 'clone' if has_reference else 'auto')
    selected = next((item for item in capabilities['modes'] if item['id'] == mode), None)
    if selected is None:
        raise ValueError(f'Unsupported speech mode: {mode!r}')
    if instruct is not None and not isinstance(instruct, str):
        raise ValueError('instruct must be text')
    if instruct and not selected.get('instruction_title'):
        raise ValueError(f'{selected["title"]} does not accept an instruction.')
    if selected.get('instruction_required') and not (instruct and instruct.strip()):
        raise ValueError(f'{selected["instruction_title"]} is required.')
    reference = selected.get('reference')
    if has_reference and reference not in ('optional', 'required'):
        raise ValueError(f'{selected["title"]} does not accept a reference voice.')
    if reference == 'required' and not has_reference:
        raise ValueError('Choose reference audio or a saved voice prompt.')
    definitions = capabilities.get('controls', []) + selected.get('controls', [])
    controls = validate_fields(definitions, {} if controls is None else controls)
    defaults = {p['key']: p['default'] for p in definitions if p.get('default') is not None}
    defaults.update(controls)
    return selected, defaults


def synthesize(model, entry, request):
    """Generate an ordered batch of PCM WAVs for the app/automation protocol."""
    import soundfile as sf

    if request.get('operation') == 'create_voice':
        if not request.get('ref_audio'):
            raise ValueError('Choose reference audio to save a voice.')
        started = time.perf_counter()
        prompt = model.create_voice_clone_prompt(
            ref_audio=request['ref_audio'], ref_text=request.get('ref_text') or None,
            preprocess_prompt=request.get('preprocess_prompt', True))
        output = io.BytesIO()
        prompt.save(output)
        return dict(voicePrompt=base64.b64encode(output.getvalue()).decode(),
                    cloneMs=(time.perf_counter() - started) * 1000)
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
    clone_seconds = 0.0
    if any(item.get('ref_audio') or item.get('voice_clone_prompt') for item in items):
        # Upstream cannot mix cloned and automatic voices in one batch. Preserve
        # each item's voice and ordering through independent complete requests.
        audio, ane_seconds = [], 0.0
        for index, item in enumerate(items):
            arguments = {key: item[key] for key in ('language', 'instruct', 'duration', 'speed', 'mode', 'controls')
                         if item.get(key) is not None}
            preparing = time.perf_counter()
            if path := item.get('voice_clone_prompt'):
                arguments['voice_clone_prompt'] = model.load_voice_clone_prompt(path)
            elif path := item.get('ref_audio'):
                arguments['voice_clone_prompt'] = model.create_voice_clone_prompt(
                    ref_audio=path, ref_text=item.get('ref_text') or None,
                    preprocess_prompt=options.get('preprocess_prompt', True))
            clone_seconds += time.perf_counter() - preparing
            audio.extend(model.generate(text=item['text'], seed=seed + index,
                                        **arguments, **options))
            ane_seconds += model.ane_seconds
    else:
        arguments = {key: [item.get(key) for item in items]
                     for key in ('language', 'instruct', 'duration', 'speed', 'mode', 'controls')
                     if any(item.get(key) is not None for item in items)}
        audio = model.generate(text=texts, seed=seed, **arguments, **options)
        ane_seconds = model.ane_seconds
    results = []
    for waveform in audio:
        output = io.BytesIO()
        sf.write(output, waveform, model.sampling_rate, format='WAV', subtype='PCM_16')
        results.append(dict(wav=base64.b64encode(output.getvalue()).decode(),
                            sampleRate=model.sampling_rate, duration=len(waveform) / model.sampling_rate))
    elapsed = time.perf_counter() - started
    return dict(audio=results, elapsedMs=elapsed * 1000, cloneMs=clone_seconds * 1000,
                audioMs=(elapsed - clone_seconds) * 1000,
                aneMs=ane_seconds * 1000, seed=seed)
