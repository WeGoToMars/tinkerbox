"""Public speech behavior against a real compiled VoxCPM2 bundle."""

import base64
import io
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def speech():
    asset = os.environ.get('TINKERBOX_VOXCPM2_TEST_ASSET')
    if not asset or not (Path(asset)/'model.json').is_file():
        pytest.skip('requires a compiled VoxCPM2 bundle')
    from tinkerbox_ane.voxcpm2.runtime import Speech
    with Speech(asset) as model:
        yield model


def test_modes_batch_replay_and_request_limits(speech):
    from tinkerbox_ane.speech import synthesize

    request = dict(items=[dict(text='Hello! Welcome to Tinkerbox.', mode='auto'),
                          dict(text='A familiar voice can tell a new story.', mode='design',
                               instruct='A warm, calm female voice')], seed=42)
    first = synthesize(speech, {}, request)
    assert len(first['audio']) == 2
    assert all(item['sampleRate'] == 48000 and item['duration'] > 0 for item in first['audio'])
    assert first['audio'] == synthesize(speech, {}, request)['audio']
    standalone = synthesize(speech, {}, dict(items=[request['items'][0]], seed=42))
    assert first['audio'][0] == standalone['audio'][0]
    for arguments in [dict(duration=1), dict(speed=1.2), dict(language='en'),
                      dict(mode='design'), dict(controls={'speed': 'fast'})]:
        with pytest.raises(ValueError):
            speech.generate('Hello.', **arguments)
    with pytest.raises(ValueError, match='context|limit'):
        speech.generate('A long input. '*1000, seed=42)
    assert standalone['audio'] == synthesize(speech, {}, dict(items=[request['items'][0]], seed=42))['audio']


def test_saved_reference_and_continuation_modes(speech, tmp_path):
    import soundfile as sf
    from tinkerbox_ane.speech import synthesize

    text = 'Hello! Welcome to Tinkerbox.'
    waveform = speech.generate(text, seed=42)[0]
    reference = tmp_path/'reference.wav'
    sf.write(reference, waveform, speech.sampling_rate)
    for mode in ('clone', 'continuation', 'clone_continuation'):
        # Reference-only cloning intentionally has no transcript.
        transcript = None if mode == 'clone' else text
        reply = synthesize(speech, {}, dict(operation='create_voice', mode=mode,
            ref_audio=str(reference), ref_text=transcript, preprocess_prompt=False))
        saved = tmp_path/f'{mode}.pt'
        saved.write_bytes(base64.b64decode(reply['voicePrompt']))
        prompt = speech.load_voice_clone_prompt(saved)
        direct = speech.generate('Every sentence should reach its final word.',
                                 voice_clone_prompt=prompt, mode=mode, seed=23)[0]
        task = synthesize(speech, {}, dict(items=[dict(text='Every sentence should reach its final word.',
            voice_clone_prompt=str(saved), mode=mode)], seed=23))
        decoded, rate = sf.read(io.BytesIO(base64.b64decode(task['audio'][0]['wav'])))
        assert rate == 48000 and np.isfinite(decoded).all() and np.max(np.abs(decoded)) > 0
        assert np.allclose(decoded, direct, atol=1/32768)
        if mode != 'clone':
            with pytest.raises(ValueError, match='conditioning'):
                speech.generate('Hello.', voice_clone_prompt=prompt, mode='clone', seed=42)
