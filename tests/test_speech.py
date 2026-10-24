"""Public speech behavior with the real local ANE bundle and codec."""

import io
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def speech():
    asset = os.environ.get('TINKERBOX_SPEECH_TEST_ASSET')
    weights = os.environ.get('TINKERBOX_OMNIVOICE_CHECKPOINT')
    if not asset or not weights or not (Path(asset) / 'model.json').is_file():
        pytest.skip('requires a compiled OmniVoice bundle and pinned checkpoint')
    from tinkerbox_ane.omnivoice.runtime import Speech
    with Speech(asset, weights=weights) as model:
        yield model


def test_speech_modes_batch_order_and_request_isolation(speech, tmp_path):
    import soundfile as sf
    from tinkerbox_ane.speech import synthesize

    options = dict(num_step=4, position_temperature=0, class_temperature=0,
                   postprocess_output=False, pad_duration=0, fade_duration=0)
    first = speech.generate('Hello.', duration=1.0, seed=42, **options)[0]
    native = speech.generate('Hello.', mode='auto', controls={'duration': 1.0}, seed=42, **options)[0]
    assert np.array_equal(native, first)
    assert len(first) == speech.sampling_rate
    assert np.isfinite(first).all() and np.max(np.abs(first)) > 0
    request = dict(items=[dict(text='Hello.', duration=1.0), dict(text='Welcome.', duration=1.4)],
                   parameters=options, seed=42)
    reply = synthesize(speech, {}, request)
    assert reply['cloneMs'] == 0 and reply['audioMs'] > 0
    decoded = [sf.read(io.BytesIO(__import__('base64').b64decode(item['wav'])))[0] for item in reply['audio']]
    assert [len(wave) for wave in decoded] == [24000, 33600]
    assert np.allclose(decoded[0], first, atol=1 / 32768)
    designed = speech.generate('Welcome.', instruct='female, low pitch', duration=1.0, seed=42, **options)[0]
    assert np.isfinite(designed).all()
    path = tmp_path / 'reference.wav'
    sf.write(path, first, speech.sampling_rate)
    prompt = speech.create_voice_clone_prompt(str(path), ref_text='Hello.', preprocess_prompt=False)
    saved = tmp_path / 'voice.pt'
    prompt.save(saved)
    cloned = speech.generate('Welcome.', voice_clone_prompt=speech.load_voice_clone_prompt(saved),
                             duration=1.0, seed=42, **options)[0]
    assert len(cloned) == speech.sampling_rate and np.isfinite(cloned).all()
    mixed = synthesize(speech, {}, dict(
        items=[dict(text='Welcome.', voice_clone_prompt=str(saved), duration=1.0),
               dict(text='Hello.', duration=1.4)], parameters=options, seed=42))
    assert [item['duration'] for item in mixed['audio']] == [1.0, 1.4]
    assert mixed['cloneMs'] > 0 and mixed['audioMs'] > 0
    assert mixed['elapsedMs'] == pytest.approx(mixed['cloneMs'] + mixed['audioMs'])
    mixed_clone = sf.read(io.BytesIO(__import__('base64').b64decode(mixed['audio'][0]['wav'])))[0]
    assert np.allclose(mixed_clone, cloned, atol=1 / 32768)
    assert np.array_equal(speech.generate('Hello.', duration=1.0, seed=42, **options)[0], first)
    stochastic = dict(options, position_temperature=5)
    unseeded = synthesize(speech, {}, dict(items=[dict(text='Hello.', duration=1.0)], parameters=stochastic))
    replay = synthesize(speech, {}, dict(items=[dict(text='Hello.', duration=1.0)],
                                        parameters=stochastic, seed=unseeded['seed']))
    assert replay['audio'] == unseeded['audio']
    with pytest.raises(ValueError, match='positive'):
        synthesize(speech, {}, dict(items=[dict(text='Hello.', speed=0)]))
    assert np.array_equal(speech.generate('Hello.', duration=1.0, seed=42, **options)[0], first)


def test_short_speech_is_intelligible_and_can_be_transcribed_for_cloning(speech):
    import re

    for text, instruction, seed in [('Hello! Welcome to Tinkerbox.', 'female, low pitch, british accent', 42),
                                    ('fuck yeah', None, 7), ('Hello', None, 7)]:
        audio = speech.generate(text, instruct=instruction, seed=seed)[0]
        transcript = speech.transcribe((audio, speech.sampling_rate))
        expected = re.sub(r'[^a-z]', '', text.lower())
        assert expected in re.sub(r'[^a-z]', '', transcript.lower())
    with pytest.raises(RuntimeError, match='manually'):
        speech.transcribe((audio, speech.sampling_rate), locale='zz-ZZ')
    prompt = speech.create_voice_clone_prompt((audio, speech.sampling_rate), preprocess_prompt=False)
    # Upstream normalizes reference loudness before ASR, which can change punctuation.
    assert re.sub(r'[^a-z]', '', prompt.ref_text.lower()) == re.sub(r'[^a-z]', '', transcript.lower())
    explicit = speech.generate('Hello.', instruct='female, low pitch', duration=1,
                               num_step=4, postprocess_output=False,
                               pad_duration=0, fade_duration=0, seed=42)[0]
    assert len(explicit) == speech.sampling_rate


def test_installed_speech_runs_without_an_upstream_checkpoint(speech, tmp_path, monkeypatch):
    """A locally installed bundle carries the CPU lookup tables and nested audio codec."""
    import json
    import tinkerbox_ane
    from tinkerbox_ane import bundle, models
    from tinkerbox_ane.omnivoice import KIND

    source = Path(os.environ['TINKERBOX_SPEECH_TEST_ASSET'])
    portable = tmp_path / 'portable'
    portable.mkdir()
    spec = bundle.read_bundle(source, KIND)
    bundle.clone(source / spec['asset'], portable / spec['asset'])
    cpu = bundle.write_checkpoint(Path(os.environ['TINKERBOX_OMNIVOICE_CHECKPOINT']), portable,
                                  ['llm.embed_tokens.weight', 'audio_embeddings.weight'])
    spec['checkpoint_hash'] = bundle.checkpoint_digest(cpu)
    (portable / 'model.json').write_text(json.dumps(spec))
    monkeypatch.setattr(bundle, 'CACHE', tmp_path / 'cache')
    installed = models.install('omnivoice', portable)
    monkeypatch.delenv('TINKERBOX_OMNIVOICE_CHECKPOINT')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    options = dict(duration=1, num_step=4, position_temperature=0, class_temperature=0,
                   postprocess_output=False, pad_duration=0, fade_duration=0, seed=42)
    expected = speech.generate('Hello.', **options)[0]
    with tinkerbox_ane.load('omnivoice', installed) as model:
        actual = model.generate('Hello.', **options)[0]
        assert model.sampling_rate == 24000
        assert len(actual) == 24000 and np.array_equal(actual, expected)
