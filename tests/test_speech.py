"""Public speech behavior with the real local ANE bundle and codec."""

import io
import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def asset(bundle):
    return bundle("omnivoice", "TINKERBOX_SPEECH_TEST_ASSET")


@pytest.fixture(scope="module")
def speech(asset):
    from tinkerbox_ane.omnivoice.runtime import Speech

    with Speech(asset) as model:
        yield model


def test_speech_modes_batch_order_and_request_isolation(speech, tmp_path):
    import soundfile as sf
    from tinkerbox_ane.speech import synthesize

    options = dict(
        num_step=4,
        position_temperature=0,
        class_temperature=0,
        postprocess_output=False,
        pad_duration=0,
        fade_duration=0,
    )
    first = speech.generate("Hello.", duration=1.0, seed=42, **options)[0]
    native = speech.generate("Hello.", mode="auto", controls={"duration": 1.0}, seed=42, **options)[0]
    assert np.array_equal(native, first)
    assert len(first) == speech.sampling_rate
    assert np.isfinite(first).all() and np.max(np.abs(first)) > 0
    request = dict(
        items=[dict(text="Hello.", duration=1.0), dict(text="Welcome.", duration=1.4)], parameters=options, seed=42
    )
    reply = synthesize(speech, {}, request)
    assert reply["cloneMs"] == 0 and reply["audioMs"] > 0
    with pytest.raises(ValueError, match="Streaming"):
        synthesize(speech, {}, dict(request, items=request["items"][:1], stream=True), emit=print)
    decoded = [sf.read(io.BytesIO(__import__("base64").b64decode(item["wav"])))[0] for item in reply["audio"]]
    assert [len(wave) for wave in decoded] == [24000, 33600]
    assert np.allclose(decoded[0], first, atol=1 / 32768)
    designed = speech.generate("Welcome.", instruct="female, low pitch", duration=1.0, seed=42, **options)[0]
    assert np.isfinite(designed).all()
    path = tmp_path / "reference.wav"
    sf.write(path, first, speech.sampling_rate)
    prompt = speech.create_voice_clone_prompt(str(path), ref_text="Hello.", preprocess_prompt=False)
    saved = tmp_path / "voice.pt"
    prompt.save(saved)
    cloned = speech.generate(
        "Welcome.", voice_clone_prompt=speech.load_voice_clone_prompt(saved), duration=1.0, seed=42, **options
    )[0]
    assert len(cloned) == speech.sampling_rate and np.isfinite(cloned).all()
    mixed = synthesize(
        speech,
        {},
        dict(
            items=[
                dict(text="Welcome.", voice_clone_prompt=str(saved), duration=1.0),
                dict(text="Hello.", duration=1.4),
            ],
            parameters=options,
            seed=42,
        ),
    )
    assert [item["duration"] for item in mixed["audio"]] == [1.0, 1.4]
    assert mixed["cloneMs"] > 0 and mixed["audioMs"] > 0
    assert mixed["elapsedMs"] == pytest.approx(mixed["cloneMs"] + mixed["audioMs"])
    mixed_clone = sf.read(io.BytesIO(__import__("base64").b64decode(mixed["audio"][0]["wav"])))[0]
    assert np.allclose(mixed_clone, cloned, atol=1 / 32768)
    # The direct Python API has the same ordered per-item voice contract as the worker.
    direct = speech.generate(
        ["Welcome.", "Hello."],
        mode=["clone", "auto"],
        voice_clone_prompt=[prompt, None],
        duration=[1.0, 1.4],
        seed=42,
        **options,
    )
    for actual, encoded in zip(direct, mixed["audio"]):
        expected = sf.read(io.BytesIO(__import__("base64").b64decode(encoded["wav"])))[0]
        assert np.allclose(actual, expected, atol=1 / 32768)
    stochastic = dict(options, position_temperature=5)
    reversed_voices = speech.generate(
        ["Hello.", "Welcome."],
        mode=["auto", "clone"],
        voice_clone_prompt=[None, prompt],
        duration=[1.4, 1.0],
        seed=42,
        **stochastic,
    )
    assert np.array_equal(reversed_voices[0], speech.generate("Hello.", duration=1.4, seed=42, **stochastic)[0])
    assert np.array_equal(
        reversed_voices[1],
        speech.generate("Welcome.", voice_clone_prompt=prompt, duration=1.0, seed=43, **stochastic)[0],
    )
    assert np.array_equal(speech.generate("Hello.", duration=1.0, seed=42, **options)[0], first)
    unseeded = synthesize(speech, {}, dict(items=[dict(text="Hello.", duration=1.0)], parameters=stochastic))
    replay = synthesize(
        speech, {}, dict(items=[dict(text="Hello.", duration=1.0)], parameters=stochastic, seed=unseeded["seed"])
    )
    assert replay["audio"] == unseeded["audio"]
    with pytest.raises(ValueError, match="positive"):
        synthesize(speech, {}, dict(items=[dict(text="Hello.", speed=0)]))
    assert np.array_equal(speech.generate("Hello.", duration=1.0, seed=42, **options)[0], first)


def test_oversized_speech_requests_fail_and_leave_the_next_request_unchanged(speech):
    options = dict(
        duration=1,
        num_step=4,
        position_temperature=0,
        class_temperature=0,
        postprocess_output=False,
        pad_duration=0,
        fade_duration=0,
        seed=42,
    )
    expected = speech.generate("Hello.", **options)[0]
    with pytest.raises(ValueError, match="bundle limit"):
        speech.generate("hello " * (speech.capabilities["context_positions"] + 1), **options)
    with pytest.raises(ValueError, match="bundle limit"):
        speech.generate("Hello.", **dict(options, duration=1_000_000, audio_chunk_threshold=2_000_000))
    assert np.array_equal(speech.generate("Hello.", **options)[0], expected)


def test_short_speech_is_intelligible_and_can_be_transcribed_for_cloning(speech):
    import re

    for text, instruction, seed in [
        ("Hello! Welcome to Tinkerbox.", "female, low pitch, british accent", 42),
        ("fuck yeah", None, 7),
        ("Hello", None, 7),
    ]:
        audio = speech.generate(text, instruct=instruction, seed=seed)[0]
        transcript = speech.transcribe((audio, speech.sampling_rate))
        expected = re.sub(r"[^a-z]", "", text.lower())
        assert expected in re.sub(r"[^a-z]", "", transcript.lower())
    with pytest.raises(RuntimeError, match="manually"):
        speech.transcribe((audio, speech.sampling_rate), locale="zz-ZZ")
    prompt = speech.create_voice_clone_prompt((audio, speech.sampling_rate), preprocess_prompt=False)
    # Upstream normalizes reference loudness before ASR, which can change punctuation.
    assert re.sub(r"[^a-z]", "", prompt.ref_text.lower()) == re.sub(r"[^a-z]", "", transcript.lower())
    explicit = speech.generate(
        "Hello.",
        instruct="female, low pitch",
        duration=1,
        num_step=4,
        postprocess_output=False,
        pad_duration=0,
        fade_duration=0,
        seed=42,
    )[0]
    assert len(explicit) == speech.sampling_rate


def test_installed_speech_runs_offline_without_upstream_files(speech, asset, tmp_path):
    """A catalog bundle carries its CPU tables and audio codec, so speech needs no network or upstream checkpoint."""
    import subprocess
    import sys

    options = dict(
        duration=1,
        num_step=4,
        position_temperature=0,
        class_temperature=0,
        postprocess_output=False,
        pad_duration=0,
        fade_duration=0,
        seed=42,
    )
    expected = speech.generate("Hello.", **options)[0]
    output = tmp_path / "speech.npy"
    script = (
        "import sys, numpy, tinkerbox_ane\n"
        "with tinkerbox_ane.load('omnivoice', sys.argv[1]) as model:\n"
        f"    audio = model.generate('Hello.', **{options!r})[0]\n"
        "    assert model.sampling_rate == 24000\n"
        "numpy.save(sys.argv[2], audio)\n"
    )
    # An empty Hugging Face cache with downloads disabled leaves only the bundle to load from.
    environment = {key: value for key, value in os.environ.items() if key != "TINKERBOX_OMNIVOICE_CHECKPOINT"}
    environment.update(HF_HOME=str(tmp_path / "huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([sys.executable, "-c", script, str(asset), str(output)], env=environment, check=True)
    actual = np.load(output)
    assert len(actual) == 24000 and np.array_equal(actual, expected)
