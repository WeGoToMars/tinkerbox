import math
import subprocess
import sys

import pytest

import tinkerbox_ane
from tinkerbox_ane.d1 import DEFAULT_ASSET, MODEL
from tinkerbox_ane.host import (
    BINARY,
    SUPPORTED_ARCHITECTURES,
    UNSUPPORTED_ARCHITECTURES,
    device_architecture,
)


@pytest.mark.skipif(sys.platform != 'darwin', reason='tinkerbox-ane-host is only built for macOS')
def test_host_binary_architecture_handshake():
    """Verify the native ANE host exists, executes, and returns a recognized architecture."""
    assert BINARY.is_file(), f"native host binary missing: {BINARY}"
    result = subprocess.run([str(BINARY), "--architecture"], capture_output=True, text=True, check=True)
    arch = result.stdout.strip()
    assert arch, "host binary returned empty architecture string"
    assert arch in SUPPORTED_ARCHITECTURES or arch in UNSUPPORTED_ARCHITECTURES, f"unexpected architecture: {arch}"
    assert device_architecture() == arch


GREETING = {'type': 'noul', 'instructions': 'Is this a greeting?'}


@pytest.fixture
def model(monkeypatch):
    if sys.platform != 'darwin' or not (DEFAULT_ASSET / 'model.json').is_file():
        pytest.skip('requires the local Core AI bundle')
    # The bundle carries everything the runtime reads from the upstream checkpoint.
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    with tinkerbox_ane.load(MODEL, DEFAULT_ASSET) as model:
        yield model


def test_public_decisions_batches_and_request_reset(model):
    questions = {
        'greeting': GREETING,
        'topic': {'type': 'choice', 'instructions': 'Choose a topic.',
                  'criteria': {'hello': 'A greeting', 'billing': 'A payment dispute'}},
        'tone': {'type': 'score', 'instructions': 'How friendly is this?',
                 'criteria': ['Unfriendly', 'Neutral', 'Friendly']},
    }
    state = 'Hi'
    result = model.system_one(state, questions)
    assert result['answers']['greeting']['noul'] == pytest.approx(0.97966643, abs=0.01)
    assert result['answers']['topic']['choice'] == 'hello'
    assert 0 <= result['answers']['tone']['score'] <= 2
    usage = result['usage']
    assert (usage['input_tokens'], usage['output_tokens']) == (87, 0)
    # Each request reports its own Neural Engine time and the FLOPs of the functions it ran.
    assert usage['neural_engine_ms'] > 0 and usage['flops'] > 0
    first, empty = model.system_one_batch([(state, questions), (state, {})])
    assert first['answers'] == result['answers']
    assert (first['usage']['input_tokens'], first['usage']['flops']) == (87, usage['flops'])
    assert empty == {'answers': {}, 'usage': {'input_tokens': 0, 'output_tokens': 0, 'neural_engine_ms': 0, 'flops': 0}}
    # Single-question requests packed into one pass split its compute.
    packed = model.system_one_batch([(state, {'greeting': GREETING}), ('Hello there', {'greeting': GREETING})])
    assert all(r['usage']['neural_engine_ms'] > 0 and r['usage']['flops'] > 0 for r in packed)
    expected = model.probabilities(state, [GREETING])
    actual = model.probabilities_batch([(state, [GREETING]), (state, [GREETING])])
    for row in actual:
        assert row[0] == pytest.approx(expected[0], abs=0.01)
    long = model.probabilities('Hello. ' * 150, [GREETING])[0]
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in long)
    assert sum(long) == pytest.approx(1)
    assert model.probabilities(state, [GREETING]) == expected


def test_public_input_limits_and_close(model):
    with pytest.raises(ValueError, match='image'):
        model.probabilities('Hi', [GREETING], images=[object()])
    with pytest.raises(ValueError, match='context limit'):
        model.probabilities('hello ' * model.context, [GREETING])
    assert model.probabilities('Hi', [GREETING])[0][0] > 0.9
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match='closed'):
        model.probabilities('Hi', [GREETING])


def test_parallel_decisions_match_individual_answers(model):
    questions = [GREETING,
                 {'type': 'choice', 'instructions': 'Choose a topic.',
                  'criteria': {'hello': 'A greeting', 'billing': 'A payment dispute'}},
                 {'type': 'score', 'instructions': 'How friendly is this?',
                  'criteria': ['Unfriendly', 'Neutral', 'Friendly']},
                 {'type': 'noul', 'instructions': 'Ignore unrelated notes. ' * 100 + 'Is this a greeting?'}]
    states = ['Hi', 'Archive note: the customer acknowledged the message. ' * 40 + 'Hi, thanks!']
    for state in states:
        individual = [model.probabilities(state, [q])[0] for q in questions]
        parallel = model.probabilities(state, questions)
        reordered = model.probabilities(state, list(reversed(questions)))
        assert len(parallel) == len(reordered) == len(individual)
        for got, reversed_got, expected in zip(parallel, reversed(reordered), individual):
            assert got == pytest.approx(expected, abs=0.01)
            assert reversed_got == pytest.approx(expected, abs=0.01)
        assert model.probabilities(state, [GREETING])[0] == individual[0]

    requests = [('Hi', [GREETING]), ('Goodbye', [GREETING])]
    expected = [model.probabilities(state, qs)[0] for state, qs in requests]
    actual = model.probabilities_batch(requests)
    assert len(actual) == len(expected)
    for got, single in zip(actual, expected):
        assert got[0] == pytest.approx(single, abs=0.01)
    with pytest.raises(ValueError, match='context limit'):
        model.probabilities_batch([('Hi', [GREETING]), ('hello ' * model.context, [GREETING])])
    assert model.probabilities('Hi', [GREETING])[0] == expected[0]


def test_multimodal_decisions_trees_and_batches(model):
    from PIL import Image

    blue_img = Image.new('RGB', (100, 100), color='blue')
    red_img = Image.new('RGB', (100, 100), color='red')
    q_blue = {'type': 'noul', 'instructions': 'Is this image blue?'}
    q_color = {
        'type': 'choice',
        'instructions': 'What color is this image?',
        'criteria': {'blue': 'Blue', 'red': 'Red', 'green': 'Green'},
    }

    # Single-question image decision
    result = model.system_one(None, {'blue': q_blue}, images=[blue_img])
    assert result['answers']['blue']['noul'] > 0.95
    assert result['usage']['input_tokens'] == 87
    assert result['usage']['output_tokens'] == 0

    # Decision tree sharing image representation in trunk
    multi = model.system_one('Color check', {'blue': q_blue, 'color': q_color}, images=[blue_img])
    assert multi['answers']['blue']['noul'] > 0.95
    assert multi['answers']['color']['choice'] == 'blue'
    assert multi['answers']['color']['confidence'] > 0.95
    assert multi['usage']['input_tokens'] == 121

    # Request isolation and answer discrimination across images
    p_blue = model.probabilities(None, [q_blue], images=[blue_img])[0]
    p_red = model.probabilities(None, [q_blue], images=[red_img])[0]
    assert p_blue[0] > 0.95
    assert p_red[0] < 0.05

    # Pictures the processor splits into tiles plus a thumbnail
    tiled = model.system_one(None, {'color': q_color}, images=[Image.new('RGB', (1000, 1000), color='red')])
    assert tiled['answers']['color']['choice'] == 'red'

    # Batched execution mixing image and text-only requests
    requests = [
        (None, {'blue': q_blue}, [blue_img]),
        ('Hi', {'greeting': GREETING}, None),
        (None, {'color': q_color}, [red_img]),
    ]
    batch = model.system_one_batch(requests)
    assert len(batch) == 3
    assert batch[0]['answers']['blue']['noul'] > 0.95
    assert batch[1]['answers']['greeting']['noul'] == pytest.approx(0.9797, abs=0.01)
    assert batch[2]['answers']['color']['choice'] == 'red'


def test_image_tile_limit_and_token_counts(model):
    from PIL import Image

    photo = Image.new('RGB', (4032, 3024), color='red')
    question = {'type': 'noul', 'instructions': 'Is this image red?'}

    def counted(tiles):
        model.max_image_tiles = tiles
        count = model.count_tokens(None, {'red': question}, images=[photo])
        result = model.system_one(None, {'red': question}, images=[photo])
        assert result['answers']['red']['noul'] > 0.95
        assert count['input_tokens'] == result['usage']['input_tokens']
        return count

    assert model.max_image_tiles == 6
    one, four, six = counted(1), counted(4), counted(6)
    assert [one['image_tiles'], four['image_tiles'], six['image_tiles']] == [1, 5, 7]
    # A 4:3 photo fills a 3×2 grid of 512-pixel tiles, plus an overview of the whole picture
    assert six['images'] == [{'source': [4032, 3024], 'cols': 3, 'rows': 2, 'size': [1536, 1024],
                              'overview': [576, 416], 'passes': 7, 'tokens': 1770}]
    assert one['images'][0]['overview'] is None and one['images'][0]['passes'] == 1
    assert one['input_tokens'] < four['input_tokens'] < six['input_tokens']
    for invalid in (0, 11, 2.5, True):
        with pytest.raises(ValueError, match='max_image_tiles'):
            model.max_image_tiles = invalid
    assert model.max_image_tiles == 6

    # A shared pass over text counts its trunk once
    questions = {'greeting': GREETING, 'red': question}
    tree = model.system_one('Hello there', questions)
    assert model.count_tokens('Hello there', questions) == {
        'input_tokens': tree['usage']['input_tokens'], 'image_tiles': 0, 'images': []}


def test_bundle_configuration_cannot_load_external_code(model, tmp_path):
    import json
    from PIL import Image
    from tinkerbox_ane.bundle import clone

    bundle = tmp_path / "bundle"
    clone(DEFAULT_ASSET, bundle)
    marker = tmp_path / "executed.txt"
    external = tmp_path / "external"
    external.mkdir()
    (external / "configuration_probe.py").write_text(
        "from pathlib import Path\n"
        "from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class ProbeConfig(Lfm2VlConfig):\n    pass\n")
    config = bundle / "checkpoint/config.json"
    content = json.loads(config.read_text())
    content.setdefault("auto_map", {})["AutoConfig"] = f"{external}--configuration_probe.ProbeConfig"
    config.write_text(json.dumps(content))
    # Mapping order is not part of a bundle's execution contract.
    manifest = bundle / "model.json"
    manifest.write_text(json.dumps(json.loads(manifest.read_text()), sort_keys=True))
    with tinkerbox_ane.load(MODEL, bundle) as runtime:
        result = runtime.system_one(None, {"blue": {"type": "noul", "instructions": "Is this image blue?"}},
                                    [Image.new("RGB", (64, 64), "blue")])
    assert result["answers"]["blue"]["noul"] > .95
    assert not marker.exists()
