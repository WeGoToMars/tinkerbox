"""Speech controls exposed to clients; names and defaults match upstream generation."""

PARAMETERS = [
    dict(key='num_step', title='Diffusion steps', type='int', default=32, min=1, max=128),
    dict(key='guidance_scale', title='Guidance scale', type='float', default=2.0, min=0, max=20),
    dict(key='t_shift', title='Time shift', type='float', default=0.1, min=0.001, max=10),
    dict(key='layer_penalty_factor', title='Codebook layer penalty', type='float', default=5.0, min=0, max=20),
    dict(key='position_temperature', title='Position temperature', type='float', default=5.0, min=0, max=20),
    dict(key='class_temperature', title='Token temperature', type='float', default=0.0, min=0, max=5),
    dict(key='denoise', title='Denoise speech', type='bool', default=True),
    dict(key='preprocess_prompt', title='Preprocess reference audio', type='bool', default=True),
    dict(key='postprocess_output', title='Trim output silence', type='bool', default=True),
    dict(key='audio_chunk_duration', title='Long-form chunk seconds', type='float', default=15.0, min=0.1, max=60),
    dict(key='audio_chunk_threshold', title='Long-form threshold seconds', type='float', default=30.0, min=0.1, max=120),
    dict(key='pad_duration', title='Silence padding seconds', type='float', default=0.1, min=0, max=5),
    dict(key='fade_duration', title='Fade seconds', type='float', default=0.1, min=0, max=5),
    dict(key='normalize_text', title='Normalize numbers and dates', type='bool', default=False),
]


def validate(parameters):
    """Reject misspellings and invalid arithmetic before expensive model work."""
    import math

    definitions = {p['key']: p for p in PARAMETERS}
    for key, value in parameters.items():
        if key not in definitions:
            raise ValueError(f'unknown speech generation parameter: {key}')
        definition = definitions[key]
        if definition['type'] == 'bool':
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be a boolean')
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value)
              or (definition['type'] == 'int' and not isinstance(value, int))
              or value < definition['min']):
            raise ValueError(f'invalid {key}: {value}')
    return dict(parameters)
