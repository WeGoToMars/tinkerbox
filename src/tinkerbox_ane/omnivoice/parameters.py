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
]

CAPABILITIES = dict(
    instruction_mode='design',
    modes=[dict(id='auto', title='Automatic voice'),
           dict(id='design', title='Voice design', instruction_title='Voice description',
                instruction_required=True, reference='optional',
                instruction_hint='female, low pitch, british accent'),
           dict(id='clone', title='Voice clone', reference='required')],
    controls=[dict(key='duration', title='Target duration (seconds)', type='float',
                   default=None, optional=True, min=0.01,
                   help='Conditions the generated audio length; overrides the speed estimate.'),
              dict(key='speed', title='Speaking speed factor', type='float', default=1.0, min=0.01,
                   help='Native duration estimate: above 1 is faster; below 1 is slower.')])


def validate(parameters):
    """Reject misspellings and invalid arithmetic before expensive model work."""
    from tinkerbox_ane.speech import validate_fields
    return validate_fields(PARAMETERS, parameters)
