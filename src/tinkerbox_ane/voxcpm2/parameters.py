"""Native VoxCPM2 modes; speech delivery remains descriptive conditioning."""

PARAMETERS = [
    dict(key='num_step', title='Diffusion steps', type='int', default=10, min=1, max=128),
    dict(key='guidance_scale', title='Guidance scale', type='float', default=2.0, min=0, max=20),
    dict(key='min_decode_steps', title='Minimum audio patches', type='int', default=2, min=0, max=512),
    dict(key='max_decode_steps', title='Maximum audio patches', type='int', default=4096, min=4, max=8192),
    dict(key='preprocess_prompt', title='Trim reference silence', type='bool', default=False),
]

_instruction = dict(instruction_title='Voice or delivery description',
                    instruction_hint='A warm, calm voice; speak slowly and clearly.')
CAPABILITIES = dict(
    instruction_mode='design',
    streaming=True,
    modes=[dict(id='auto', title='Automatic voice', language=False),
           dict(id='design', title='Voice design', language=False, reference='optional',
                reference_transcript='none', instruction_required=True, **_instruction),
           dict(id='clone', title='Voice clone', language=False, reference='required',
                reference_transcript='none', **_instruction),
           dict(id='continuation', title='Continue reference audio', language=False, reference='required',
                reference_transcript='required', **_instruction),
           dict(id='clone_continuation', title='Clone and continue reference', language=False,
                reference='required', reference_transcript='required', **_instruction)],
    controls=[])


def validate(parameters):
    from tinkerbox_ane.speech import validate_fields
    return validate_fields(PARAMETERS, parameters)
