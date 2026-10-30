"""On-device reference transcription through Apple's SpeechTranscriber."""

from pathlib import Path
import tempfile

from .host import native_command


def transcribe(audio, *, locale=None):
    """Transcribe a file or (waveform, sample_rate); errors require a manual transcript."""
    with tempfile.TemporaryDirectory(prefix='tinkerbox-transcription-') as temporary:
        if isinstance(audio, (str, Path)):
            path = Path(audio)
        else:
            import numpy as np
            import soundfile as sf

            waveform, rate = audio
            if hasattr(waveform, 'detach'):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=0)
            path = Path(temporary) / 'reference.wav'
            sf.write(path, waveform, rate, subtype='FLOAT')
        arguments = ['--transcribe', str(path.resolve())]
        if locale:
            arguments.append(locale)
        return native_command(*arguments)['text']
