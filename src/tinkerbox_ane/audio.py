"""Read reference audio using Apple's installed codecs."""

from pathlib import Path
import tempfile

from .host import native_command


def read_audio(path):
    """Decode a file into a (channels × samples, sample_rate) tuple."""
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix='tinkerbox-audio-') as temporary:
        wav = Path(temporary) / 'reference.wav'
        native_command('--convert-audio', str(Path(path).resolve()), str(wav), timeout=60)
        waveform, rate = sf.read(wav, dtype='float32', always_2d=True)
    return waveform.T, rate
