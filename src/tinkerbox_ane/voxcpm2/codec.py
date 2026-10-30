"""CPU AudioVAE V2 from OpenBMB/VoxCPM (Apache-2.0).

Pinned upstream cce58a5b59303c9bd63d12f23afe6a49a4a80c59, reduced to the configuration VoxCPM2
ships (depthwise, sample-rate scale/bias conditioning, no noise blocks). Encoding, offline decoding
and the stateful causal decoder match that source; module names match its checkpoint.
See LICENSES/VoxCPM-Apache-2.0.txt and NOTICE.
"""

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def conv1d(module, x):
    """Unpadded convolution; depthwise layers sum their taps, which avoids PyTorch's
    per-channel CPU loop for grouped convolutions."""
    channels, taps = module.in_channels, module.kernel_size[0]
    if not (module.groups == channels == module.out_channels and module.stride == (1,) and module.padding == (0,)):
        return nn.Conv1d.forward(module, x)
    dilation, length = module.dilation[0], x.shape[-1] - module.dilation[0] * (taps - 1)
    weight = module.weight.reshape(1, channels, taps)
    y = x[..., :length] * weight[..., :1]
    for tap in range(1, taps):
        y = y + x[..., tap * dilation: tap * dilation + length] * weight[..., tap:tap + 1]
    return y if module.bias is None else y + module.bias.reshape(1, channels, 1)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = padding * 2 - output_padding

    def forward(self, x):
        return conv1d(self, F.pad(x, (self.causal_padding, 0)))


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_trim = padding * 2 - output_padding

    def forward(self, x):
        return super().forward(x)[..., :-self.causal_trim]


def WNCausalConv1d(*args, **kwargs):
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


def fold_weight_norm(codec):
    """Replace each weight-normalized layer's weight with its computed value, once."""
    for module in codec.modules():
        if hasattr(module, 'weight_g'):
            torch.nn.utils.remove_weight_norm(module)
    return codec


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class CausalResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int, groups: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation, groups=groups),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


def residual_units(dim: int, groups: int):
    return [CausalResidualUnit(dim, dilation, groups) for dilation in (1, 3, 9)]


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int, stride: int):
        super().__init__()
        input_dim = output_dim // 2
        self.block = nn.Sequential(
            *residual_units(input_dim, input_dim),
            Snake1d(input_dim),
            WNCausalConv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride,
                           padding=math.ceil(stride / 2), output_padding=stride % 2),
        )

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, strides: list):
        super().__init__()
        blocks = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]
        # Each block doubles the channels as it downsamples by its stride.
        for stride in strides:
            d_model *= 2
            blocks.append(CausalEncoderBlock(d_model, stride))
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        # Unused at inference; kept so the checkpoint loads strictly.
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        # Registered last, like upstream, so the state dict keeps its key order.
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.fc_mu(self.block(x))


class CausalDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNCausalTransposeConv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride,
                                    padding=math.ceil(stride / 2), output_padding=stride % 2),
            *residual_units(output_dim, output_dim),
        )
        self.input_channels = input_dim

    def forward(self, x):
        return self.block(x)


class SampleRateConditionLayer(nn.Module):
    """Scale and shift a block's input by the output sample rate's bucket."""

    def __init__(self, input_dim: int, buckets: int):
        super().__init__()
        self.scale_embed = nn.Embedding(buckets, input_dim)
        self.bias_embed = nn.Embedding(buckets, input_dim)

    def forward(self, x, bucket):
        return x * self.scale_embed(bucket).unsqueeze(-1) + self.bias_embed(bucket).unsqueeze(-1)


class CausalDecoder(nn.Module):
    def __init__(self, input_channel: int, channels: int, rates: list, sr_bin_boundaries: list):
        super().__init__()
        layers = [
            WNCausalConv1d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
            WNCausalConv1d(input_channel, channels, kernel_size=1),
        ]
        for i, stride in enumerate(rates):
            layers.append(CausalDecoderBlock(channels // 2**i, channels // 2 ** (i + 1), stride))
        output_dim = channels // 2 ** len(rates)
        layers += [Snake1d(output_dim), WNCausalConv1d(output_dim, 1, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.ModuleList(layers)
        self.register_buffer('sr_bin_boundaries', torch.tensor(sr_bin_boundaries, dtype=torch.int32))
        self.sr_cond_model = nn.ModuleList(
            SampleRateConditionLayer(layer.input_channels, len(sr_bin_boundaries) + 1)
            if isinstance(layer, CausalDecoderBlock) else None for layer in layers)

    def forward(self, x, sample_rate):
        bucket = torch.bucketize(sample_rate, self.sr_bin_boundaries)
        for layer, condition in zip(self.model, self.sr_cond_model):
            if condition is not None:
                x = condition(x, bucket)
            x = layer(x)
        return x


@dataclass(frozen=True)
class AudioVAEConfig:
    encoder_dim: int = 128
    encoder_rates: list = field(default_factory=lambda: [2, 5, 8, 8])
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: list = field(default_factory=lambda: [8, 6, 5, 2, 2, 2])
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    sr_bin_boundaries: list = field(default_factory=lambda: [20000, 30000, 40000])


class AudioVAE(nn.Module):
    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.encoder = CausalEncoder(config.encoder_dim, config.latent_dim, config.encoder_rates)
        self.decoder = CausalDecoder(config.latent_dim, config.decoder_dim, config.decoder_rates,
                                     config.sr_bin_boundaries)
        self.sample_rate = config.sample_rate
        self.out_sample_rate = config.out_sample_rate
        self.chunk_size = math.prod(config.encoder_rates)
        self.decode_chunk_size = math.prod(config.decoder_rates)

    def decode(self, z: torch.Tensor):
        """(batch, latent, frames) latents -> (batch, 1, samples) audio at the output sample rate."""
        return self.decoder(z, torch.tensor([self.out_sample_rate], dtype=torch.int32))

    def streaming_decode(self):
        """A context manager whose ``decode_chunk`` decodes consecutive latent chunks, carrying causal
        convolution state between calls instead of decoding overlap."""
        return StreamingVAEDecoder(self)

    def encode(self, audio_data: torch.Tensor):
        """(batch, samples) or (batch, 1, samples) audio at `sample_rate` -> (batch, latent, frames)."""
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        return self.encoder(F.pad(audio_data, (0, -audio_data.shape[-1] % self.chunk_size)))


def trim_reference(
    audio: torch.Tensor, sample_rate: int, max_silence_ms: float = 200.0, top_db: float = 35.0
) -> torch.Tensor:
    import librosa

    if audio.numel() == 0:
        return audio
    y = audio.squeeze(0).numpy()
    n = len(y)
    frame_length = 2048
    hop_length = 512
    ref = np.max(np.abs(y))
    if ref <= 0:
        return audio
    threshold = ref * (10.0 ** (-top_db / 20.0))

    try:
        _, (start, end) = librosa.effects.trim(
            y, top_db=top_db, ref=np.max, frame_length=frame_length, hop_length=hop_length
        )
    except Exception:
        start, end = 0, n

    # Find the last frame with continuous energy, trim the long pseudo-silence at the end (low energy background noise, etc.)
    n_frames = max(0, (n - frame_length) // hop_length + 1)
    last_voice_frame = -1
    for j in range(n_frames):
        idx = j * hop_length
        if idx + frame_length > n:
            break
        rms = np.sqrt(np.mean(y[idx : idx + frame_length] ** 2))
        if rms >= threshold:
            last_voice_frame = j
    if last_voice_frame >= 0:
        end_by_vad = min(n, (last_voice_frame + 1) * hop_length + (frame_length - hop_length))
        end = min(end, end_by_vad)

    max_silence_samples = int(max_silence_ms * sample_rate / 1000.0)
    new_start = max(0, start - max_silence_samples)
    new_end = min(n, end + max_silence_samples)
    return audio[:, new_start:new_end]


class StreamingVAEDecoder:
    """Stateful streaming wrapper for :class:`AudioVAE`.

    Carries causal-convolution padding buffers between calls so that each
    ``decode_chunk`` processes only the new latent patch — no overlap needed.
    """

    def __init__(self, vae: AudioVAE):
        self._vae = vae
        self._states: dict = {}
        self._originals: list = []

    def __enter__(self):
        self._states.clear()
        for mod in self._vae.decoder.modules():
            if isinstance(mod, CausalConv1d) and mod.causal_padding > 0:
                self._patch_causal_conv(mod, mod.causal_padding)
            elif isinstance(mod, CausalTransposeConv1d):
                context = (mod.kernel_size[0] - 1) // mod.stride[0]
                if context > 0:
                    self._patch_transpose_conv(mod, context, mod.causal_trim)
        return self

    def __exit__(self, *exc):
        for mod, orig in self._originals:
            mod.forward = orig
        self._originals.clear()
        self._states.clear()

    def decode_chunk(self, z_chunk: torch.Tensor) -> torch.Tensor:
        """Decode a single latent chunk and return the audio waveform."""
        return self._vae.decode(z_chunk)

    def _patch_causal_conv(self, mod, pad_size):
        states = self._states
        key = id(mod)
        orig = mod.forward

        def fwd(x, _k=key, _p=pad_size, _m=mod):
            x_pad = torch.cat([states[_k], x], dim=-1) if _k in states else F.pad(x, (_p, 0))
            if x.shape[-1] >= _p:
                states[_k] = x[:, :, -_p:].detach()
            else:
                prev = states.get(_k, torch.zeros(x.shape[0], x.shape[1], _p, device=x.device, dtype=x.dtype))
                states[_k] = torch.cat([prev, x], dim=-1)[:, :, -_p:].detach()
            return conv1d(_m, x_pad)

        mod.forward = fwd
        self._originals.append((mod, orig))

    def _patch_transpose_conv(self, mod, ctx, trim):
        states = self._states
        key = id(mod)
        orig = mod.forward

        def fwd(x, _k=key, _c=ctx, _t=trim, _m=mod):
            x_full = torch.cat([states[_k], x], dim=-1) if _k in states else F.pad(x, (_c, 0))
            states[_k] = x[:, :, -_c:].detach()
            out = nn.ConvTranspose1d.forward(_m, x_full)
            left = _c * _m.stride[0]
            return out[..., left:-_t] if _t > 0 else out[..., left:]

        mod.forward = fwd
        self._originals.append((mod, orig))
