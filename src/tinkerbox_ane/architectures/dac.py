"""DAC acoustic decoder as fixed-width Neural Engine calls."""

import copy
import math

import numpy as np
import torch
from torch import nn


class Snake(nn.Module):
    """x + sin(αx)²/α as x ± (sin(αx)·|α|^-½)²: some α are ~1e-7, whose reciprocal overflows FP16."""

    def __init__(self, source):
        super().__init__()
        alpha = source.alpha.detach().float()
        shifted = alpha + 1e-9  # The original divides by α + 1e-9.
        self.register_buffer('alpha', alpha)
        self.register_buffer('scale', shifted.abs().rsqrt())
        self.register_buffer('sign', shifted.sign())

    def forward(self, x):
        return x + self.sign * (torch.sin(self.alpha * x) * self.scale) ** 2


class Upsample(nn.Module):
    """A kernel = 2 × stride transposed convolution as one kernel-2 convolution over all output
    phases, interleaved. The Neural Engine does not place transposed convolutions."""

    def __init__(self, source):
        super().__init__()
        (stride,), (kernel,), (padding,) = source.stride, source.kernel_size, source.padding
        if kernel != 2 * stride or source.groups != 1 or source.dilation != (1,):
            raise ValueError('expected a kernel = 2 × stride transposed convolution')
        weight = source.weight.detach()  # (in, out, kernel)
        inputs, outputs = weight.shape[:2]
        taps = weight.reshape(inputs, outputs, 2, stride)  # (in, out, tap pair, phase)
        # Phase r of output frame n reads x[n] through tap r and x[n-1] through tap r + stride.
        self.conv = nn.Conv1d(inputs, stride * outputs, 2, padding=1)
        self.conv.weight.data.copy_(torch.stack((taps[:, :, 1], taps[:, :, 0]), -1).permute(2, 1, 0, 3)
                                    .reshape(stride * outputs, inputs, 2))
        self.conv.bias.data.copy_(source.bias.detach().repeat(stride))
        self.stride, self.padding, self.outputs = stride, padding, outputs

    def forward(self, x):
        length = x.shape[-1]
        phases = self.conv(x).reshape(1, self.stride, self.outputs, length + 1)
        interleaved = phases.permute(0, 2, 3, 1).reshape(1, self.outputs, -1)
        return interleaved[..., self.padding:self.padding + self.stride * length]


def _repeat(mask, factor):
    return mask[..., None].expand(*mask.shape, factor).reshape(*mask.shape[:-1], -1)


class Decoder(nn.Module):
    """(1, channels, frames) latents and a (1, 1, frames) validity mask -> (1, 1, hop × frames) audio.

    Masking each convolution's input reproduces the original zero padding at the true
    sequence edges, so a window decodes exactly; only its overlap context is unreliable."""

    def __init__(self, source):
        super().__init__()
        source = copy.deepcopy(source)
        self.conv1, self.conv2, self.activation = source.conv1, source.conv2, source.tanh
        self.blocks = nn.ModuleList()
        for block in source.block:
            units = nn.ModuleList(nn.ModuleList([Snake(unit.snake1), unit.conv1, Snake(unit.snake2), unit.conv2])
                                  for unit in (block.res_unit1, block.res_unit2, block.res_unit3))
            self.blocks.append(nn.ModuleList([Snake(block.snake1), Upsample(block.conv_t1), units]))
        self.snake = Snake(source.snake1)
        self.strides = [block.conv_t1.stride[0] for block in source.block]
        self.hop = math.prod(self.strides)

    def forward(self, hidden, mask):
        x = self.conv1(hidden * mask)
        for (snake, upsample, units), stride in zip(self.blocks, self.strides):
            x = upsample(snake(x) * mask)
            mask = _repeat(mask, stride)
            for snake1, conv1, snake2, conv2 in units:
                x = x + conv2(snake2(conv1(snake1(x) * mask)))
        return self.activation(self.conv2(self.snake(x) * mask))


def decode(run, hidden, width, overlap, hop):
    """Decode (channels, frames) latents with `run(window, mask)` calls of `width` frames.

    Each call keeps `overlap` frames of context on both sides, which must cover the
    decoder's receptive field, and contributes the audio of its central frames."""
    channels, frames = hidden.shape
    core = width - 2 * overlap
    pieces = []
    for start in range(0, frames, core):
        first, last = max(start - overlap, 0), min(start - overlap + width, frames)
        window = np.zeros((1, channels, width), np.float16)
        mask = np.zeros((1, 1, width), np.float16)
        offset = first - (start - overlap)
        window[0, :, offset:offset + last - first] = hidden[:, first:last]
        mask[..., offset:offset + last - first] = 1
        audio = run(window, mask).reshape(-1)
        pieces.append(audio[overlap * hop:(overlap + min(core, frames - start)) * hop])
    return np.concatenate(pieces).astype(np.float32)
