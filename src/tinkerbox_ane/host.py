"""Streaming pipe client for the Core AI host."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

BINARY = Path(__file__).with_name("tinkerbox-ane-host")


def native_command(*arguments, timeout=300):
    """Run a native host command and read its framed JSON reply."""
    result = subprocess.run([str(BINARY), *arguments], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=timeout)
    if len(result.stdout) < 4:
        raise RuntimeError(result.stderr.decode(errors='replace').strip() or 'Native audio command failed.')
    size = struct.unpack('<I', result.stdout[:4])[0]
    reply = json.loads(result.stdout[4:4 + size])
    if result.returncode or 'error' in reply:
        raise RuntimeError(reply.get('error', 'Native audio command failed.'))
    return reply


SUPPORTED_ARCHITECTURES: dict[str, str] = json.loads(
    Path(__file__).with_name("models.json").read_text()
)["architectures"]


def chip_name(architecture: str) -> str:
    """Return the compiler's chip label for an Apple Silicon architecture code."""
    return SUPPORTED_ARCHITECTURES.get(architecture, f"Apple Silicon ({architecture})")


def validate_architecture(architecture: str) -> None:
    """Validate that the architecture is recognized; bundle availability is per model."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            f"This Mac's Neural Engine architecture ({architecture}) is not supported by this version "
            "of tinkerbox-ane. Update tinkerbox-ane to use this Mac."
        )


@lru_cache(maxsize=1)
def device_architecture() -> str:
    result = subprocess.run([str(BINARY), "--architecture"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def check_layout(spec: dict) -> None:
    """The cache tiers, dimensions, and convolution/attention layers the chunked runner allocates state for."""
    width, context, tiers = spec["width"], spec["context"], spec["tiers"]
    if (width <= 0 or context <= width or context % width or not tiers
            or tiers != sorted(set(tiers)) or tiers[-1] != context
            or any(t <= width or t > context or t % width for t in tiers)):
        raise ValueError("invalid fused attention capacities")
    for name in ("hidden", "heads", "kv_heads", "head_dim", "conv_taps"):
        if not isinstance(spec[name], int) or spec[name] <= 0:
            raise ValueError(f"invalid {name}")
    if spec["heads"] % spec["kv_heads"] or spec["conv_taps"] < 2:
        raise ValueError("inconsistent model dimensions")
    for name in ("conv_layers", "attention_layers"):
        layers = spec[name]
        if not layers or layers != sorted(set(layers)) or any(i < 0 for i in layers):
            raise ValueError(f"invalid {name}")
    if set(spec["conv_layers"]) & set(spec["attention_layers"]):
        raise ValueError("overlapping convolution and attention layers")
    counted = [(spec.get("flops"), [f"c{tier}" for tier in tiers])]
    if "vision_asset" in spec:
        counted.append((spec.get("vision_flops"), ["main"]))
    check_flops(counted)


def check_flops(counted) -> None:
    """Each asset's manifest counts the FLOPs of exactly its functions: `counted` pairs them."""
    if any(not isinstance(flops, dict) or set(flops) != set(functions)
           or any(type(n) is not int or n <= 0 for n in flops.values()) for flops, functions in counted):
        raise ValueError("bundle lacks FLOP counts for its functions; re-export the model")


class _PipeHost:
    _timeout = 300.0

    def _read(self, count: int) -> bytes:
        import select
        import time

        deadline = time.monotonic() + self._timeout
        data = bytearray()
        while len(data) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._process.stdout], [], [], remaining)[0]:
                self._process.kill()
                raise RuntimeError(f'Core AI host did not respond within {self._timeout:g} seconds')
            chunk = os.read(self._process.stdout.fileno(), min(count - len(data), 1024 * 1024))
            if not chunk:
                raise RuntimeError(f'the Core AI host exited (status {self._process.poll()})')
            data.extend(chunk)
        return bytes(data)

    def _u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def _reply(self) -> None:
        if self._read(1) != b"\0":
            raise RuntimeError(f"Core AI host: {self._read(self._u32()).decode()}")

    def _send(self, *payloads) -> None:
        try:
            for payload in payloads:
                self._process.stdin.write(payload)
            self._process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(f"the Core AI host exited (status {self._process.poll()})") from error

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.write(b"Q")
                process.stdin.close()
                process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        if process is not None:
            process.wait()
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        self._directory.cleanup()


class TensorHost(_PipeHost):
    """Stateless named FP16 tensor calls through the shared native host."""

    def __init__(self, asset: Path, timeout: float = 300):
        asset = Path(asset).resolve()
        if asset.suffix != '.aimodelc':
            raise ValueError('TensorHost requires an AOT asset')
        self._timeout = timeout
        self._directory = tempfile.TemporaryDirectory(prefix='.host-', dir=asset.parent)
        clones = Path(self._directory.name) / 'clones'
        clones.mkdir()
        environment = dict(os.environ, COW_CLIENT_DEST=str(clones), COW_ASSET_PREFIX=str(asset))
        try:
            self._process = subprocess.Popen([str(BINARY), '--tensor', str(asset)],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment)
            ready = json.loads(self._read(self._u32()))
            if not ready.get('ready'):
                raise RuntimeError(ready.get('error', 'native tensor host failed to load'))
        except BaseException:
            self.close()
            raise

    def run(self, function: str, inputs: dict):
        import numpy as np

        encoded = function.encode()
        payload = [b'F' + struct.pack('<I', len(encoded)) + encoded + struct.pack('<I', len(inputs))]
        for name, value in inputs.items():
            array = np.ascontiguousarray(value, dtype=np.float16)
            key = name.encode()
            payload += [struct.pack('<I', len(key)) + key + struct.pack('<I', array.size),
                        memoryview(array).cast('B')]
        self._send(*payload)
        self._reply()
        rank = self._u32()
        shape = struct.unpack(f'<{rank}I', self._read(4 * rank))
        return np.frombuffer(self._read(2 * int(np.prod(shape))), dtype=np.float16).reshape(shape).copy()


class Host(_PipeHost):
    def __init__(self, asset: str | Path, kind: str, check=check_layout):
        """`check` validates the manifest fields the bundle's runner allocates state for."""
        from tinkerbox_ane.bundle import compiled, read_bundle

        asset = Path(asset).resolve()
        spec = read_bundle(asset, kind)
        check(spec)
        if not compiled(spec):
            raise ValueError("load requires a compiled AOT bundle; compile the exported model first")
        current = device_architecture()
        if spec.get("architecture") and spec["architecture"] != current:
            target_chip = chip_name(spec["architecture"])
            current_chip = chip_name(current)
            raise ValueError(
                f"Bundle targets {target_chip} ({spec['architecture']}), but this Mac is {current_chip} ({current}). Choose a bundle compiled for this Mac."
            )
        # Stage on the asset volume: APFS clones must stay on the same filesystem.
        self._directory = tempfile.TemporaryDirectory(prefix=".host-", dir=asset.parent)
        clones = Path(self._directory.name) / "clones"
        clones.mkdir()
        environment = dict(os.environ, COW_CLIENT_DEST=str(clones),
                           COW_ASSET_PREFIX=str(asset.resolve()))
        try:
            self._process = subprocess.Popen([str(BINARY), str(asset)],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment)
            ready = json.loads(self._read(self._u32()))
            if "error" in ready:
                raise RuntimeError(f"the Core AI host could not load {asset}: {ready['error']}")
            self.hidden_size = ready["hidden"]
            self.width, self.context = ready["width"], ready["context"]
            if ready.get("version") != 2:
                raise RuntimeError("the native host does not support this bundle; rebuild or reinstall tinkerbox-ane")
            self.conv_taps = ready["conv_taps"]
        except BaseException:
            self.close()
            raise

    def vision(self, inputs: dict, function: str = "main"):
        """Run a function of the bundle's vision asset on named FP16 inputs, each row-major in its
        declared shape, and return the `features` output."""
        import numpy as np
        encoded = function.encode()
        payload = [b"V" + struct.pack("<I", len(encoded)) + encoded + struct.pack("<I", len(inputs))]
        for name, value in inputs.items():
            array = np.ascontiguousarray(value, dtype=np.float16)
            encoded = name.encode()
            payload += [struct.pack("<I", len(encoded)) + encoded + struct.pack("<I", array.size),
                        memoryview(array).cast("B")]
        self._send(*payload)
        self._reply()
        rank = self._u32()
        shape = struct.unpack(f"<{rank}I", self._read(4 * rank))
        return np.frombuffer(self._read(2 * int(np.prod(shape))), dtype=np.float16).reshape(shape).copy()

    def metrics(self) -> dict:
        """Neural Engine `seconds` and `calls` per function since the previous call."""
        self._send(b"M")
        return json.loads(self._read(self._u32()))

    def set_rope(self, cos, sin) -> None:
        import numpy as np
        cos, sin = (np.ascontiguousarray(t, dtype=np.float16) for t in (cos, sin))
        if cos.shape != sin.shape or cos.ndim != 2:
            raise ValueError("cos and sin must be matching (head_dim, positions) tables")
        self._send(b"R" + struct.pack("<II", *cos.shape), memoryview(cos).cast("B"), memoryview(sin).cast("B"))
        self._reply()

    def hidden_chunks(self, chunks, tokens: int, *, plan=None):
        import numpy as np
        if not 0 < tokens <= self.context:
            raise ValueError(f"prompt needs {tokens} tokens; context limit is {self.context}")
        if plan is not None and (len(plan.ids) != tokens or plan.width != self.width
                                 or plan.taps != self.conv_taps or plan.context != self.context):
            raise ValueError("tree plan does not match the request or bundle")
        length = -(-tokens // self.width) * self.width
        producer = iter(chunks)
        try:
            if plan is None:
                self._send(b"S" + struct.pack("<I", tokens))
            else:
                self._send(b"T" + struct.pack("<II", tokens, len(plan.leaves)),
                           struct.pack(f"<{len(plan.leaves)}I", *plan.leaves))
            for offset in range(0, length, self.width):
                self._reply()
                chunk = np.ascontiguousarray(next(producer), dtype=np.float16)
                if chunk.shape != (self.hidden_size, self.width):
                    raise ValueError(f"embedding chunk must be ({self.hidden_size}, {self.width})")
                if plan is not None:
                    self._send(plan.metadata(offset), memoryview(chunk).cast("B"))
                else:
                    self._send(memoryview(chunk).cast("B"))
            self._reply()
            count = 1 if plan is None else len(plan.leaves)
            hidden = np.frombuffer(self._read(2 * self.hidden_size * count), dtype=np.float16).copy()
            return hidden if plan is None else hidden.reshape(count, self.hidden_size)
        except BaseException:
            # A failed producer cannot leave the native host waiting for a partial chunk.
            if self._process.poll() is None:
                self._process.kill()
            self.close()
            raise
