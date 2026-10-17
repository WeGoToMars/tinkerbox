"""Chunk metadata for a shared trunk followed by isolated question branches."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TreePlan:
    trunk: list[int]
    branches: list[list[int]]
    width: int
    taps: int
    context: int

    def __post_init__(self):
        if not self.branches or any(not b for b in self.branches):
            raise ValueError("a tree needs nonempty branches")
        self.ids = self.trunk + [token for branch in self.branches for token in branch]
        if not 0 < len(self.ids) <= self.context:
            raise ValueError(f"packed tree needs {len(self.ids)} tokens; context limit is {self.context}")
        self.starts = np.asarray([len(self.trunk)] + [len(b) for b in self.branches[:-1]]).cumsum()
        self.leaves = (self.starts + [len(b) - 1 for b in self.branches]).tolist()

    def metadata(self, offset: int) -> bytes:
        """Routing indices, RoPE positions, and two allowed key intervals per row.

        Convolution input slots are [previous tail | trunk tail | chunk].
        Routing outputs are [one token plane per tap | next tail | next trunk tail].
        The host expands indices to a one-hot FP16 matrix shared by all layers.
        """
        keep, p, width = self.taps - 1, len(self.trunk), self.width

        def ancestor(index, lag):
            if index < p:
                return index - lag
            start = int(self.starts[min(np.searchsorted(self.starts, index, side="right") - 1,
                                        len(self.starts) - 1)])
            return index - lag if index - lag >= start else p + index - start - lag

        def source(index):
            if index >= offset:
                return 2 * keep + index - offset
            if index < p and offset >= p:
                return 2 * keep + index - p
            return keep + index - offset

        indices, positions, ranges = [], [], []
        for index in range(offset, offset + width):
            indices.extend(source(ancestor(index, lag)) for lag in range(keep, -1, -1))
            if index < p:
                positions.append(index)
                ranges.extend((0, index + 1, 0, 0))
            else:
                start = int(self.starts[min(np.searchsorted(self.starts, index, side="right") - 1,
                                            len(self.starts) - 1)])
                positions.append(p + index - start)
                ranges.extend((0, p, start, index + 1))
        last = offset + width - 1
        indices = np.asarray(indices).reshape(width, self.taps).T.reshape(-1).tolist()
        indices.extend(source(ancestor(last, lag)) for lag in range(keep - 1, -1, -1))
        indices.extend(source(p - keep + j) if offset + width >= p else keep + j for j in range(keep))
        assert all(0 <= i < width + 2 * keep for i in indices)
        return np.asarray(indices + positions + ranges, dtype="<u4").tobytes()


def shared_prefix(rows: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    """Keep at least the answer token on each branch, as upstream does."""
    count = 0
    while count < min(map(len, rows)) - 1 and all(r[count] == rows[0][count] for r in rows):
        count += 1
    return rows[0][:count], [r[count:] for r in rows]


def groups(trunk: list[int], branches: list[list[int]], context: int):
    """Bound physical cache use while preserving every complete logical prompt."""
    current, used = [], len(trunk)
    for index, branch in enumerate(branches):
        if not branch or len(trunk) + len(branch) > context:
            raise ValueError(f"prompt needs {len(trunk) + len(branch)} tokens; context limit is {context}")
        if current and used + len(branch) > context:
            yield current
            current, used = [], len(trunk)
        current.append(index)
        used += len(branch)
    if current:
        yield current
