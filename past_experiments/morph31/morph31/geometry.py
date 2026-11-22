"""Arithmetic Stim detector ordering; no detector-to-coordinate ROM.

For d=31, (x/2,y/2,time) uses 5+5+5 bits, so all 29,760 detectors
fit in 15 bits. This direct lattice padding avoids folded-coordinate locality
assumptions. Start/end layers contain only the prepared stabilizer family.
"""

from dataclasses import dataclass

from .circuit import Experiment


@dataclass(frozen=True)
class LatticeCodec:
    distance: int
    rounds: int
    basis: str

    def __post_init__(self):
        Experiment(self.distance, self.rounds, self.basis)

    @property
    def width(self):
        return self.distance.bit_length()

    @property
    def side(self):
        return 1 << self.width

    @property
    def half(self):
        return (self.distance**2 - 1) // 2

    @property
    def num_detectors(self):
        return 2 * self.half * self.rounds

    def coordinate(self, detector):
        if not 0 <= detector < self.num_detectors:
            raise ValueError("detector out of range")
        d, m, h = self.distance, (self.distance - 1) // 2, self.half
        end = h + (self.rounds - 1) * 2 * h
        if detector < h or detector >= end:
            t = 0 if detector < h else self.rounds
            rank = detector if detector < h else detector - end
            if self.basis == "x":
                x, q = divmod(rank, m + 1)
                x += 1
                y = 2 * q + (1 - x % 2)
            else:
                x, q = divmod(rank, m)
                y = 2 * q + (2 if x % 2 == 0 else 1)
        else:
            t, rank = divmod(detector - h, 2 * h)
            t += 1
            if rank < m:
                x, y = 2 * rank + 1, 0
            elif rank >= m + (d - 1) * d:
                x, y = 2 * (rank - m - (d - 1) * d) + 2, d
            else:
                y, x = divmod(rank - m, d)
                y += 1
                x += y % 2
        return x, y, t

    def detector(self, x, y, t):
        d, h, m = self.distance, self.half, (self.distance - 1) // 2
        if not 0 <= t <= self.rounds or not 0 <= x <= d or not 0 <= y <= d:
            raise ValueError("coordinate out of range")
        valid = (0 < x < d and 0 < y < d) or (
            y == 0 and 0 < x < d and x % 2 == 1
        ) or (y == d and 0 < x < d and x % 2 == 0) or (
            x == 0 and 0 < y < d and y % 2 == 0
        ) or (x == d and 0 < y < d and y % 2 == 1)
        if not valid:
            raise ValueError("padded coordinate has no stabilizer")
        if t in (0, self.rounds):
            family = (x + y) % 2
            if family != (self.basis == "x"):
                raise ValueError("stabilizer is absent in initial/final layer")
            rank = (x - 1) * (m + 1) + y // 2 if family else x * m + (y - 1) // 2
            return rank if t == 0 else h + (self.rounds - 1) * 2 * h + rank
        if y == 0:
            rank = (x - 1) // 2
        elif y == d:
            rank = m + (d - 1) * d + (x - 2) // 2
        else:
            rank = m + (y - 1) * d + x - y % 2
        return h + (t - 1) * 2 * h + rank

    def pack(self, x, y, t):
        return x + (y << self.width) + (t << (2 * self.width))

    def unpack(self, address):
        mask, w = self.side - 1, self.width
        return address & mask, (address >> w) & mask, address >> (2 * w)

    def address(self, detector):
        return self.pack(*self.coordinate(detector))

    def validate(self, graph):
        if graph.num_detectors != self.num_detectors:
            raise ValueError("Stim detector count does not match the arithmetic codec")
        for i, c in enumerate(graph.coordinates):
            x, y, t = self.coordinate(i)
            if tuple(c) != (2 * x, 2 * y, t) or self.detector(x, y, t) != i:
                raise ValueError(f"unsupported Stim detector ordering at {i}")
