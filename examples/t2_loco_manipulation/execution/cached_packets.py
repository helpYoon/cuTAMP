"""Precompute unchanged reference evaluations on the exact controller grid."""

import numpy as np


class CachedPackets:
    def __init__(self, reference, dt=0.002):
        self.duration = reference.duration
        self.dt = dt
        self.ts = np.arange(int(np.ceil(self.duration / dt)) + 1) * dt
        self.packets = [
            tuple(np.array(x, copy=True) for x in reference.packet(t)) for t in self.ts
        ]
        self.feet = [
            tuple(np.array(x, copy=True) for x in reference.feet_packet(t))
            for t in self.ts
        ]

    def _index(self, t):
        if t >= self.duration:
            return len(self.ts) - 1
        x = float(np.clip(t, 0, self.duration)) / self.dt
        i = int(np.rint(x))
        assert (
            abs(x - i) < 1e-5 or t >= self.duration
        ), "Reference requested off controller grid"
        return min(i, len(self.ts) - 1)

    def packet(self, t):
        return tuple(x.copy() for x in self.packets[self._index(t)])

    def feet_packet(self, t):
        return tuple(x.copy() for x in self.feet[self._index(t)])
