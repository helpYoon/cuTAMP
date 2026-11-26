"""The Trunk tilt's cosine (:func:`trunk_tilt_cosine`), read where a posture's tilt is judged:
``particle_initialization.worst_posture`` (the motion stage's row order without centring) and
``scripts/real_legs_report.py``.
"""

from __future__ import annotations

import torch


def trunk_tilt_cosine(q: torch.Tensor, pitch_index: int, roll_index: int) -> torch.Tensor:
    """``q.shape[:-1]``: ``cos(pitch) cos(roll)``, the cosine of the Trunk's tilt from the vertical.

    The base's chain is pitch, then roll, then yaw (YXZ), so the Trunk's z axis keeps ``cos(pitch) cos(roll)``
    of the vertical, whatever the yaw. ``q`` is ``[..., dof]``. Capture-safe."""
    return torch.cos(q[..., pitch_index]) * torch.cos(q[..., roll_index])
