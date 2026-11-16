"""Single-object hold lifecycle; release indices are transitions, not open frames."""

import numpy as np


def hold_lifecycle(time, gripper, relative, settle_s=0.7):
    time = np.asarray(time)
    closed = np.asarray(gripper) > 128
    relative = np.asarray(relative)
    previous = np.r_[False, closed[:-1]]
    closes = np.flatnonzero(closed & ~previous)
    opens = np.flatnonzero(~closed & previous)
    carry = (
        (time >= time[closes[0]] + settle_s) & (time < time[opens[-1]])
        if len(closes) and len(opens)
        else np.zeros(len(time), bool)
    )
    drift = (
        float(np.linalg.norm(relative[carry] - relative[carry][:1], axis=-1).max())
        if carry.any()
        else None
    )
    return closes, opens, carry, drift
