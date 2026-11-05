"""The envelope a policy is trained on, which bounds what the planner may send it (design 7.10, call 22).

One YAML file per trained policy, pinned under ``assets/envelopes/``. The planner enforces the joint ranges, and the
Trunk height floor and tilt maximum when the file sets them (``cutamp/robots/t1.py`` cuts the limits; the Motion value
``trunk_tilt``, GN's tilt row and ``TrunkTiltCost`` bound the tilt). The caps, payload, stance range and hold lengths
are stored for later stages.
"""

import hashlib
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import yaml

ENVELOPES_DIR = Path(__file__).parent / "assets" / "envelopes"


class Envelope(NamedTuple):
    name: str
    sha1: str                                     # of the file's bytes
    trunk_height_min: Optional[float]             # m, Trunk origin above the soles; None: no floor
    trunk_tilt_max: Optional[float]               # rad; None: no tilt bound
    joint_ranges: Dict[str, Tuple[float, float]]  # cuts inside the soft limits; unlisted joints keep them
    speed_caps: Optional[Dict[str, float]]        # unset (None) in route_a_start
    acceleration_caps: Optional[Dict[str, float]]
    trunk_speed_caps: Optional[Dict[str, float]]
    payload_kg: Dict[str, object]                 # stored, not enforced (objects have no mass)
    stance_range: Dict[str, float]                # stored; walks are stage 5
    hold_s: Dict[str, float]                      # stored; export is stage 6


def load_envelope(name: str) -> Envelope:
    """The envelope pinned as ``assets/envelopes/<name>.yml``. Raises unless the file names itself ``name`` and holds
    exactly the Envelope's fields."""
    path = ENVELOPES_DIR / f"{name}.yml"
    raw = path.read_bytes()
    fields = yaml.safe_load(raw)
    if fields.get("name") != name:
        raise ValueError(f"{path} names itself {fields.get('name')!r}, not {name!r}")
    fields["joint_ranges"] = {joint: (float(lo), float(hi)) for joint, (lo, hi) in fields["joint_ranges"].items()}
    for key in ("trunk_height_min", "trunk_tilt_max"):  # null in the file (None): no such bound
        if fields.get(key) is not None:
            fields[key] = float(fields[key])
    return Envelope(sha1=hashlib.sha1(raw).hexdigest(), **fields)
