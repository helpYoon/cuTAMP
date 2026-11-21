"""Load the frozen T1 step-sequence export used in the particle objective."""

import json
from functools import lru_cache
from pathlib import Path

import torch

from cutamp.landing_accuracy import CHECKPOINT_SHA256, LANDING_MODEL_DIR

DEFAULT_STEP_SEQUENCE_MODEL = LANDING_MODEL_DIR / "step-sequence-v1.pt"
_KNOTS = ["quarter_deadline", "half_deadline", "three_quarters_deadline", "deadline", "mid_hold", "end_hold"]


def load_student(path, device="cuda"):
    extra = {"step_sequence.json": ""}
    model = torch.jit.load(str(path), map_location=device, _extra_files=extra).eval()
    meta = json.loads(extra["step_sequence.json"])
    if (
        meta["schema_version"] != 1
        or meta["hold_ticks"] != 75
        or meta["knots"] != _KNOTS
        or meta["side_order"] != ["L", "R"]
        or meta["padding"] != -1
    ):
        raise ValueError(f"Unsupported T1 step-sequence export at {path}")
    if meta["teacher_sha256"] != CHECKPOINT_SHA256:
        raise ValueError(f"Step-sequence model at {path} was trained against a different teacher")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, meta


@lru_cache(maxsize=4)
def _cached_model(path, version, device):
    return load_student(path, device)


def load_step_sequence_model(path=DEFAULT_STEP_SEQUENCE_MODEL, device="cuda", *, expected_deployment=None):
    """Reuse a loaded export across requests on the same device until the file changes."""
    path = Path(path).resolve()
    stat = path.stat()
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    model, meta = _cached_model(path, (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size), device)
    if expected_deployment is not None and meta["deployment_sha256"] != expected_deployment:
        raise ValueError(f"Step-sequence model at {path} was trained against a different landing deployment")
    return model
