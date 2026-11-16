"""Hash-bound admission of composed research references before physical execution.
Receipts describe prior checks; this verifier does not replace those algorithms.
"""

import hashlib, json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, model, reference, scene):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if (
        record.get("schema_version") != "execution_admission_v1"
        or record.get("qualified") is not True
    ):
        raise ValueError("Admission is missing qualified execution_admission_v1")
    for key, artifact in [
        ("model_sha256", model),
        ("reference_sha256", reference),
        ("scene_sha256", scene),
    ]:
        if record.get(key) != sha256(artifact):
            raise ValueError("Admission hash mismatch: " + key)
    stages = record.get("stages", [])
    if not stages:
        raise ValueError("Admission must contain checked stages")
    required = {
        "manipulation": ["native_gates", "cad", "comfort"],
        "step": ["kinematic_gates", "cad", "comfort"],
        "bridge": ["bridge", "cad", "comfort"],
    }
    for stage in stages:
        if stage.get("kind") not in required:
            raise ValueError("Unknown stage kind")
        if stage.get("model_sha256") != record["model_sha256"]:
            raise ValueError("Stage model hash mismatch")
        scene_path = Path(stage.get("scene_path", ""))
        scene_path = (
            scene_path if scene_path.is_absolute() else path.parent / scene_path
        )
        if (
            not stage.get("scene_sha256")
            or not scene_path.is_file()
            or sha256(scene_path) != stage["scene_sha256"]
        ):
            raise ValueError("Stage planner scene hash mismatch")
        if not stage.get("reference_sha256") or str(
            stage.get("planner_generator_version", "")
        ).lower() in ["", "unknown", "none"]:
            raise ValueError("Stage reference or generator provenance missing")
        for gate in required[stage["kind"]]:
            check = stage.get(gate, {})
            if check.get("passed") is not True:
                raise ValueError("Stage gate not passed: " + gate)
            receipt = check.get("receipt", {})
            if not receipt.get("path") or not receipt.get("sha256"):
                raise ValueError("Stage gate requires hashed receipt: " + gate)
            rp = Path(receipt["path"])
            rp = rp if rp.is_absolute() else path.parent / rp
            if sha256(rp) != receipt["sha256"]:
                raise ValueError("Gate receipt hash mismatch: " + gate)
            result = json.loads(rp.read_text())
            accepted = result.get(
                "passed", result.get("qualified", result.get("motion_plan_success"))
            )
            if accepted is not True:
                raise ValueError("Gate receipt itself is not passed: " + gate)
            if (
                result.get("model_sha256")
                and result["model_sha256"] != stage["model_sha256"]
            ):
                raise ValueError("Gate receipt model mismatch: " + gate)
            if (
                result.get("reference_sha256")
                and result["reference_sha256"] != stage["reference_sha256"]
            ):
                raise ValueError("Gate receipt reference mismatch: " + gate)
        rp = Path(stage.get("reference_path", ""))
        rp = rp if rp.is_absolute() else path.parent / rp
        if not rp.is_file() or sha256(rp) != stage["reference_sha256"]:
            raise ValueError("Stage reference hash mismatch")
    return record
