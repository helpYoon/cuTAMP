"""Restore portable archive inputs and issue a new local admission receipt.

Original bytes/receipts remain untouched. Only a new composition spec has its
host-dependent paths rebound. Building an admission does not execute physics.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from admission import sha256


def archive_mapping(archive):
    archive = Path(archive).resolve()
    rows = json.loads((archive / "SHA256_MANIFEST.json").read_text())
    mapping = {}
    for row in rows:
        path = (archive / row["archive_path"]).resolve()
        try:
            path.relative_to(archive)
        except ValueError:
            raise ValueError("Archive path escapes the archive")
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(
                "Missing or changed archive member: " + row["archive_path"]
            )
        mapping[row["original"]] = str(path)
    return mapping


def remap_spec(spec, mapping):
    spec = json.loads(json.dumps(spec))

    def local(path):
        if path not in mapping:
            raise ValueError("Unarchived required input: " + path)
        return mapping[path]

    for key in ["model", "scene"]:
        spec[key] = local(spec[key])
    for stage in spec["stages"]:
        for key in [
            "reference_npz",
            "scene_path",
            "native_gates",
            "kinematic_gates",
            "bridge",
            "cad",
            "comfort",
        ]:
            if key in stage:
                stage[key] = local(stage[key])
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument(
        "--spec",
        default="composition/full-closed-spec.json",
        help="Composition spec path relative to the archive",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    mapping = archive_mapping(a.archive)
    spec = remap_spec(json.loads((a.archive / a.spec).read_text()), mapping)
    a.output.mkdir(parents=True, exist_ok=False)
    restored = a.output / "local-spec.json"
    restored.write_text(json.dumps(spec, indent=2))
    reference, admission = a.output / "reference.npz", a.output / "admission.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_admission.py")),
            "--spec",
            str(restored),
            "--output-reference",
            str(reference),
            "--output-admission",
            str(admission),
        ],
        check=True,
    )
    print(
        json.dumps(
            dict(
                model=spec["model"],
                scene=spec["scene"],
                reference=str(reference),
                admission=str(admission),
                note="Inputs verified and locally admitted; no simulation executed",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
