"""Run independent exact-reference admission audits with bounded CPU concurrency."""

import argparse, json, os, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--output", required=True)
p.add_argument("--workers", type=int, default=3)
a = p.parse_args()
spec = json.loads(Path(a.manifest).read_text())
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parent
jobs = []
for row in spec["references"]:
    label = row["label"]
    if not label or Path(label).name != label:
        raise ValueError("Reference label must be a filename component")
    for gate, script in [
        ("cad", "audit_reference_cad.py"),
        ("comfort", "audit_reference_comfort.py"),
        ("kinematics", "audit_reference_kinematics.py"),
    ]:
        receipt = out / (label + "-" + gate + ".json")
        cmd = [
            sys.executable,
            str(source / script),
            "--model",
            spec["model"],
            "--reference",
            row["reference_npz"],
            "--output",
            str(receipt),
        ]
        if gate == "cad" and spec.get("known_payload"):
            cmd += ["--known-payload", spec["known_payload"]]
        if gate == "kinematics":
            cmd += ["--audit", spec["audit"]]
        jobs.append((label, gate, receipt, cmd))


def run(job):
    label, gate, receipt, cmd = job
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with receipt.with_suffix(".log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    data = (
        json.loads(receipt.read_text())
        if receipt.is_file() and result.returncode == 0
        else {}
    )
    return dict(
        label=label,
        gate=gate,
        passed=data.get("passed") is True,
        exit_code=result.returncode,
        receipt=str(receipt),
    )


with ThreadPoolExecutor(max_workers=a.workers) as pool:
    results = list(pool.map(run, jobs))
report = {
    "passed": all(row["passed"] for row in results),
    "references": len(spec["references"]),
    "results": results,
}
(out / "batch-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if report["passed"] else 1)
