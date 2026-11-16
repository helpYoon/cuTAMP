import json, tempfile, unittest
from pathlib import Path
from admission import sha256, verify


class AdmissionTests(unittest.TestCase):
    def test_exact_artifacts_and_receipts_required(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            model = r / "model"
            ref = r / "reference"
            stage = r / "stage"
            receipt = r / "check.json"
            model.write_bytes(b"model")
            ref.write_bytes(b"reference")
            stage.write_bytes(b"stage")
            receipt.write_text(json.dumps({"passed": True}))
            gate = {
                "passed": True,
                "receipt": {"path": str(receipt), "sha256": sha256(receipt)},
            }
            entry = {
                "kind": "manipulation",
                "reference_path": str(stage),
                "reference_sha256": sha256(stage),
                "model_sha256": sha256(model),
                "planner_generator_version": "v1",
                "scene_path": str(receipt),
                "scene_sha256": sha256(receipt),
                "native_gates": gate,
                "cad": gate,
                "comfort": gate,
            }
            record = {
                "schema_version": "execution_admission_v1",
                "qualified": True,
                "model_sha256": sha256(model),
                "reference_sha256": sha256(ref),
                "scene_sha256": sha256(model),
                "stages": [entry],
            }
            p = r / "admission.json"
            p.write_text(json.dumps(record))
            verify(p, model, ref, model)
            entry["planner_generator_version"] = "UNKNOWN"
            p.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "generator provenance"):
                verify(p, model, ref, model)
            entry["planner_generator_version"] = "v1"
            p.write_text(json.dumps(record))
            ref.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "reference_sha256"):
                verify(p, model, ref, model)
            ref.write_bytes(b"reference")
            receipt.write_text(json.dumps({"passed": False}))
            gate["receipt"]["sha256"] = sha256(receipt)
            entry["scene_sha256"] = sha256(receipt)
            p.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "itself is not passed"):
                verify(p, model, ref, model)

    def test_empty_stage_set_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            m = r / "model"
            q = r / "ref"
            m.write_bytes(b"m")
            q.write_bytes(b"q")
            p = r / "admission.json"
            p.write_text(
                json.dumps(
                    {
                        "schema_version": "execution_admission_v1",
                        "qualified": True,
                        "model_sha256": sha256(m),
                        "reference_sha256": sha256(q),
                        "scene_sha256": sha256(m),
                        "stages": [],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "checked stages"):
                verify(p, m, q, m)


if __name__ == "__main__":
    unittest.main()
