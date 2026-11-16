import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from restore_evidence import archive_mapping, remap_spec


class PortableEvidence(unittest.TestCase):
    def test_manifest_rejects_tampering_and_escape(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td)
            (b / "member").write_text("original")
            row = dict(
                original="/old/member",
                archive_path="member",
                sha256=hashlib.sha256(b"original").hexdigest(),
            )
            (b / "SHA256_MANIFEST.json").write_text(json.dumps([row]))
            self.assertEqual(
                archive_mapping(b)["/old/member"], str((b / "member").resolve())
            )
            (b / "member").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                archive_mapping(b)
            row["archive_path"] = "../escape"
            (b / "SHA256_MANIFEST.json").write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "escapes"):
                archive_mapping(b)

    def test_rebind_required_inputs_without_changing_provenance(self):
        s = dict(
            model="/old/model",
            scene="/old/scene",
            stages=[
                dict(
                    reference_npz="/old/ref",
                    cad="/old/receipt",
                    planner_generator_version="v2",
                )
            ],
        )
        m = {
            p: "/new/" + p.split("/")[-1]
            for p in ["/old/model", "/old/scene", "/old/ref", "/old/receipt"]
        }
        result = remap_spec(s, m)
        self.assertEqual(result["stages"][0]["cad"], "/new/receipt")
        self.assertEqual(result["stages"][0]["planner_generator_version"], "v2")
        self.assertEqual(s["model"], "/old/model")
        del m["/old/ref"]
        with self.assertRaisesRegex(ValueError, "Unarchived"):
            remap_spec(s, m)


if __name__ == "__main__":
    unittest.main()
