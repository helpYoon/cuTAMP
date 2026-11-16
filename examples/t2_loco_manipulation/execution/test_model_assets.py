import hashlib
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET
from model_assets import resolve_model_assets, verify_asset_inventory


class ModelAssets(unittest.TestCase):
    def test_explicit_asset_directory_preserves_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td)
            (b / "mesh.stl").write_bytes(b"mesh")
            r = ET.fromstring(
                '<mujoco><asset><mesh name="leg" file="/old/mesh.stl" scale="1 2 3"/></asset></mujoco>'
            )
            resolve_model_assets(r, b / "robot.xml", b)
            m = r.find("asset/mesh")
            self.assertEqual(m.get("file"), str((b / "mesh.stl").resolve()))
            self.assertEqual(m.get("scale"), "1 2 3")
            inv = dict(
                assets=[
                    dict(
                        filename="mesh.stl", sha256=hashlib.sha256(b"mesh").hexdigest()
                    )
                ]
            )
            verify_asset_inventory(b, inv)
            (b / "mesh.stl").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "different"):
                verify_asset_inventory(b, inv)


if __name__ == "__main__":
    unittest.main()
