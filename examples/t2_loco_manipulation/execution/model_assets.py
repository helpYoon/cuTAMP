"""Resolve model assets without modifying robot geometry or physics parameters."""

from pathlib import Path
from admission import sha256


def resolve_model_assets(root, robot_xml, asset_root=None):
    base = Path(robot_xml).resolve().parent
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    for element in root.iter():
        filename = element.get("file")
        if filename is None:
            continue
        path = Path(filename)
        if asset_root is not None and element.tag in ["mesh", "texture", "hfield"]:
            path = Path(asset_root).resolve() / path.name
        elif not path.is_absolute():
            directory = (
                meshdir
                if element.tag == "mesh"
                else texturedir if element.tag == "texture" else ""
            )
            path = base / directory / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError("Required model asset: " + str(path))
        element.set("file", str(path))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    return root


def verify_asset_inventory(asset_root, inventory):
    for row in inventory["assets"]:
        path = Path(asset_root) / row["filename"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("Missing or different model asset: " + row["filename"])
