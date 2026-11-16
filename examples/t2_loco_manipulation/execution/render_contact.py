"""Video from recorded physical states, explicitly labeled with admission result."""

import argparse, json
from pathlib import Path
import mujoco, numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("--feet-inset", action="store_true")
p.add_argument("--goal-inset", action="store_true")
p.add_argument(
    "--diagnostic-label",
    help="Explicitly label a partial run without claiming complete-task qualification",
)
p.add_argument("--run", type=Path, required=True)
p.add_argument("--scene", type=Path)
p.add_argument("--camera-azimuth", type=float, default=210.0)
p.add_argument("--camera-lookat", nargs=3, type=float, default=[0.2, 0.1, 0.65])
p.add_argument("--camera-distance", type=float, default=2.3)
p.add_argument("--camera-elevation", type=float, default=-20.0)
a = p.parse_args()
run = a.run.resolve()
s = json.loads(
    (run / ("execution.json" if a.diagnostic_label else "summary.json")).read_text()
)
z = dict(np.load(run / "trace.npz"))
model_path = (run / "model-path.txt").read_text().strip()
m = (
    mujoco.MjModel.from_binary_path(model_path)
    if model_path.endswith(".mjb")
    else mujoco.MjModel.from_xml_path(model_path)
)
d = mujoco.MjData(m)
if a.scene:
    spec = json.loads(a.scene.read_text())
    for name, g in spec["geometries"]["cuboid"].items():
        bid = m.body(name).id
        start = int(m.body_geomadr[bid])
        end = start + int(m.body_geomnum[bid])
        m.geom_rgba[start:end] = np.r_[
            np.array(g.get("color", [150, 150, 150])) / 255.0, 1.0
        ]
m.vis.global_.offwidth = 960
m.vis.global_.offheight = 640
renderer = mujoco.Renderer(m, 640, 960)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = a.camera_lookat
cam.distance = a.camera_distance
cam.azimuth = a.camera_azimuth
cam.elevation = a.camera_elevation
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
except OSError:
    font = ImageFont.load_default()
feet_renderer = (
    mujoco.Renderer(m, 200, 300)
    if (a.feet_inset and "soles" in z) or a.goal_inset
    else None
)
feet_cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(feet_cam)
if feet_renderer:
    feet_cam.lookat[:] = [
        (z["soles"][0, :, 0].mean() + z["soles"][-1, :, 0].mean()) / 2,
        0,
        0.06,
    ]
    feet_cam.distance = 0.9
    feet_cam.azimuth = a.camera_azimuth
    feet_cam.elevation = -25
    if a.goal_inset:
        if not a.scene:
            raise ValueError("Goal inset requires scene region geometry")
        goal = spec["geometries"]["cuboid"]["goal"]
        feet_cam.lookat[:] = [
            goal["pose"][0],
            goal["pose"][1],
            goal["pose"][2] + goal["dims"][2] / 2 + 0.07,
        ]
        feet_cam.distance = 0.65
        feet_cam.elevation = -30
out = run / "physical-rollout.mp4"
writer = imageio.get_writer(out, fps=20, codec="libx264", quality=7, macro_block_size=1)
for t in np.arange(0, z["time"][-1], 0.05):
    i = int(np.argmin(abs(z["time"] - t)))
    d.qpos[:] = z["qpos"][i]
    d.qvel[:] = z["qvel"][i]
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, cam)
    im = Image.new("RGB", (960, 720), (20, 24, 32))
    im.paste(Image.fromarray(renderer.render()), (0, 80))
    draw = ImageDraw.Draw(im)
    draw.text(
        (12, 7),
        f"T2 cuTAMP + contact QP | recorded execution | {run.parent.name} | t={t:.2f}s | 1x playback",
        font=font,
        fill="white",
    )
    draw.text(
        (12, 32),
        (
            f"DIAGNOSTIC: {a.diagnostic_label} | max foot error={s['max_active_sole_error_m']*1000:.1f} mm"
            if a.diagnostic_label
            else f"Measured task checks: {'PASS' if s['qualified'] else 'FAIL'} | max foot tracking error={s['foot_max_drift_m']*1000:.1f} mm (limit10)"
        ),
        font=font,
        fill="white",
    )
    phase = str(z["phases"][i]) if "phases" in z else "fixed-feet manipulation"
    draw.text(
        (12, 56),
        "Actual physics | " + phase + " | free object; no pose resets",
        font=font,
        fill="white",
    )
    if feet_renderer:
        feet_renderer.update_scene(d, feet_cam)
        im.paste(Image.fromarray(feet_renderer.render()), (650, 500))
        draw = ImageDraw.Draw(im)
        delta = (z["soles"][i, :, 0] - z["soles"][0, :, 0]) * 100
        draw.text(
            (655, 503),
            "ACTUAL FLOOR" if a.goal_inset else "ACTUAL FEET",
            font=font,
            fill="white",
        )
        draw.text(
            (655, 680),
            (
                "Free object; physical floor"
                if a.goal_inset
                else f"L {delta[0]:+.1f} | R {delta[1]:+.1f} cm"
            ),
            font=font,
            fill="white",
        )
    writer.append_data(np.asarray(im))
    if abs(t - z["time"][-1] + 0.1) < 0.026:
        im.save(run / "physical-final.png")
writer.close()
renderer.close()
if feet_renderer:
    feet_renderer.close()
print(out)
