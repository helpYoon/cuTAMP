"""Plot the fixed-foot T1 maps and their low-pick order scores without a display or network."""

import base64
import html
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

INK = "#20333b"
MUTED = "#52666c"
GRID = "#d9e1df"
PAPER = "#fafbf8"
STYLES = {"standing": ("Standing", "#227293"), "crouching": ("Crouching", "#ad542a")}
TARGET = "#a51e56"
ORDER = {"none": "No step", "L": "L", "R": "R", "LR": "LR", "RL": "RL"}


def _distance(meters, zero="Overlap", blocked=False):
    if meters is None or not np.isfinite(meters):
        return "Foot overlap" if blocked else "Unresolved"
    if meters == 0:
        return zero
    if meters < 0.001:
        return "< 0.1 cm"
    return f"{100 * meters:.1f} cm"


def _xyz(points):
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _box_height(data):
    center, half = data["target_center"][2], data["target_size"][2] / 2
    bottom, top = center - half, center + half
    if abs(bottom) < 1e-9:
        return f"Box on the floor, z = 0–{100 * top:g} cm"
    return f"Box z = {100 * bottom:g}–{100 * top:g} cm"


def _side_views(axes, data):
    center, size = np.asarray(data["target_center"]), np.asarray(data["target_size"])
    sols = np.asarray(data["home_soles"])
    outlines = [center - size / 2, center + size / 2, np.r_[sols[0, :2], 0], np.r_[sols[1, :2], 0]]
    clouds = {}
    for key in STYLES:
        points = _xyz(data[key + "_points"])
        in_band = np.abs(points[:, 1] - center[1]) <= (size[1] + data["voxel_size"]) / 2 + 1e-9
        clouds[key] = (points, points[in_band])
        if len(points):
            outlines.extend([points.min(axis=0), points.max(axis=0)])
        for line in data["skeletons"][key]:
            line = _xyz(line)
            if len(line):
                outlines.extend([line.min(axis=0), line.max(axis=0)])

    bounds = np.asarray(outlines)
    xlim = (bounds[:, 0].min() - .10, bounds[:, 0].max() + .09)
    zlim = (-.035, bounds[:, 2].max() + .08)
    for ax, (key, (title, color)) in zip(axes, STYLES.items()):
        points, in_band = clouds[key]
        ax.set_facecolor(PAPER)
        ax.grid(axis="both", color=GRID, lw=.65, zorder=0)
        ax.axhspan(zlim[0], 0, facecolor="#e8ede9", zorder=0)
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        # Collapse identical projected centers so dense layers do not hide sparse ones.
        for part, shade, alpha, marker_size in ((points, color, .11, 8), (in_band, color, .68, 11)):
            xz = np.unique(np.round(part[:, (0, 2)], 5), axis=0)
            ax.scatter(xz[:, 0], xz[:, 1], s=marker_size, c=shade, marker="s", alpha=alpha,
                       linewidths=0, zorder=2)
        for run in data["skeletons"][key]:
            line = _xyz(run)
            ax.plot(line[:, 0], line[:, 2], "o-", lw=1.8, ms=2.8, color=INK, zorder=4)
        ax.scatter(sols[:, 0], [0, 0], marker="^", color=INK, s=24, zorder=4)
        ax.add_patch(Rectangle((center[0] - size[0] / 2, center[2] - size[2] / 2),
                               size[0], size[2], fc="none", ec=TARGET, lw=2.5, zorder=5))
        hand = data["hand_points"][key]
        if hand is not None:
            ax.scatter([hand[0]], [hand[2]], s=52, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=6)
        minimum_cm = 100 * data["min_hand_z_m"][key]
        ax.set(title=f"{title} · lowest center {minimum_cm:.1f} cm", xlim=xlim, ylim=zlim, aspect="equal",
               xlabel="Forward x (m)", ylabel="Height z (m)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(length=0, labelsize=8, labelcolor=MUTED)
        ax.title.set(color=color, size=12, weight="bold")


def _static_report(data, rows, path):
    fig = Figure(figsize=(10, 9.7), dpi=155, facecolor=PAPER)
    FigureCanvasAgg(fig)
    fig.text(.08, .95, "T1 reach for a low pick", color=INK, size=23, weight="bold")
    center = np.asarray(data["target_center"]) * 100
    desc = (f"{data['arm'].capitalize()} hand   ·   {_box_height(data)}   ·   center ({center[0]:g}, {center[1]:g}, {center[2]:g}) cm"
            f"   ·   {100 * data['voxel_size']:g} cm voxels")
    fig.text(.08, .916, desc, color=MUTED, size=11)
    fig.text(.08, .875, "Fixed feet  /  side view", color=INK, size=13, weight="bold")
    subtitle = ("The target intersects an initial sole. Both panels show the cached workspace."
                if data["hand_points"]["standing"] is None else
                "Magenta box = target. Ring = sampled gripper center. Dark cells share the box’s lateral span.")
    fig.text(.08, .85, subtitle,
             color=MUTED, size=9.6)
    axes = [fig.add_axes([.07, .43, .43, .37]), fig.add_axes([.54, .43, .43, .37])]
    _side_views(axes, data)
    fig.text(.08, .354, "Foot-order ranking", color=INK, size=13, weight="bold")
    fig.text(.08, .328, "3D gaps to the box for the nearest occupied voxel and its saved gripper center.",
             color=MUTED, size=9.5)
    table_ax = fig.add_axes([.075, .078, .85, .229])
    table_ax.axis("off")
    cols = ["Order", "Standing rank", "Standing voxel", "With crouch rank", "With crouch voxel", "Sample gap¹"]
    table = table_ax.table(
        cellText=[[ORDER[r["order"]], r["rank_standing"], _distance(r["standing_m"], blocked=r.get("blocked")),
                   r["rank"], _distance(r["combined_m"], blocked=r.get("blocked")),
                   _distance(r["combined_point_m"], "Inside", r.get("blocked"))] for r in rows],
        colLabels=cols, colWidths=[.12, .17, .18, .19, .20, .14], cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, _), cell in table.get_celld().items():
        cell.visible_edges = "B"
        cell.set_edgecolor(GRID)
        cell.set_linewidth(.75)
        cell.set_text_props(color=MUTED if i == 0 else INK, weight="bold" if i in (0, 1) else "normal")
    fig.text(.08, .037, "¹ With crouching enabled. Rank by voxel overlap, then step count, then voxel gap. Zero grasps sampled.",
             color=MUTED, size=9)
    fig.savefig(path, facecolor=PAPER)


def _payload(data):
    payload = {k: np.asarray(data[k]).tolist() for k in ("target_center", "target_size", "home_soles")}
    payload["skeletons"] = {k: [_xyz(line).round(5).tolist() for line in data["skeletons"][k]] for k in STYLES}
    payload["hand_points"] = {k: np.asarray(data["hand_points"][k]).tolist() for k in STYLES}
    # Use the entire fixed-stance map; compact quantization affects display only.
    payload["points"] = {k: np.unique(_xyz(data[k + "_points"]).round(5), axis=0).tolist() for k in STYLES}
    payload["voxel"] = float(data["voxel_size"])
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def _foot_views(data, rows):
    """World-frame top views with a common scale, initial soles dotted and final soles solid."""
    home = np.asarray(data["home_soles"])
    target = np.asarray(data["target_center"])[:2]
    def marks(feet):
        # Each glyph starts at a sole center and points in its heading.
        feet = np.asarray(feet).reshape(2, 3)
        direction = np.stack([np.cos(feet[:, 2]), np.sin(feet[:, 2])], axis=1)
        return feet[:, :2] - .055 * direction, feet[:, :2] + .075 * direction

    starts = [home] + [r["feet"] for r in rows if r["feet"] is not None]
    points = np.vstack([segment for feet in starts for segment in marks(feet)] + [target])
    low, high = points.min(axis=0), points.max(axis=0)
    # Left is robot-left (+y), upward is forward (+x).
    width, height = 96, 74
    scale = min((height - 12) / max(.2, high[0] - low[0]), (width - 16) / max(.2, high[1] - low[1]))
    mid = (low + high) / 2

    def svg_xy(p):
        return (width / 2 - scale * (p[1] - mid[1]), height / 2 - scale * (p[0] - mid[0]))

    result = []
    for row in rows:
        items = []
        for feet, color, dash, thickness in ((home, "#758488", 'stroke-dasharray="1 3"', 3),
                                             (row["feet"], INK, "", 3.6)):
            if feet is None:
                continue
            back, front = marks(feet)
            for side in range(2):
                x1, y1 = svg_xy(back[side])
                x2, y2 = svg_xy(front[side])
                items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                             f'stroke="{color}" stroke-width="{thickness}" stroke-linecap="round" {dash}/>')
        x, y = svg_xy(target)
        items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{TARGET}"/>')
        result.append(f'<svg viewBox="0 0 {width} {height}" aria-label="Top view of soles and target">'
                      + "".join(items) + "</svg>")
    return result


def _html_report(data, rows, path):
    table = []
    for row, feet in zip(rows, _foot_views(data, rows)):
        table.append(f'<tr><th scope="row">{ORDER[row["order"]]}</th>'
                     f'<td>{row["rank_standing"]}</td><td>{_distance(row["standing_m"], blocked=row.get("blocked"))}</td>'
                     f'<td class="rank">{row["rank"]}</td><td>{_distance(row["combined_m"], blocked=row.get("blocked"))}</td>'
                     f'<td>{_distance(row["combined_point_m"], "Inside", row.get("blocked"))}</td>'
                     f'<td>{feet}</td></tr>')
    center = ", ".join(f"{100 * float(x):g}" for x in data["target_center"])
    source = _HTML.replace("{{TABLE}}", "".join(table)).replace("{{DATA}}", _payload(data))
    mesh = path.with_name("t1-fixed-feet-poses.png")
    panel = ""
    if data["hand_points"]["standing"] is None:
        panel = '<p class="note">The target intersects an initial sole. Both panels show the cached workspace.</p>'
    elif mesh.is_file():
        pixels = base64.b64encode(mesh.read_bytes()).decode("ascii")
        panel = ('<figure class="mesh"><img alt="T1 mesh views of standing and crouched fixed-foot poses" '
                 f'src="data:image/png;base64,{pixels}"></figure>')
    source = source.replace("{{ROBOT_POSES}}", panel)
    source = source.replace("{{ARM}}", html.escape(data["arm"].capitalize()))
    source = source.replace("{{CENTER}}", center).replace("{{VOXEL}}", f"{100 * data['voxel_size']:g}")
    source = source.replace("{{BOX_HEIGHT}}", _box_height(data))
    for key in STYLES:
        source = source.replace("{{MIN_" + key.upper() + "}}", f"{100 * data['min_hand_z_m'][key]:.1f}")
    path.write_text(source, encoding="utf-8")


def plot_report(data: dict, output_dir: Path):
    """Write t1-low-pick.png and t1-low-pick.html, using only measured geometry and scores."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(data["frame_rows"], key=lambda r: r["rank"])
    _static_report(data, rows, out / "t1-low-pick.png")
    _html_report(data, rows, out / "t1-low-pick.html")


_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>T1 reach for a low pick</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#fafbf8;color:#20333b;font:16px/1.5 system-ui,sans-serif}
main{max-width:1110px;margin:auto;padding:clamp(18px,4vw,52px)}h1,h2,h3,p{margin:0}
h1{font-size:clamp(2rem,4.5vw,3.3rem);line-height:1.12;letter-spacing:-.04em}
h2{font-size:1.4rem;letter-spacing:-.02em}h3{font-size:1.04rem}
header>p,.note{color:#52666c}header>p{margin:.65rem 0 2.2rem}.note{font-size:.93rem;margin:.35rem 0}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin:.8rem 0}
button{color:#20333b;font:inherit;border:1px solid #bfcac8;background:transparent;border-radius:5px;padding:4px 12px;cursor:pointer}
button[aria-pressed=true]{background:#20333b;color:white}
.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}
figure{margin:0;border:1px solid #d9e1df;border-radius:8px;background:#fff;overflow:hidden}
.mesh{margin:.75rem 0 1.5rem}.mesh img{display:block;width:100%;height:auto}
figcaption{padding:10px 14px 0;font-size:.98rem;font-weight:650}
.st{color:#227293}.cr{color:#ad542a}canvas{width:100%;height:auto;aspect-ratio:1.12;display:block;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.86rem;color:#52666c;margin:.5rem 0 2rem}
.key{width:.65rem;height:.65rem;display:inline-block;background:#a51e56;margin-right:.3rem}.key.pose{background:#20333b}
.key.hand{background:transparent;border:2px solid #20333b;border-radius:50%}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:3px 10px;border-bottom:1px solid #d9e1df;white-space:nowrap}
thead th{font-size:.85rem;font-weight:580;color:#52666c;padding-top:8px;padding-bottom:8px}
tbody th,.rank{font-weight:700}tbody tr:first-child{background:#ecf2ee}td svg{display:block;width:65px;height:48px}
footer{margin-top:1.2rem}.tech{max-width:80ch}
@media(max-width:620px){.views{grid-template-columns:1fr}table{font-size:.85rem}th,td{padding:3px 6px}}
</style></head><body><main><header><h1>T1 reach for a low pick</h1>
<p>{{ARM}} hand · {{BOX_HEIGHT}} · center ({{CENTER}}) cm · {{VOXEL}} cm voxels</p></header>
<section><h2>Fixed feet</h2>{{ROBOT_POSES}}<p class="note">Drag either view to rotate both. Each dot marks an occupied 3D voxel center.</p>
<div class="controls" aria-label="Camera presets"><button data-view="orbit" aria-pressed="true">3D view</button>
<button data-view="side" aria-pressed="false">Side</button><button data-view="front" aria-pressed="false">Front</button>
<button data-view="top" aria-pressed="false">Top</button></div>
<div class="views"><figure><figcaption class="st">Standing · lowest sampled center {{MIN_STANDING}} cm</figcaption><canvas id="standing" aria-label="Standing hand reach"></canvas></figure>
<figure><figcaption class="cr">Crouching · lowest sampled center {{MIN_CROUCHING}} cm</figcaption><canvas id="crouching" aria-label="Crouched hand reach"></canvas></figure></div>
<div class="legend"><span><i class="key"></i>Pick target</span><span><i class="key pose"></i>Selected pose</span><span><i class="key hand"></i>Its gripper center</span>
<span>Axes in meters · x forward · y left · z up</span></div></section>
<section><h2>Foot-order ranking</h2><p class="note tech">Voxel gap measures the distance from an occupied cell to the box. Sample gap measures the distance from its saved gripper center to the box, with crouching enabled.
Schedules with overlap rank first, followed by fewer steps and smaller gap. Exact ties use a stable order.</p>
<div class="scroll"><table><thead><tr><th>Order</th><th>Standing rank</th><th>Standing voxel gap</th><th>With crouch rank</th>
<th>With crouch voxel gap</th><th>Sample gap</th><th>Soles</th></tr></thead><tbody>{{TABLE}}</tbody></table></div>
<footer class="note tech">Top views show the selected endpoints with crouching enabled. Forward is up. Dotted marks show the initial soles and solid marks show the final soles. The magenta point is the target. Each score measures static reach under the planner’s configured self-collision checks. Grasp orientation, closure and the approach path belong to refinement.</footer></section>
</main><script id="data" type="application/json">{{DATA}}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const colors={standing:'#227293',crouching:'#ad542a'};
let yaw=.47,pitch=.17,drag=null;
const bounds=[[Infinity,Infinity,Infinity],[-Infinity,-Infinity,-Infinity]];
function extend(p){for(let k=0;k<3;k++){bounds[0][k]=Math.min(bounds[0][k],p[k]);bounds[1][k]=Math.max(bounds[1][k],p[k])}}
for(const key of Object.keys(colors)){
  data.points[key].forEach(extend);
  data.skeletons[key].forEach(line=>line.forEach(extend));
}
const target=[];
for(let z of [-1,1])for(let y of [-1,1])for(let x of [-1,1]){
  const p=[x,y,z].map((s,k)=>data.target_center[k]+s*data.target_size[k]/2);target.push(p);extend(p)
}
data.home_soles.forEach(p=>extend([p[0],p[1],0]));
const mid=bounds[0].map((x,k)=>(x+bounds[1][k])/2);
function draw(key){
  const canvas=document.getElementById(key),w=canvas.clientWidth,h=canvas.clientHeight;
  const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  function project(p){
    const [x,y,z]=p.map((n,k)=>n-mid[k]);
    return [cy*x+sy*y,-sp*(-sy*x+cy*y)+cp*z];
  }
  const corners=[];
  for(let a=0;a<2;a++)for(let b=0;b<2;b++)for(let c=0;c<2;c++)corners.push(project([bounds[a][0],bounds[b][1],bounds[c][2]]));
  const span=k=>Math.max(...corners.map(p=>p[k]))-Math.min(...corners.map(p=>p[k]));
  const scale=Math.min((w-42)/Math.max(span(0),.1),(h-33)/Math.max(span(1),.1));
  function screen(p){const [x,y]=project(p);return [w/2+scale*x,h/2-scale*y]}
  function line(points,color,width){
    if(!points.length)return;
    ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();
    points.forEach((p,i)=>{let [x,y]=screen(p);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();
  }
  const [lo,hi]=bounds,step=.2;
  for(let i=Math.ceil(lo[0]/step);i*step<=hi[0];i++)line([[i*step,lo[1],0],[i*step,hi[1],0]],'#d9e1df',.7);
  for(let i=Math.ceil(lo[1]/step);i*step<=hi[1];i++)line([[lo[0],i*step,0],[hi[0],i*step,0]],'#d9e1df',.7);
  ctx.fillStyle=colors[key];ctx.globalAlpha=.30;
  const r=Math.max(.9,Math.min(2.2,scale*data.voxel*.25));
  for(const p of data.points[key]){let [x,y]=screen(p);ctx.fillRect(x-r,y-r,2*r,2*r)}
  ctx.globalAlpha=1;ctx.lineJoin='round';ctx.lineCap='round';
  for(const run of data.skeletons[key]){
    line(run,'#20333b',2.5);
    for(const p of run){let [x,y]=screen(p);ctx.fillStyle='#20333b';ctx.beginPath();ctx.arc(x,y,2.2,0,Math.PI*2);ctx.fill()}
  }
  for(const p of data.home_soles){
    const [x,y,a]=p;line([[-.055,0],[.075,0]].map(([f,l])=>[x+f*Math.cos(a)-l*Math.sin(a),y+f*Math.sin(a)+l*Math.cos(a),0]),'#20333b',3.3)
  }
  for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit))line([target[i],target[i|bit]],'#a51e56',2);
  if(data.hand_points[key]){
    const [hx,hy]=screen(data.hand_points[key]);ctx.beginPath();ctx.arc(hx,hy,5,0,Math.PI*2);
    ctx.strokeStyle='white';ctx.lineWidth=3.5;ctx.stroke();ctx.strokeStyle='#20333b';ctx.lineWidth=1.7;ctx.stroke();
  }
  // Orthographic scale in the plane of the screen, valid for every preset.
  const bar=.2*scale;ctx.strokeStyle='#52666c';ctx.lineWidth=1.3;ctx.beginPath();
  ctx.moveTo(12,h-11);ctx.lineTo(12+bar,h-11);ctx.stroke();
  ctx.font='11px system-ui';ctx.fillStyle='#52666c';ctx.fillText('0.2 m',12,h-16)
  const origin=[w-38,h-35];
  for(const [dx,dy,label] of [[cy,sp*sy,'x'],[sy,-sp*cy,'y'],[0,cp,'z']]){
    if(Math.hypot(dx,dy)<.2)continue;
    ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(origin[0]+19*dx,origin[1]-19*dy);ctx.stroke();
    ctx.fillText(label,origin[0]+24*dx-2,origin[1]-24*dy+3)
  }
}
function redraw(){Object.keys(colors).forEach(draw)}
function preset(name){
  [yaw,pitch]=({orbit:[.47,.17],side:[0,0],front:[Math.PI/2,0],top:[0,Math.PI/2]})[name];
  document.querySelectorAll('button[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===name)));redraw();
}
document.querySelectorAll('button[data-view]').forEach(b=>b.onclick=()=>preset(b.dataset.view));
let queued=false;
function schedule(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;redraw()})}}
for(const canvas of document.querySelectorAll('canvas')){
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];canvas.setPointerCapture(e.pointerId)};
  canvas.onpointermove=e=>{
    if(!drag)return;
    yaw=drag[2]+(e.clientX-drag[0])*.008;
    pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,drag[3]+(e.clientY-drag[1])*.008));
    document.querySelectorAll('button[data-view]').forEach(b=>b.setAttribute('aria-pressed','false'));schedule();
  };
  canvas.onlostpointercapture=()=>{drag=null};
}
new ResizeObserver(schedule).observe(document.querySelector('.views'));redraw();
</script></body></html>"""
