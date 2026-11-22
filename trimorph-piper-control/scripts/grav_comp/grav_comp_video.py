#!/usr/bin/env python3
"""Videos of the gravity_comp stage motions with the TRIMI hand in the workspace-sim layout
(config 2 of the trimorph-piper-control workspace sim), to watch before running them on the arm.

Replays the motions grav_comp_clearance.py checks (poses imported from midas-piper-control's
gravity_comp/poses.py) with hw_session's timing: slews at 0.6 rad/s, the final approach of a
capture at half that, the settle and averaging holds. In the torque stages the probe joint
swings out to the catch radius both ways (on the arm it droops one way under gravity and is
caught there). Each stage plays as fast as lets it last --stage-seconds, between real time
and --speed times faster (the panel shows the factor), so the short probe stages run in real
time and the long capture stages at x4.

Each frame shows an oblique view from the south-east, a side view from the south (east to the
right), a top view (north up) and the live clearances of grav_comp_clearance.Checker (mm).
Parts behind the base's mounting face turn orange (the sim has no model of the structure
there); a part touching the desk, cart, post, floor or mount plate turns red.

    python3 scripts/grav_comp/grav_comp_video.py --viscous            # every stage + all_stages
    python3 scripts/grav_comp/grav_comp_video.py --stages S3 S4b S5   # only these

Rendering uses MuJoCo's default GL context (GLFW, needs a display); headless, set
MUJOCO_GL=osmesa (software, much slower). VS Code plays only the .webm (no H.264 there).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grav_comp import grav_comp_clearance as C  # noqa: E402

# Timing of gravity_comp/hw_session.py.
SLEW_RAD_S = 0.6            # SLEW_STEP_RAD 0.03 per waypoint at SLEW_RATE_HZ 20
APPROACH_RAD_S = 0.3        # capture_static_sample: final approach at half the step
STAGING_SETTLE_S = 0.3
CAPTURE_SETTLE_S = 1.5
CAPTURE_AVG_S = 1.0
SLEW_SETTLE_S = 1.0
SWING_RAD_S = 0.2           # probe swing as drawn (on the arm gravity drives the droop)
SWEEP_RAD_S = (0.10, 0.25)  # measure_friction --viscous: one there-and-back sweep at each speed

COMMANDS = {  # python -m gravity_comp.<...>, from gravity_comp/README.md
    "S1 feedback": "characterize_feedback",
    "S2 gravity": "calibrate --stage gravity",
    "S3 delivery J1": "verify_delivery --joint 1",
    "S3 delivery J2": "verify_delivery --joint 2",
    "S3 delivery J3": "verify_delivery --joint 3",
    "S4a payload": "calibrate --stage payload",
    "S4b wrist": "verify_delivery --wrist",
    "S4c J6": "verify_delivery --j6",
    "S5 friction": "measure_friction --all",
    "S5 friction+viscous": "measure_friction --all --viscous",
    "S6 accept": "drag_mode (acceptance poses, moved by hand)",
}
W, H = 1280, 720
CW, CH = 640, 360           # one view
VIEWS = (  # caption, azimuth, elevation (deg), distance (m), look-at offset from the base (m)
    ("oblique, from the south-east", 135.0, -14.0, 2.0, (0.0, 0.15, 0.0)),
    ("side, from the south (east →)", 90.0, -4.0, 2.1, (0.12, 0.20, 0.10)),
    ("top (north ↑, east →)", 90.0, -89.9, 2.1, (0.15, 0.18, -0.20)),
)
ORANGE = (0.96, 0.50, 0.10, 1.0)
RED = (0.90, 0.05, 0.05, 1.0)
WARN = 0.05                 # m, amber below this


# --------------------------------------------------------------------------- timing

@dataclass
class Segment:
    q0: np.ndarray
    q1: np.ndarray
    seconds: float          # on the arm
    what: str


def segments(it: C.Item) -> list[Segment]:
    """The item's motion as the arm runs it: straight joint-space moves and holds."""
    out: list[Segment] = []

    def move(a, b, speed, what):
        a, b = np.asarray(a, float), np.asarray(b, float)
        out.append(Segment(a, b, float(np.abs(b - a).max()) / speed, what))

    def hold(seconds, what):
        out.append(Segment(out[-1].q1, out[-1].q1, seconds, what))

    p = it.path
    if it.kind == "capture":
        here, below, q, above, _ = p
        for side, start, staging in (("below", here, below), ("above", q, above)):
            move(start, staging, SLEW_RAD_S, f"slew to the staging point {side}")
            hold(STAGING_SETTLE_S, "settle")
            move(staging, q, APPROACH_RAD_S, f"final approach from {side} (half speed)")
            hold(CAPTURE_SETTLE_S, "settle")
            hold(CAPTURE_AVG_S, f"capture from {side} (averaging)")
    elif it.kind == "probe":
        here, q, up, down, _ = p
        joint = f"J{it.probe + 1}"
        move(here, q, SLEW_RAD_S, "slew to the probe pose")
        hold(SLEW_SETTLE_S, "settle")
        move(q, up, SWING_RAD_S, f"{joint} swings +{it.sag:.2f} rad (catch radius)")
        move(up, down, SWING_RAD_S, f"{joint} swings -{it.sag:.2f} rad (catch radius)")
        move(down, q, SWING_RAD_S, "back to the probe pose")
    elif it.kind == "sweep":  # [here, a, b, a, b, a]: slew to a, then one sweep pair per speed
        move(p[0], p[1], SLEW_RAD_S, "slew to the sweep start")
        for k, speed in enumerate(SWEEP_RAD_S):
            hold(0.5, "settle")
            move(p[1 + 2 * k], p[2 + 2 * k], speed, f"constant-speed sweep, {speed:.2f} rad/s")
            move(p[2 + 2 * k], p[3 + 2 * k], speed, f"sweep back, {speed:.2f} rad/s")
    else:
        what = "moved by hand (straight line)" if it.kind == "drag" else "slew"
        for a, b in zip(p, p[1:]):
            move(a, b, SLEW_RAD_S, what)
        hold(SLEW_SETTLE_S, "hold")
    return out


@dataclass
class Frame:
    item: C.Item
    q: np.ndarray
    what: str
    t: float                # on the arm, since the stage started (s)


def timeline(items: list[C.Item], speed: float, fps: float) -> list[Frame]:
    """One frame per 1/fps of video, `speed` times faster than the arm, from the stage's first
    pose to its last."""
    segs = [(it, s) for it in items for s in segments(it)]
    ends = np.cumsum([s.seconds for _, s in segs])
    total = float(ends[-1])
    frames = []
    for k in range(int(math.ceil(total * fps / speed)) + 1):
        t = min(k * speed / fps, total)
        i = min(int(np.searchsorted(ends, t)), len(segs) - 1)
        it, s = segs[i]
        u = 1.0 if s.seconds <= 0 else float(np.clip((t - (ends[i] - s.seconds)) / s.seconds, 0.0, 1.0))
        frames.append(Frame(it, s.q0 + (s.q1 - s.q0) * u, s.what, t))
    return frames


def playback(items: list[C.Item], fastest: float, target: float) -> float:
    """Speed-up for a stage: the largest of 1, 1.5, 2, 3, 4, ... up to `fastest` that still
    leaves `target` seconds of video."""
    seconds = sum(s.seconds for it in items for s in segments(it))
    nice = [v for v in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0) if v <= max(fastest, 1.0)]
    return max([v for v in nice if seconds / v >= target] or [1.0])


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


# --------------------------------------------------------------------------- drawing

def decorate(spec: mujoco.MjSpec) -> None:
    """Lighting, sky, a tiled floor and a larger, orange mounting-face plane (video only)."""
    v = spec.visual
    v.headlight.ambient, v.headlight.diffuse, v.headlight.specular = [0.32] * 3, [0.42] * 3, [0.05] * 3
    v.quality.shadowsize = 4096
    v.map.shadowclip = 3.0          # the default shadow box leaves speckles on the floor in the top view
    T, B = mujoco.mjtTexture, mujoco.mjtBuiltin
    spec.add_texture(name="sky", type=T.mjTEXTURE_SKYBOX, builtin=B.mjBUILTIN_GRADIENT,
                     rgb1=[0.95, 0.96, 0.98], rgb2=[0.60, 0.65, 0.72], width=256, height=1536)
    spec.add_texture(name="tiles", type=T.mjTEXTURE_2D, builtin=B.mjBUILTIN_CHECKER,
                     rgb1=[0.72, 0.73, 0.75], rgb2=[0.64, 0.65, 0.67], width=256, height=256)
    tiles = spec.add_material(name="tiles", texrepeat=[4, 4], texuniform=True)
    tiles.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tiles"
    floor = spec.geom("env_floor")
    floor.material, floor.rgba = "tiles", [1, 1, 1, 1]
    face = spec.geom("mount_face")
    face.size, face.rgba = [0.6, 0.6, 0.01], [0.96, 0.50, 0.10, 0.10]
    spec.worldbody.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0.5, -1.0, 3.5],
                             dir=[-0.25, 0.35, -1.0], diffuse=[0.5] * 3, specular=[0.1] * 3, castshadow=True)


def font(size: int, style: str = ""):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(f"DejaVuSans{style}.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def dashed(draw, p, q, fill, width=2, dash=10, gap=7) -> None:
    p, q = np.asarray(p, float), np.asarray(q, float)
    length = float(np.linalg.norm(q - p))
    u, s = (q - p) / max(length, 1e-9), 0.0
    while s < length:
        e = min(s + dash, length)
        draw.line([tuple(p + u * s), tuple(p + u * e)], fill=fill, width=width)
        s = e + gap


def mm(v: float) -> str:
    return f">{C.DISTMAX * 1000:.0f}" if v >= C.DISTMAX - 1e-9 else f"{v * 1000:.0f}"


class Painter:
    """One video frame: three views of the arm at q and the clearance panel."""

    ROWS = (  # key, label
        ("desk", "desk"),
        ("other", "cart · post · floor · mount plate"),
        ("self", "hand ↔ arm"),
        ("face", "in front of the mounting face"),
        ("person", "config-2 person (standing)"),
        ("guard", "gravity_comp guard (table plane)"),
    )

    def __init__(self, scene: C.Scene, check: C.Checker, guard):
        self.s, self.check, self.guard = scene, check, guard
        self.m, self.d = scene.model, check.d
        self.renderer = mujoco.Renderer(self.m, height=CH, width=CW)
        self.cams = []
        for _, azimuth, elevation, distance, offset in VIEWS:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = scene.base_pos + np.asarray(offset)
            cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
            self.cams.append(cam)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 1                      # the mounting-face plane
        self.rgba = self.m.geom_rgba.copy()
        self.by_name: dict[str, list[int]] = {}
        for g in check.front:
            self.by_name.setdefault(check.parts[g].name, []).append(g)
        self.f_banner, self.f_title = font(28, "-Bold"), font(21, "-Bold")
        self.f_text, self.f_small, self.f_mono = font(16), font(13), font(14, "Mono")
        # Lines on the mounting face for the side view (vertical) and the top view (horizontal).
        pos = scene.base_pos
        along = np.cross([0.0, 0.0, 1.0], scene.base_rot[:, 2])       # horizontal, in the face
        self.face_lines = {1: (np.array([pos[0], pos[1], 0.0]), np.array([pos[0], pos[1], 2.4])),
                           2: (pos - 1.0 * along, pos + 1.4 * along)}

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Pixel coordinates of world points in the view rendered last."""
        cam = self.renderer.scene.camera
        pos = 0.5 * (np.asarray(cam[0].pos) + np.asarray(cam[1].pos))
        fwd, up = np.asarray(cam[0].forward), np.asarray(cam[0].up)
        rel = np.atleast_2d(pts) - pos
        depth = rel @ fwd
        f = 0.5 * CH / math.tan(math.radians(self.m.vis.global_.fovy) / 2)
        return np.column_stack([0.5 * CW + f * (rel @ np.cross(fwd, up)) / depth, 0.5 * CH - f * (rel @ up) / depth])

    def tint(self, c: C.Clear) -> None:
        self.m.geom_rgba[:] = self.rgba
        check = self.check
        fpos, frot = check._frame(self.s.face)
        for g in check.front:
            part = check.parts[g]
            pos, rot = check._frame(g)
            if (pos + rot @ part.centre - fpos) @ frot[:, 2] > part.radius:
                continue
            if check.sdf(self.s.face, check.world(part)).min() < 0:
                self.m.geom_rgba[g] = ORANGE
        for name in C.ENV_NAMES:
            if c.env[name] < 0:
                for g in self.by_name.get(c.env_part[name], []):
                    self.m.geom_rgba[g] = RED

    def paint(self, fr: Frame, stage: str, stage_no: str, pose_no: str, stage_seconds: float, speed: float,
              mins: dict, banner: bool):
        from PIL import Image, ImageDraw
        c = self.check(fr.q)
        mujoco.mj_camlight(self.m, self.d)
        other, other_name = c.other()
        now = dict(desk=c.env["desk"], other=other, self=c.self_, face=c.face, person=c.person,
                   guard=self.guard.clearance(fr.q)[0])
        for k, v in now.items():
            mins[k] = min(mins.get(k, np.inf), v)
        self.tint(c)

        canvas = Image.new("RGB", (W, H), (20, 22, 26))
        for k, (cam, (x0, y0)) in enumerate(zip(self.cams, ((0, 0), (CW, 0), (0, CH)))):
            self.renderer.update_scene(self.d, camera=cam, scene_option=self.opt)
            view = Image.fromarray(self.renderer.render())
            draw = ImageDraw.Draw(view, "RGBA")
            if k in self.face_lines:
                a, b = self.project(np.vstack(self.face_lines[k]))
                dashed(draw, a, b, fill=(235, 110, 20, 255), width=2)
                top = a if a[1] < b[1] else b
                x, y = float(top[0]), float(max(top[1], 34))
                draw.text((x - 8, y + 4), "behind ←", font=self.f_small, fill=(235, 110, 20, 255), anchor="ra")
                draw.text((x + 8, y + 4), "→ front", font=self.f_small, fill=(70, 80, 95, 255), anchor="la")
            caption = VIEWS[k][0]
            box = draw.textbbox((10, 8), caption, font=self.f_small)
            draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(20, 22, 26, 170))
            draw.text((10, 8), caption, font=self.f_small, fill=(235, 238, 242, 255))
            canvas.paste(view, (x0, y0))

        draw = ImageDraw.Draw(canvas, "RGBA")
        x0, y0 = CW + 16, CH + 10
        white, grey, blue = (232, 236, 240), (150, 156, 166), (130, 190, 255)
        draw.text((x0, y0), stage, font=self.f_title, fill=white)
        draw.text((W - 16, y0 + 4), stage_no, font=self.f_small, fill=grey, anchor="ra")
        draw.text((x0, y0 + 30), f"python -m gravity_comp.{COMMANDS.get(stage, '?')}", font=self.f_small, fill=grey)
        draw.text((x0, y0 + 52), f"{pose_no}  {fr.item.label}", font=self.f_text, fill=white)
        draw.text((x0, y0 + 74), fr.what, font=self.f_text, fill=blue)
        draw.text((x0, y0 + 98), f"on the arm {clock(fr.t)} / {clock(stage_seconds)}    video ×{speed:g}",
                  font=self.f_small, fill=grey)
        q = np.degrees(fr.q)
        draw.text((x0, y0 + 118), "q (deg) " + "  ".join(f"J{j + 1}{q[j]:7.1f}" for j in range(6)),
                  font=self.f_mono, fill=white)
        y = y0 + 146
        draw.text((x0, y), "clearance (mm)", font=self.f_small, fill=grey)
        draw.text((x0 + 430, y), "now", font=self.f_small, fill=grey, anchor="ra")
        draw.text((x0 + 530, y), "min so far", font=self.f_small, fill=grey, anchor="ra")
        y += 20
        margin = self.guard.margin_m
        for key, label in self.ROWS:
            def colour(v):
                if key == "face":
                    return (255, 150, 40) if v < 0 else white
                if key == "guard":
                    return (255, 80, 70) if v < margin else white
                return (255, 80, 70) if v < 0 else (255, 190, 60) if v < WARN else white
            v, low = now[key], mins[key]
            note = {"other": other_name, "face": "behind" if v < 0 else "",
                    "person": "inside" if v < 0 else ""}.get(key, "")
            draw.text((x0, y), label, font=self.f_text, fill=white)
            draw.text((x0 + 430, y), mm(v), font=self.f_text, fill=colour(v), anchor="ra")
            draw.text((x0 + 530, y), mm(low), font=self.f_text, fill=colour(low), anchor="ra")
            draw.text((x0 + 542, y + 2), note, font=self.f_small, fill=colour(v))
            y += 22
        draw.text((x0, H - 34), "orange: behind the base's mounting face (the structure there is not in the sim)",
                  font=self.f_small, fill=(255, 150, 40))
        draw.text((x0, H - 18), "red: touching the desk, cart, post, floor or mount plate", font=self.f_small,
                  fill=(255, 80, 70))
        draw.rectangle((CW, H - 4, CW + int(CW * fr.t / max(stage_seconds, 1e-9)), H), fill=(130, 190, 255))
        if banner:
            draw.rectangle((0, 0, W, 54), fill=(10, 12, 16, 215))
            draw.text((W // 2, 27), f"{stage}   ·   python -m gravity_comp.{COMMANDS.get(stage, '?')}",
                      font=self.f_banner, fill=white, anchor="mm")
        return np.asarray(canvas), c

    def close(self) -> None:
        self.renderer.close()


# --------------------------------------------------------------------------- video files

def writer(path: Path, fps: float, webm: bool = False):
    import imageio_ffmpeg
    if webm:
        codec, params = "libvpx-vp9", ["-b:v", "0", "-crf", "34", "-deadline", "realtime", "-cpu-used", "8",
                                       "-row-mt", "1"]
    else:
        codec, params = "libx264", ["-crf", "22", "-preset", "medium", "-movflags", "+faststart"]
    gen = imageio_ffmpeg.write_frames(str(path), (W, H), fps=fps, codec=codec, quality=None,
                                      pix_fmt_out="yuv420p", output_params=params)
    gen.send(None)
    return gen


def intervals(times: list[float], step: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for t in times:
        if out and t - out[-1][1] <= 1.5 * step:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--midas-repo", type=Path, default=Path("~/workspace/midas-piper-control"),
                   help="midas-piper-control checkout with gravity_comp (norman/piper-midas-gravity-comp)")
    p.add_argument("--base-above-desk", type=float, default=0.49, help="base origin above the desk top (m)")
    p.add_argument("--desk-height", type=float, default=C.DESK_HEIGHT, help="desk top above the floor (m)")
    p.add_argument("--start-deg", type=float, nargs=6, default=[0.0] * 6,
                   help="joints the arm starts each stage from (deg)")
    p.add_argument("--hand-turn-deg", type=float, default=0.0,
                   help="extra turn of the hand about the flange axis vs the sim mount (CAD's own: +57.5)")
    p.add_argument("--catch-delivery", type=float, default=None, help="S3/S4 probe swing (rad, default: theirs)")
    p.add_argument("--catch-friction", type=float, default=None, help="S5 probe swing (rad, default: theirs)")
    p.add_argument("--viscous", action="store_true", help="also the measure_friction --viscous sweeps")
    p.add_argument("--stages", nargs="*", help="only stages whose name starts with one of these (e.g. S3 S5)")
    p.add_argument("--speed", type=float, default=4.0, help="fastest playback vs the arm")
    p.add_argument("--stage-seconds", type=float, default=30.0,
                   help="slow a stage down (to real time at most) so that it lasts this long")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out", type=Path, default=C.OUTPUT / "video", help="output folder")
    args = p.parse_args(argv)

    gp, vd, mf, gw, gm = C.load_gravity_comp(args.midas_repo.expanduser())
    catch_d = vd.parse_args(["--j6"]).catch_rad if args.catch_delivery is None else args.catch_delivery
    catch_f = mf.CATCH_RAD if args.catch_friction is None else args.catch_friction
    scene = C.build_scene(args.desk_height, args.base_above_desk, args.hand_turn_deg, decorate=decorate)
    check = C.Checker(scene)
    model = gm.ArmGravityModel(gravity=gm.GRAVITY_RIGHT_MOUNT)
    guard = gw.WorkspaceGuard(model, [gw.table_plane(np.asarray(gm.GRAVITY_RIGHT_MOUNT), args.base_above_desk)])
    stages = C.stage_items(gp, vd, mf, np.radians(args.start_deg), catch_d, catch_f, args.viscous)
    if args.stages:
        stages = {k: v for k, v in stages.items() if any(k.startswith(s) for s in args.stages)}
    if not stages:
        raise SystemExit("no stage matches --stages")

    args.out.mkdir(parents=True, exist_ok=True)
    painter = Painter(scene, check, guard)
    combined = [writer(args.out / "all_stages.mp4", args.fps), writer(args.out / "all_stages.webm", args.fps, True)]
    step, start, rows = 1.0 / args.fps, 0.0, []
    for n, (stage, items) in enumerate(stages.items(), 1):
        speed = playback(items, args.speed, args.stage_seconds)
        frames = timeline(items, speed, args.fps)
        seconds = frames[-1].t
        path = args.out / f"{n:02d}_{stage.replace(' ', '_')}.mp4"
        own, mins, behind = writer(path, args.fps), {}, []
        order = {id(it): i for i, it in enumerate(items, 1)}
        for k, fr in enumerate(frames):
            img, c = painter.paint(fr, stage, f"stage {n}/{len(stages)}", f"pose {order[id(fr.item)]}/{len(items)}",
                                   seconds, speed, mins, banner=k * step < 1.5)
            for w in (own, *combined):
                w.send(img)
            if c.face < 0:
                behind.append((start + k * step, c.face))
        own.close()
        spans = intervals([t for t, _ in behind], step)
        deepest = [min(f for t, f in behind if a <= t <= b) for a, b in spans]
        rows.append((stage, start, len(frames) * step, seconds, speed, mins, list(zip(spans, deepest))))
        start += len(frames) * step
        print(f"  {path.name}: {clock(len(frames) * step)} video (x{speed:g}), {clock(seconds)} on the arm", flush=True)
    for w in combined:
        w.close()
    painter.close()

    print(f"\n{args.out / 'all_stages.mp4'} (and .webm): {clock(start)}")
    print(f"  {'stage':16s} {'starts':>6s} {'length':>6s} {'speed':>5s} {'on arm':>6s} {'desk':>5s} {'other':>5s} "
          f"{'face':>5s} {'person':>6s}  behind the mounting face (all_stages time, deepest mm)")
    for stage, t0, length, seconds, speed, mins, spans in rows:
        where = ", ".join(f"{clock(a)}-{clock(b)} ({-f * 1000:.0f})" for (a, b), f in spans) or "-"
        print(f"  {stage:16s} {clock(t0):>6s} {clock(length):>6s} {'x' + format(speed, 'g'):>5s} {clock(seconds):>6s} "
              f"{mm(mins['desk']):>5s} {mm(mins['other']):>5s} {mm(mins['face']):>5s} {mm(mins['person']):>6s}  {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
