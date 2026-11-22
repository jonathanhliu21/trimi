#!/usr/bin/env python3
"""Clearance of the gravity_comp calibration motions with the TRIMI hand, in the
rig layout (config 2 of the trimorph-piper-control workspace sim), before they run on the arm.

gravity_comp (midas-piper-control, branch norman/piper-midas-gravity-comp) drives
the arm through fixed joint poses in every stage, imported here from its
gravity_comp/poses.py so the check follows that branch. Its own guard
(gravity_comp/workspace.py) watches the link3-6 origins and one point 22 cm down
the flange axis against a table plane. This replays each stage's motion on the
full arm meshes and the TRIMI hand (assets/piper/trimi_hand.npz: hand, Quest
controller, camera, flange adapter) and reports for every pose, at the pose and
over the motion around it (mm):

  desk     distance to the desk (solid down to the floor);
  other    the cart, the post, the floor and the mount plate behind the base;
  hand-arm hand to base and links 1-3;
  face     how far links 2-6 and the hand stay in front of the base's mounting
           face (negative = behind it, where the real mount structure is; the
           sim does not model that structure);
  person   the config-2 wearer (standing, arms down), for information;
  guard    the clearance gravity_comp's own guard computes (table plane at
           --base-above-desk below the base, its default 0.22 m hand reach); a
           stage refuses to start, or stops mid-slew, below its 50 mm margin.

Motions are joint-space straight lines, as hw_session.slew_to drives them:
  S1, S2, S4a  start -> per pose: below, target, above, target -> start
  S3, S4b, S4c start -> each probe pose, probe joint swept +-catch radius
  S5           start -> each friction pose, probe joint swept +-catch radius; with --viscous
               each joint's two constant-speed sweeps follow its friction poses
  S6           acceptance poses (dragged by hand; straight lines between them)
The start pose is wherever the arm rests when a stage script starts (--start-deg).

    python3 scripts/grav_comp/grav_comp_clearance.py --midas-repo ~/workspace/midas-piper-control
    python3 scripts/grav_comp/grav_comp_clearance.py --stages S3 S5 --render   # pictures of flagged poses

The poses come from a midas-piper-control checkout (--midas-repo), the Piper and TRIMI hand
models from the trimorph-piper-control workspace sim (--workspace-repo).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Config 2, the workspace sim's chosen layout (trimorph-piper-control web viewer defaults());
# the base height comes from --base-above-desk (measured on the rig: 0.49 m).
DESK_CENTRE = (0.271, 0.43)
DESK_SIZE = (0.50, 0.976)            # along x, along y (m)
DESK_HEIGHT = 0.75                   # desk top above the floor (m)
BASE_XY = (0.0, 0.0)
BASE_RPY_DEG = (90.0, 0.0, 90.0)     # J1 axis east over the desk, base +y up (gravity = base -y)
PERSON = dict(x=-0.255, y=0.225, facing_deg=51.0, height=1.7)
ENV_NAMES = ("desk", "cart", "post", "floor", "plate")
DISTMAX = 0.30                       # distances are reported up to this (m)
OUTPUT = ROOT / "output" / "grav_comp_clearance"

ARM_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6")
# The Piper URDF and meshes, the TRIMI hand hulls and the old cart and post come from the
# trimorph-piper-control workspace sim (scripts/workspace/analysis.py, assets/piper): the first
# of these checkouts that has it, unless --workspace-repo names one.
WORKSPACE_REPOS = (ROOT, Path("~/trimorph-piper-control").expanduser())
_WORKSPACE_MODULE = "piper_workspace_analysis"


def workspace(repo: Path | None = None):
    """The workspace sim's analysis module, loaded from its file (this repo and
    trimorph-piper-control both have a top-level `scripts` package)."""
    repos = [Path(repo).expanduser()] if repo else list(WORKSPACE_REPOS)
    path = next((r / "scripts" / "workspace" / "analysis.py" for r in repos
                 if (r / "scripts" / "workspace" / "analysis.py").exists()), None)
    if path is None:
        raise FileNotFoundError("no workspace sim (scripts/workspace/analysis.py) in "
                                + ", ".join(map(str, repos)) + "; pass --workspace-repo")
    loaded = sys.modules.get(_WORKSPACE_MODULE)
    if loaded is not None and Path(loaded.__file__) == path:
        return loaded
    spec = importlib.util.spec_from_file_location(_WORKSPACE_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_WORKSPACE_MODULE] = module  # its dataclasses look the module up there
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- scene

@dataclass
class Scene:
    model: mujoco.MjModel
    qadr: np.ndarray
    env: dict[str, int]                 # obstacle -> geom
    person: list[int]
    face: int                           # plane on the base's mounting face, normal along the J1 axis
    links: dict[str, list[int]]         # link1..link6 -> mesh geoms
    base: list[int]
    hand: list[int]
    base_pos: np.ndarray
    base_rot: np.ndarray


def build_scene(desk_height: float, base_above_desk: float, hand_turn_deg: float = 0.0,
                decorate: Callable[[mujoco.MjSpec], None] | None = None,
                workspace_repo: Path | None = None) -> Scene:
    """Config-2 scene; `decorate` may restyle the spec (lights, textures) before it compiles."""
    A = workspace(workspace_repo)
    rotation = Rotation.from_euler("xyz", BASE_RPY_DEG, degrees=True).as_matrix()
    position = np.array([*BASE_XY, desk_height + base_above_desk])
    spec = mujoco.MjSpec.from_string(A._urdf_text(rotation, position))
    spec.meshdir = str(A.ASSET_DIR / "meshes")
    spec.compiler.discardvisual = False  # keep the contype=0 obstacle geoms
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
    for geom in spec.geoms:
        geom.rgba = [0.80, 0.81, 0.84, 1.0]
    g = mujoco.mjtGeom
    world = spec.worldbody

    def fixed(name, kind, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=(0.6, 0.6, 0.6, 1), **kw):
        world.add_geom(name=name, type=kind, size=list(size), pos=list(pos), quat=list(quat),
                       rgba=list(rgba), contype=0, conaffinity=0, **kw)

    (cx, cy), (width, depth) = DESK_CENTRE, DESK_SIZE
    fixed("env_desk", g.mjGEOM_BOX, [width / 2, depth / 2, desk_height / 2], [cx, cy, desk_height / 2],
          rgba=(0.86, 0.70, 0.42, 1))
    fixed("env_floor", g.mjGEOM_PLANE, [3, 3, 0.1], rgba=(0.62, 0.62, 0.64, 1))
    for name, (bx, by, bz, hx, hy, hz) in zip(("cart", "post"), A.CART_BOXES):
        fixed(f"env_{name}", g.mjGEOM_BOX, [hx, hy, hz], [bx, by, bz], rgba=(0.45, 0.46, 0.50, 0.7))
    fixed("env_plate", g.mjGEOM_BOX, [0.05, 0.05, 0.006], position - 0.006 * rotation[:, 2], quat=A._quat(rotation),
          rgba=(0.55, 0.56, 0.60, 1))
    fixed("mount_face", g.mjGEOM_PLANE, [0.4, 0.4, 0.01], position, quat=A._quat(rotation),
          rgba=(0.3, 0.5, 0.9, 0.12), group=3)

    p, H = PERSON, PERSON["height"]
    yaw = math.radians(p["facing_deg"])
    c, s = math.cos(yaw), math.sin(yaw)

    def at(fwd, left, z):
        return [p["x"] + c * fwd - s * left, p["y"] + s * fwd + c * left, z]

    skin, face = (0.93, 0.76, 0.62, 0.45), A._quat(Rotation.from_euler("z", yaw).as_matrix())
    fixed("person_legs", g.mjGEOM_BOX, [0.09, 0.15, 0.265 * H], at(0, 0, 0.265 * H), quat=face, rgba=skin)
    fixed("person_torso", g.mjGEOM_BOX, [0.11, 0.17, 0.145 * H], at(0, 0, 0.675 * H), quat=face, rgba=skin)
    fixed("person_head", g.mjGEOM_SPHERE, [0.10, 0, 0], at(0, 0, 0.925 * H), rgba=skin)
    for side, tag in ((1, "left"), (-1, "right")):
        fixed(f"person_{tag}_arm", g.mjGEOM_CAPSULE, [0.045, 0, 0], rgba=skin,
              fromto=[*at(0.0, side * 0.20, 0.80 * H), *at(0.04, side * 0.22, 0.45 * H)])

    hand = A.load_hand()
    if hand_turn_deg:
        turn = Rotation.from_euler("z", hand_turn_deg, degrees=True).as_matrix()
        hand = {k: v @ turn.T for k, v in hand.items()}
    link6 = spec.body("link6")
    for name, pts in hand.items():
        spec.add_mesh(name=f"hand_{name}", uservert=pts.reshape(-1).tolist())
        link6.add_geom(name=f"hand_{name}", type=g.mjGEOM_MESH, meshname=f"hand_{name}",
                       rgba=[0.35, 0.55, 0.85, 1.0])
    if decorate is not None:
        decorate(spec)
    model = spec.compile()

    names = [model.geom(i).name for i in range(model.ngeom)]
    body = [model.body(model.geom_bodyid[i]).name for i in range(model.ngeom)]
    mesh = [model.geom_type[i] == g.mjGEOM_MESH and not names[i].startswith("hand_") for i in range(model.ngeom)]
    joints = [model.joint(f"joint{k}").id for k in range(1, 7)]
    return Scene(
        model=model,
        qadr=model.jnt_qposadr[joints].copy(),
        env={n[4:]: i for i, n in enumerate(names) if n.startswith("env_")},
        person=[i for i, n in enumerate(names) if n.startswith("person_")],
        face=names.index("mount_face"),
        links={b: [i for i in range(model.ngeom) if mesh[i] and body[i] == b] for b in ARM_BODIES},
        base=[i for i in range(model.ngeom) if mesh[i] and (model.geom_bodyid[i] == 0 or body[i] == "base_link")],
        hand=[i for i, n in enumerate(names) if n.startswith("hand_")],
        base_pos=position, base_rot=rotation)


# --------------------------------------------------------------------------- distances

@dataclass
class Clear:
    """Clearances of one configuration (m) and what attains them."""
    env: dict[str, float]
    env_part: dict[str, str]
    self_: float
    self_part: str
    face: float
    face_part: str
    face_point: np.ndarray              # world point deepest toward / behind the mounting face
    person: float

    def other(self) -> tuple[float, str]:
        """Closest obstacle other than the desk."""
        name = min((n for n in self.env if n != "desk"), key=self.env.get)
        return self.env[name], name


def mesh_arrays(model: mujoco.MjModel, geom: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (geom frame) and faces of a mesh geom."""
    mid = model.geom_dataid[geom]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    return model.mesh_vert[va:va + vn].astype(float), model.mesh_face[fa:fa + fn]


def surface_points(verts: np.ndarray, faces: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Vertices plus area-uniform samples of the faces, thinned to one point per `spacing` voxel,
    plus every convex-hull vertex (so distances to planes are exact)."""
    from scipy.spatial import ConvexHull
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(np.ceil(6 * area.sum() / spacing ** 2))
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    samples = tri[pick, 0] * (1 - r1) + tri[pick, 1] * (r1 * (1 - r2)) + tri[pick, 2] * (r1 * r2)
    cloud = np.vstack([verts, samples])
    _, keep = np.unique(np.floor(cloud / spacing).astype(np.int64), axis=0, return_index=True)
    return np.vstack([cloud[np.sort(keep)], verts[ConvexHull(verts).vertices]])


@dataclass
class Part:
    geom: int
    name: str
    pts: np.ndarray        # surface points, geom frame
    centre: np.ndarray     # bounding sphere, geom frame
    radius: float


def sphere_directions(n: int) -> np.ndarray:
    """n near-uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


class Checker:
    """Clearances of one arm configuration, from surface point clouds (spacing ~5 mm).

    Obstacles are MuJoCo primitives (box, plane, sphere, capsule) with exact signed
    distance functions, so env/face/person distances are accurate to the sampling
    (a few mm). The hand-arm check measures the hand points against a 128-direction
    bounding polytope of base and links 1-3 (contains their convex hull, ~1 mm
    looser; half-space distance: exact inside, a lower bound outside), so it errs
    on the tight side. MuJoCo's mj_geomDistance is not used: in MuJoCo 3.5 it returns
    0 instead of the distance for some mesh/box pairs (e.g. link5 vs the desk once
    distmax exceeds the true distance)."""

    def __init__(self, scene: Scene, spacing: float = 0.005, distmax: float = DISTMAX):
        self.s, self.m, self.d = scene, scene.model, mujoco.MjData(scene.model)
        self.distmax = distmax
        rng = np.random.default_rng(0)
        names = {g: b for b in ARM_BODIES for g in scene.links[b]}
        names.update({g: "base" for g in scene.base})
        names.update({g: self.m.geom(g).name[5:] for g in scene.hand})
        self.parts: dict[int, Part] = {}
        for g in names:
            verts, faces = mesh_arrays(self.m, g)
            pts = surface_points(verts, faces, spacing, rng)
            centre = 0.5 * (pts.min(0) + pts.max(0))
            self.parts[g] = Part(g, names[g], pts, centre, float(np.linalg.norm(pts - centre, axis=1).max()))
        links = scene.links
        self.front = [g for b in ARM_BODIES[1:] for g in links[b]] + scene.hand  # links 2-6 + hand
        self._radii = np.array([self.parts[g].radius for g in self.front])
        normals = sphere_directions(128)
        self.hulls = {}  # geom -> (normals, offsets): bounding polytope, geom frame
        for g in scene.base + links["link1"] + links["link2"] + links["link3"]:
            verts, _ = mesh_arrays(self.m, g)
            self.hulls[g] = (normals, (verts @ normals.T).max(axis=0))

    def set(self, q) -> None:
        self.d.qpos[:] = 0.0
        self.d.qpos[self.s.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def _frame(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        return self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3)

    def sdf(self, obstacle: int, pts: np.ndarray) -> np.ndarray:
        """Signed distance of world points to a primitive geom (negative inside)."""
        m = self.m
        pos, rot = self._frame(obstacle)
        size, kind = m.geom_size[obstacle], m.geom_type[obstacle]
        g = mujoco.mjtGeom
        if kind == g.mjGEOM_PLANE:
            return (pts - pos) @ rot[:, 2]
        if kind == g.mjGEOM_SPHERE:
            return np.linalg.norm(pts - pos, axis=-1) - size[0]
        if kind == g.mjGEOM_CAPSULE:
            axis = rot[:, 2]
            t = np.clip((pts - pos) @ axis, -size[1], size[1])
            return np.linalg.norm(pts - pos - t[..., None] * axis, axis=-1) - size[0]
        if kind == g.mjGEOM_BOX:
            q = np.abs((pts - pos) @ rot) - size
            return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
        raise ValueError(f"obstacle geom {obstacle} has unsupported type {kind}")

    def world(self, part: Part) -> np.ndarray:
        pos, rot = self._frame(part.geom)
        return pos + part.pts @ rot.T

    def closest(self, obstacle: int, centres: np.ndarray, cache: dict) -> tuple[float, str, np.ndarray | None]:
        """Smallest signed distance of the front parts (links 2-6, hand) to an obstacle, the part
        and the world point attaining it."""
        bound = self.sdf(obstacle, centres) - self._radii  # sdf is 1-Lipschitz: a lower bound per part
        best, who, point = self.distmax, "", None
        for k in np.argsort(bound):
            if bound[k] >= best:
                break
            g = self.front[k]
            if g not in cache:
                cache[g] = self.world(self.parts[g])
            dist = self.sdf(obstacle, cache[g])
            i = int(np.argmin(dist))
            if dist[i] < best:
                best, who, point = float(dist[i]), self.parts[g].name, cache[g][i].copy()
        return best, who, point

    def hand_arm(self, cache: dict) -> tuple[float, str]:
        best, who = self.distmax, ""
        for h in self.s.hand:
            part = self.parts[h]
            hpos, hrot = self._frame(h)
            hc = hpos + hrot @ part.centre
            for g, (normals, offsets) in self.hulls.items():
                target = self.parts[g]
                tpos, trot = self._frame(g)
                if np.linalg.norm(hc - (tpos + trot @ target.centre)) - part.radius - target.radius >= best:
                    continue
                if h not in cache:
                    cache[h] = self.world(part)
                local = (cache[h] - tpos) @ trot
                dist = float((local @ normals.T - offsets).max(axis=1).min())
                if dist < best:
                    best, who = dist, f"{part.name}-{target.name}"
        return best, who

    def __call__(self, q) -> Clear:
        self.set(q)
        centres = np.array([self.d.geom_xpos[g] + self.d.geom_xmat[g].reshape(3, 3) @ self.parts[g].centre
                            for g in self.front])
        cache: dict = {}
        env, env_part = {}, {}
        for name in ENV_NAMES:
            env[name], env_part[name], _ = self.closest(self.s.env[name], centres, cache)
        self_, self_part = self.hand_arm(cache)
        face, face_part, face_point = self.closest(self.s.face, centres, cache)
        person = min(self.closest(p, centres, cache)[0] for p in self.s.person)
        return Clear(env, env_part, self_, self_part, face, face_part, face_point, person)


# --------------------------------------------------------------------------- gravity_comp motions

@dataclass
class Item:
    """One pose of a stage and the motion attributed to it (slew in + staging or swing)."""
    stage: str
    label: str
    q: np.ndarray                       # target / probe pose
    path: list[np.ndarray]              # key points from the previous item's end to this one's end
    probe: int | None = None            # probe joint (torque stages), swept +-sag
    sag: float = 0.0
    kind: str = "move"                  # capture, probe, sweep, move (slew) or drag (by hand)
    result: dict = field(default_factory=dict)


def load_gravity_comp(repo: Path):
    """gravity_comp modules of midas-piper-control (laptop-safe, no SDK)."""
    if not (repo / "gravity_comp" / "poses.py").exists():
        raise SystemExit(f"{repo} has no gravity_comp/poses.py (check out norman/piper-midas-gravity-comp)")
    sys.path.append(str(repo))
    import gravity_comp.measure_friction as mf
    import gravity_comp.model as gm
    import gravity_comp.poses as gp
    import gravity_comp.verify_delivery as vd
    import gravity_comp.workspace as gw
    return gp, vd, mf, gw, gm


def stage_items(gp, vd, mf, start: np.ndarray, catch_delivery: float, catch_friction: float,
                viscous: bool = False) -> dict[str, list[Item]]:
    stages: dict[str, list[Item]] = {}

    def capture(stage, poses):
        items, here = [], start
        for pose in poses:
            below, above = gp.approach_pair(pose)
            q = np.asarray(pose.q, float)
            items.append(Item(stage, pose.label, q, [here, below, q, above, q], kind="capture"))
            here = q
        items.append(Item(stage, "return to start", start, [here, start]))
        stages[stage] = items

    def probes(stage, plist, sag, here=start) -> list[Item]:
        items = []
        for joint, q, label in plist:
            q = np.asarray(q, float)
            up, down = q.copy(), q.copy()
            up[joint] += sag
            down[joint] -= sag
            items.append(Item(stage, label, q, [here, q, up, down, q], probe=joint, sag=sag, kind="probe"))
            here = q
        stages[stage] = items
        return items

    capture("S1 feedback", gp.feedback_char_poses())
    capture("S2 gravity", gp.gravity_poses())
    arm = gp.delivery_probes_arm()
    for j in range(3):  # --joint 1, 2, 3: one run each, from the start pose
        probes(f"S3 delivery J{j + 1}", [(p.joint, p.q, p.label) for p in arm if p.joint == j], catch_delivery)
    capture("S4a payload", gp.payload_poses())
    wrist = gp.delivery_probes_wrist()
    probes("S4b wrist", [(p.joint, p.q, p.label) for p in wrist if p.joint == 4]
           + [(p.joint, p.q, p.label) for p in wrist if p.joint == 3], catch_delivery)
    probes("S4c J6", [(5, vd.J6_PROBE_POSE, "J6 breakaway")], catch_delivery)
    friction = [(j, q, f"J{j + 1} pose{i}") for j in range(6) for i, q in enumerate(gp.friction_poses(j))]
    if not viscous:
        probes("S5 friction", friction, catch_friction)
    else:
        # measure_friction --viscous sweeps each joint right after its breakaway trials: slew to
        # one end, then a there-and-back sweep at each of VISCOUS_SPEEDS_RAD_S (0.10, 0.25 rad/s)
        # over +-VISCOUS_SWEEP_FRACTION of the joint range around its friction pose 1.
        name, items, here, limits = "S5 friction+viscous", [], start, gp.JOINT_LIMITS
        for j in range(6):
            items += probes(name, [f for f in friction if f[0] == j], catch_friction, here)
            base = np.asarray(gp.friction_poses(j)[1], float)
            half = mf.VISCOUS_SWEEP_FRACTION * (limits[j, 1] - limits[j, 0]) / 2
            a, b = base.copy(), base.copy()
            a[j] -= half
            b[j] += half
            items.append(Item(name, f"J{j + 1} viscous sweeps +-{half:.2f}", base, [items[-1].q, a, b, a, b, a],
                              kind="sweep"))
            here = a
        stages[name] = items
    accept, here, items = gp.acceptance_poses(), start, []
    for i, q in enumerate(accept):
        q = np.asarray(q, float)
        items.append(Item("S6 accept", f"A1 pose{i + 1}", q, [here, q], kind="drag"))
        here = q
    stages["S6 accept"] = items
    return stages


def densify(path: list[np.ndarray], step: float) -> np.ndarray:
    out = [np.asarray(path[0], float)]
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / step)))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.asarray(out)


# --------------------------------------------------------------------------- evaluation

def evaluate(items: list[Item], check: Checker, guard, step: float) -> None:
    """Fill it.result: the pose's clearances and the worst over its motion (with the q there)."""
    for it in items:
        static = check(it.q)
        samples = densify(it.path, step)
        worst = dict(desk=(np.inf, "", None), other=(np.inf, "", "", None), self=(np.inf, "", None),
                     face=(np.inf, "", None, None), person=(np.inf, None))
        for q in samples:
            c = check(q)
            if c.env["desk"] < worst["desk"][0]:
                worst["desk"] = (c.env["desk"], c.env_part["desk"], q)
            other, name = c.other()
            if other < worst["other"][0]:
                worst["other"] = (other, name, c.env_part[name], q)
            if c.self_ < worst["self"][0]:
                worst["self"] = (c.self_, c.self_part, q)
            if c.face < worst["face"][0]:
                worst["face"] = (c.face, c.face_part, c.face_point, q)
            if c.person < worst["person"][0]:
                worst["person"] = (c.person, q)
        pre = guard.precheck_pose(it.q, probe_joint=it.probe, sag_rad=it.sag)[1]
        path = min(guard.clearance(q)[0] for q in samples[::2])
        it.result = dict(static=static, worst=worst, guard_pre=pre, guard_path=path)


def mm(v: float) -> str:
    return f">{DISTMAX * 1000:.0f}" if v >= DISTMAX - 1e-9 else f"{v * 1000:.0f}"


def notes_of(it: Item, warn: float, margin: float) -> list[str]:
    r, w = it.result, it.result["worst"]
    out = []
    for key, what in (("desk", "desk"), ("other", None)):
        dist = w[key][0]
        name = what or w[key][1]
        if dist < 0:
            out.append(f"HITS {name} ({w[key][1 if what else 2]})")
        elif dist < warn:
            out.append(f"{name} {dist * 1000:.0f} mm")
    if w["self"][0] < 0:
        out.append(f"HAND HITS ARM ({w['self'][1]})")
    if w["face"][0] < 0:
        x, y, z = w["face"][2]
        out.append(f"{-w['face'][0] * 1000:.0f} mm behind face ({w['face'][1]} at x{x:+.2f} y{y:+.2f} z{z:.2f})")
    if w["person"][0] < 0:
        out.append("into the config-2 person")
    if r["guard_pre"] < margin:
        out.append("guard refuses the stage")
    elif r["guard_path"] < margin:
        out.append("guard stops the slew")
    hit = min(w["desk"][0], w["other"][0], w["self"][0]) < 0
    if hit and r["guard_pre"] >= margin and r["guard_path"] >= margin:
        out.append("GUARD MISSES IT")
    return out


def report(stages: dict[str, list[Item]], warn: float, margin: float) -> list[tuple[Item, str]]:
    flagged = []
    head = (f"{'pose':25s} {'desk pose/move':>14s} {'(part)':8s} {'other':>5s} {'(obj)':7s} {'hand-arm':>8s} "
            f"{'face pose/move':>14s} {'person':>6s} {'guard':>5s}  notes")
    for stage, items in stages.items():
        print(f"\n== {stage} " + "=" * max(0, len(head) - len(stage) - 4))
        print(head)
        for it in items:
            r, st, w = it.result, it.result["static"], it.result["worst"]
            notes = notes_of(it, warn, margin)
            print(f"{it.label[:25]:25s} {mm(st.env['desk']):>6s}/{mm(w['desk'][0]):>7s} {w['desk'][1][:8]:8s} "
                  f"{mm(w['other'][0]):>5s} {w['other'][1][:7]:7s} {mm(w['self'][0]):>8s} "
                  f"{mm(st.face):>6s}/{mm(w['face'][0]):>7s} {mm(w['person'][0]):>6s} "
                  f"{mm(min(r['guard_pre'], r['guard_path'])):>5s}  {'; '.join(notes)}")
            if notes:
                flagged.append((it, "; ".join(notes)))
    return flagged


# --------------------------------------------------------------------------- pictures

def render(scene: Scene, check: Checker, flagged: list[tuple[Item, str]], out: Path) -> list[Path]:
    """One picture per flagged pose at its worst sample: a view from the north-east and one from
    above; the mounting face is the translucent blue plane."""
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    m = scene.model
    width, height = 720, 540
    renderer = mujoco.Renderer(m, height=height, width=width)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    views = []
    for azimuth, elevation, distance in ((-135.0, -15.0, 1.9), (-90.0, -89.0, 2.0)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = scene.base_pos + np.array([0.12, 0.15, -0.30])
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        views.append(cam)
    paths = []
    for k, (it, note) in enumerate(flagged):
        w = it.result["worst"]
        q = min((w["desk"][0], w["desk"][2]), (w["other"][0], w["other"][3]), (w["self"][0], w["self"][2]),
                (w["face"][0], w["face"][3]), key=lambda t: t[0])[1]
        q = it.q if q is None else q
        check.set(q)
        sheet = Image.new("RGB", (2 * width, height + 50), (0, 0, 0))
        for i, cam in enumerate(views):
            renderer.update_scene(check.d, camera=cam, scene_option=opt)
            sheet.paste(Image.fromarray(renderer.render()), (i * width, 50))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), f"{it.stage} | {it.label} | q (rad) = {np.round(q, 3).tolist()}", fill=(255, 255, 255))
        draw.text((8, 26), note, fill=(255, 200, 120))
        path = out / f"{k:02d}_{it.stage.split()[0]}_{''.join(ch if ch.isalnum() else '_' for ch in it.label)}.png"
        sheet.save(path)
        paths.append(path)
    renderer.close()
    return paths


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--midas-repo", type=Path, default=Path("~/workspace/midas-piper-control"),
                   help="midas-piper-control checkout with gravity_comp (norman/piper-midas-gravity-comp)")
    p.add_argument("--base-above-desk", type=float, default=0.49, help="base origin above the desk top (m)")
    p.add_argument("--desk-height", type=float, default=DESK_HEIGHT, help="desk top above the floor (m)")
    p.add_argument("--start-deg", type=float, nargs=6, default=[0.0] * 6,
                   help="joints the arm starts each stage from (deg); the first slew starts here")
    p.add_argument("--hand-turn-deg", type=float, default=0.0,
                   help="extra turn of the hand about the flange axis vs the sim mount (CAD's own: +57.5)")
    p.add_argument("--workspace-repo", type=Path, default=None,
                   help="trimorph-piper-control checkout with the workspace sim (default: this repo if it "
                        "has scripts/workspace, else ~/trimorph-piper-control)")
    p.add_argument("--step", type=float, default=0.005, help="path sampling: max joint step (rad)")
    p.add_argument("--catch-delivery", type=float, default=None, help="S3/S4 probe swing (rad, default: theirs)")
    p.add_argument("--catch-friction", type=float, default=None, help="S5 probe swing (rad, default: theirs)")
    p.add_argument("--viscous", action="store_true", help="also check measure_friction --viscous sweeps")
    p.add_argument("--warn-mm", type=float, default=50.0, help="flag env clearance below this (mm)")
    p.add_argument("--stages", nargs="*", help="only stages whose name starts with one of these (e.g. S2 S3)")
    p.add_argument("--render", action="store_true", help=f"save a picture of each flagged pose to {OUTPUT}")
    args = p.parse_args(argv)

    gp, vd, mf, gw, gm = load_gravity_comp(args.midas_repo.expanduser())
    catch_d = vd.parse_args(["--j6"]).catch_rad if args.catch_delivery is None else args.catch_delivery
    catch_f = mf.CATCH_RAD if args.catch_friction is None else args.catch_friction
    scene = build_scene(args.desk_height, args.base_above_desk, args.hand_turn_deg,
                        workspace_repo=args.workspace_repo)
    check = Checker(scene)
    model = gm.ArmGravityModel(gravity=gm.GRAVITY_RIGHT_MOUNT)
    guard = gw.WorkspaceGuard(model, [gw.table_plane(np.asarray(gm.GRAVITY_RIGHT_MOUNT), args.base_above_desk)])
    start = np.radians(args.start_deg)
    stages = stage_items(gp, vd, mf, start, catch_d, catch_f, args.viscous)
    if args.stages:
        stages = {k: v for k, v in stages.items() if any(k.startswith(s) for s in args.stages)}

    print(f"layout: config 2, base {scene.base_pos.round(3).tolist()} (J1 axis east), desk top "
          f"{args.desk_height:.3f} m = {args.base_above_desk:.3f} m below the base, desk x "
          f"{DESK_CENTRE[0] - DESK_SIZE[0] / 2:.3f}..{DESK_CENTRE[0] + DESK_SIZE[0] / 2:.3f} y "
          f"{DESK_CENTRE[1] - DESK_SIZE[1] / 2:.3f}..{DESK_CENTRE[1] + DESK_SIZE[1] / 2:.3f}")
    print(f"hand: TRIMI hulls {sorted(n[5:] for n in (scene.model.geom(g).name for g in scene.hand))}"
          + (f", turned {args.hand_turn_deg:+.1f} deg about the flange" if args.hand_turn_deg else ""))
    print(f"start pose (deg): {np.round(args.start_deg, 1).tolist()}; catch radius S3/S4 {catch_d} rad, S5 {catch_f} rad")
    print(f"guard: table plane {args.base_above_desk} m below the base, margin {guard.margin_m * 1000:.0f} mm, "
          f"hand reach {guard.hand_reach_m} m (gravity_comp defaults)")
    print("columns (mm): desk and face at the pose / min over its motion (slew in + staging or probe swing);")
    print("  other = cart, post, floor, mount plate; face < 0 = behind the base's mounting face;")
    print("  person = config-2 wearer (info); guard = gravity_comp's own clearance (stage refuses/stops < "
          f"{guard.margin_m * 1000:.0f})")
    for items in stages.values():
        evaluate(items, check, guard, args.step)
    flagged = report(stages, args.warn_mm / 1000, guard.margin_m)

    def low(items, key):
        return min(it.result["worst"][key][0] for it in items)

    def hit(it):
        w = it.result["worst"]
        return min(w["desk"][0], w["other"][0], w["self"][0]) < 0

    print(f"\nsummary (min over each stage, mm)\n  {'stage':20s} {'desk':>5s} {'other':>5s} {'hand-arm':>8s} "
          f"{'face':>5s} {'person':>6s} {'guard':>5s}")
    for stage, items in stages.items():
        guard_min = min(min(it.result["guard_pre"], it.result["guard_path"]) for it in items)
        hits = [it.label for it in items if hit(it)]
        print(f"  {stage:20s} {mm(low(items, 'desk')):>5s} {mm(low(items, 'other')):>5s} {mm(low(items, 'self')):>8s} "
              f"{mm(low(items, 'face')):>5s} {mm(low(items, 'person')):>6s} {mm(guard_min):>5s}"
              + (f"  COLLIDES: {', '.join(hits)}" if hits else ""))
    if args.render and flagged:
        for path in render(scene, check, flagged, OUTPUT):
            print(f"  picture: {path}")
    return 1 if any(hit(it) for items in stages.values() for it in items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
