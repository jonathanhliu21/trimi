"""Set the mirror's joint zeros from the hard stops that homing drives into.

    python3 scripts/sim/sync_hard_stops.py --source ../TRIMI_URDF_v2.zip           # report
    python3 scripts/sim/sync_hard_stops.py --source ../TRIMI_URDF_v2.zip --write   # save zeros

Homing drives each motor until its link hits a mechanical stop and saves the
offset so that the motor then reads -joint_sign * cad_offset (the homing table
in the hand YAML). This script replays the homing sequence on the URDF meshes
(full resolution with --source) and finds where each stop is; the reading at
the stop and the URDF angle there fix the zero:

    zero = reading_at_stop - q_urdf_at_stop / (sign * scale)

The replay follows midas_hand_api's homing: thumb tip to base, then index,
middle, ring (PIP, MCP, ABAD each). A homed motor holds its stop until the end,
so the homed link (with the parts fixed to it) meets the palm, the links it is
mounted on and the fingers homed before it, all posed where homing holds them.
Fingers not yet homed are unpowered and move aside, so they are ignored; the
middle MCP goes to its pre-abduction angle before the middle ABAD homes. Links
further out on the finger are not counted: in the URDF the ring's proximal
link would meet the palm's thin hand band near -75 deg, which the real hand
passes (its ABAD stops near -90 deg, on the ABAD link).

A stop is where the moving links first press into an obstacle. Depths are
exact distances to the obstacle surface, counted beyond any overlap the CAD
already has in the starting pose (a cover modelled 0.5 mm into a motor case
slides along it, it does not stop on it), and the angle of first touch is
extrapolated from how fast the deepest points sink. Parts that turn with the
joint in reality but sit in the parent mesh are ignored: the motor horn and
shaft around the joint axis and, on geared joints, the drive gear and the
sector teeth that mesh with it.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# Support both direct execution and package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial import cKDTree

from scripts.sim.build_trimi_sim_model import locate_urdf, read_stl
from scripts.sim.trimi_mirror import (  # noqa: F401  (homing helpers re-exported)
    DEFAULT_JOINT_MAP,
    DEFAULT_MODEL,
    HomingStop,
    default_hand_config,
    homing_rows,
    homing_stops,
    load_joint_map,
    write_zeros,
)

STEP_RAD = math.radians(1.0)
BISECTIONS = 10
EXTRAPOLATE_RAD = math.radians(0.5)  # second angle for the sinking rate at first touch
SAMPLE_CELL_M = 0.0004
PENETRATION_M = 0.0002  # depth beyond the starting overlap that counts as pressing in
MIN_SINK_RATE = 0.003  # m/rad (0.05 mm per degree); slower points slide rather than stop
MAX_EDGE_M = 0.003  # finer triangles, so no surface point lies far from a centroid
CANDIDATE_M = 0.004
LOCAL_RADIUS_M = 0.005
MIN_POINTS = 4
DIRECT_CORE_M = 0.013  # motor horn and its screws around a direct-drive axis
GEARED_CORE_M = 0.0095  # shaft and bushing around a geared joint axis
TOOTH_BAND_M = (0.026, 0.034)  # 84-tooth sector, pitch radius 31.5 mm
DRIVE_TEETH = 34
DRIVE_CENTRE_RANGE_M = (0.0425, 0.0461)
DRIVE_RADIUS_M = 0.015
DRIVE_HALF_THICKNESS_M = 0.006
# midas_hand_api.homing: the middle MCP is set to an angle before the middle ABAD homes.
MIDDLE_MCP_FLEX_MOTOR_ID = 8
MIDDLE_MCP_ABDUCT_MOTOR_ID = 9


@dataclass(frozen=True)
class Sweep:
    q_stop: float
    contact_radius_m: float
    obstacles: tuple[str, ...]  # obstacle meshes pressed into
    movers: tuple[str, ...]  # moving meshes that press
    note: str = ""


@dataclass(frozen=True)
class StopResult:
    motor_id: int
    joint: str
    q_stop: float
    zero: float
    reading: float
    contact_radius_m: float
    obstacles: tuple[str, ...]
    movers: tuple[str, ...] = ()
    note: str = ""


def zero_from_stop(reading: float, q_stop: float, sign: float, scale: float) -> float:
    return reading - q_stop / (sign * scale)


def subdivide(tris: np.ndarray, max_edge: float = MAX_EDGE_M) -> np.ndarray:
    """Split triangles along their longest edge until no edge exceeds max_edge."""

    done, todo = [], np.asarray(tris, dtype=np.float64)
    while len(todo):
        edges = np.stack([
            np.linalg.norm(todo[:, 1] - todo[:, 0], axis=1),
            np.linalg.norm(todo[:, 2] - todo[:, 1], axis=1),
            np.linalg.norm(todo[:, 0] - todo[:, 2], axis=1),
        ], axis=1)
        big = edges.max(axis=1) > max_edge
        done.append(todo[~big])
        todo, longest = todo[big], edges[big].argmax(axis=1)
        order = np.stack([longest, (longest + 1) % 3, (longest + 2) % 3], axis=1)
        todo = np.take_along_axis(todo, order[:, :, None].repeat(3, axis=2), axis=1)
        mid = (todo[:, 0] + todo[:, 1]) / 2
        todo = np.concatenate([
            np.stack([todo[:, 0], mid, todo[:, 2]], axis=1),
            np.stack([mid, todo[:, 1], todo[:, 2]], axis=1),
        ])
    return np.concatenate(done)


def solid_angles(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = tris[:, 0] - p, tris[:, 1] - p, tris[:, 2] - p
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
           + np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb)
    return 2.0 * np.arctan2(num, den)


def point_triangle_distance(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact distance from p to each triangle (closest point by Voronoi region)."""

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        out = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]  # inside the face
        # Edges and vertices, lowest priority first so the earlier tests win.
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        out[m] = (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None])[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out[m] = (a + ac * (d2 / (d2 - d6))[:, None])[m]
        m = (d6 >= 0) & (d5 <= d6)
        out[m] = c[m]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out[m] = (a + ab * (d1 / (d1 - d3))[:, None])[m]
        m = (d3 >= 0) & (d4 <= d3)
        out[m] = b[m]
        m = (d1 <= 0) & (d2 <= 0)
        out[m] = a[m]
    d = np.linalg.norm(p - out, axis=1)
    return np.where(np.isfinite(d), d, np.inf)


class HardStopFinder:
    """Sweep hinge joints of an MJCF hand against full-resolution link meshes."""

    def __init__(self, model, mesh_dir: Path):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self._fine: dict[int, list[tuple[np.ndarray, str]]] = {}  # body -> subdivided local triangles
        cache: dict[str, np.ndarray] = {}
        for geom in range(model.ngeom):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[geom]:
                continue  # visual meshes only
            mesh = model.geom_dataid[geom]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh)
            if name not in cache:
                vertices, faces = read_stl(mesh_dir / f"{name}.stl")
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, model.mesh_quat[mesh])
                cache[name] = subdivide(((vertices - model.mesh_pos[mesh]) @ rot.reshape(3, 3))[faces])
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
            local = model.geom_pos[geom] + cache[name] @ rot.reshape(3, 3).T
            self._fine.setdefault(int(model.geom_bodyid[geom]), []).append((local, name))
        self._samples: dict[int, list[tuple[np.ndarray, str]]] = {}

    # -- bodies
    def body_joint(self, body: int) -> int:
        """The hinge that moves a body (its own or its nearest ancestor's), -1 for the palm."""

        while body:
            if self.model.body_jntnum[body]:
                return int(self.model.body_jntadr[body])
            body = self.model.body_parentid[body]
        return -1

    def ancestors(self, body: int) -> set[int]:
        out = set()
        body = self.model.body_parentid[body]
        while body:
            out.add(int(body))
            body = self.model.body_parentid[body]
        return out

    def subtree(self, body: int) -> set[int]:
        return {b for b in range(1, self.model.nbody) if b == body or body in self.ancestors(b)}

    def palm(self) -> set[int]:
        return {b for b in self._fine if self.body_joint(b) < 0}

    def homing_obstacles(self, joint: int, held: set[int]) -> set[int]:
        """Palm, the links the joint is mounted on, and links of motors already holding a stop."""

        link = int(self.model.jnt_bodyid[joint])
        out = self.palm() | self.ancestors(link)
        out |= {b for b in self._fine if self.body_joint(b) in held}
        return out - self.subtree(link)

    def _world(self, body: int, points: np.ndarray) -> np.ndarray:
        return self.data.xpos[body] + points @ self.data.xmat[body].reshape(3, 3).T

    def _pose(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.mujoco.mj_kinematics(self.model, self.data)

    def _body_samples(self, body: int) -> list[tuple[np.ndarray, str]]:
        if body not in self._samples:
            out = []
            for fine, name in self._fine.get(body, []):
                points = np.vstack([fine.reshape(-1, 3), fine.mean(axis=1)])
                _, idx = np.unique(np.floor(points / SAMPLE_CELL_M).astype(np.int64), axis=0, return_index=True)
                out.append((points[np.sort(idx)], name))
            self._samples[body] = out
        return self._samples[body]

    def _drive_gear(self, joint: int, origin, axis, parent_tris: np.ndarray):
        """Centre of the parent's 34-tooth drive gear, by tooth-frequency search."""

        child = self.model.jnt_bodyid[joint]
        u = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 0.1:
            u = np.cross(axis, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        child_pts = np.vstack([self._world(child, p) for p, _ in self._body_samples(child)])
        radial = np.hypot((child_pts - origin) @ u, (child_pts - origin) @ v)
        band = child_pts[(radial > 0.028) & (radial < 0.033)]
        if len(band) < 50 or not len(parent_tris):
            return None
        hist, edges = np.histogram((band - origin) @ axis, bins=80)
        k = int(np.argmax(hist))
        plane = (edges[k] + edges[k + 1]) / 2
        parent_pts = parent_tris.reshape(-1, 3)
        slab = parent_pts[np.abs((parent_pts - origin) @ axis - plane) < 0.001]
        flat = np.column_stack([(slab - origin) @ u, (slab - origin) @ v])

        def strength(centre):
            rel = flat - centre
            r = np.hypot(rel[:, 0], rel[:, 1])
            sel = (r > 0.011) & (r < 0.0145)
            bins = 1440
            idx = (np.mod(np.arctan2(rel[sel, 1], rel[sel, 0]), 2 * np.pi) / (2 * np.pi) * bins).astype(int) % bins
            profile = np.full(bins, np.nan)
            np.fmax.at(profile, idx, r[sel])
            ok = ~np.isnan(profile)
            if ok.mean() < 0.2:
                return 0.0
            x = np.interp(np.arange(bins), np.arange(bins)[ok], profile[ok], period=bins)
            spectrum = np.abs(np.fft.rfft(x - x.mean()))
            return float(spectrum[DRIVE_TEETH] / np.median(spectrum[10:200]))

        best, centre = max(
            ((strength(dist * np.array([math.cos(t), math.sin(t)])), dist * np.array([math.cos(t), math.sin(t)]))
             for dist in np.arange(*DRIVE_CENTRE_RANGE_M, 0.0005)
             for t in np.radians(np.arange(0, 360, 2))),
            key=lambda item: item[0],
        )
        for step in (0.0002, 0.00005):
            for _ in range(30):
                trial = [centre + step * d for d in np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])]
                scores = [strength(c) for c in trial]
                if max(scores) <= best:
                    break
                best, centre = max(scores), trial[int(np.argmax(scores))]
        if best < 5.0:
            return None
        return origin + centre[0] * u + centre[1] * v, plane, best

    def find(self, joint_name: str, direction: int, *, geared: bool, qpos=None, obstacles=None) -> Sweep:
        """Turn one joint from 0 in `direction`, the rest held at `qpos`, until it meets `obstacles`.

        `obstacles` are body ids (default: the palm and the links the joint is
        mounted on). The joint's link and the bodies fixed to it move; links on
        further joints are left out of both sets.
        """

        mujoco = self.mujoco
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint < 0:
            raise ValueError(f"No joint {joint_name!r} in the model")
        address = self.model.jnt_qposadr[joint]
        link = int(self.model.jnt_bodyid[joint])
        pose = np.zeros(self.model.nq) if qpos is None else np.array(qpos, dtype=np.float64)
        pose[address] = 0.0
        self._pose(pose)
        origin, axis = self.data.xanchor[joint].copy(), self.data.xaxis[joint].copy()
        if obstacles is None:
            obstacles = self.homing_obstacles(joint, set())
        moving = sorted(b for b in self.subtree(link) & set(self._fine) if self.body_joint(b) == joint)

        def radius(points):
            rel = points - origin
            return np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)

        # Obstacles stay put while the joint turns.
        tris, owners, parent_tris = [], [], []
        names: list[str] = []
        for body in sorted(set(obstacles) - set(moving)):
            for fine, name in self._fine.get(body, []):
                world = self._world(body, fine.reshape(-1, 3)).reshape(-1, 3, 3)
                tris.append(world)
                names.append(name)
                owners.append(np.full(len(world), len(names) - 1))
                if body == self.model.body_parentid[link]:
                    parent_tris.append(world)
        if not tris:
            raise RuntimeError(f"{joint_name}: nothing to stop against")
        tris, owners = np.concatenate(tris), np.concatenate(owners)
        note = ""
        if geared:
            gear = self._drive_gear(joint, origin, axis, np.concatenate(parent_tris) if parent_tris else np.empty((0, 3, 3)))
            if gear is None:
                note = "drive gear not found; geared contacts may be early"
            else:
                centre, plane, _strength = gear
                mid = tris.mean(axis=1)
                rel = mid - centre
                gear_radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
                keep = ~((gear_radial < DRIVE_RADIUS_M)
                         & (np.abs((mid - origin) @ axis - plane) < DRIVE_HALF_THICKNESS_M))
                tris, owners = tris[keep], owners[keep]
        tree = cKDTree(tris.mean(axis=1))

        # Moving sample points, minus the horn/shaft core and the link's own sector teeth.
        core = GEARED_CORE_M if geared else DIRECT_CORE_M
        samples, sample_body, mover_names, mover = [], [], [], []
        for body in moving:
            for points, name in self._body_samples(body):
                r = radius(self._world(body, points))
                keep = r > core
                if geared and body == link:
                    keep &= ~((r > TOOTH_BAND_M[0]) & (r < TOOTH_BAND_M[1]))
                samples.append(points[keep])
                sample_body.append(np.full(int(keep.sum()), body))
                mover_names.append(name)
                mover.append(np.full(int(keep.sum()), len(mover_names) - 1))
        samples, sample_body, mover = np.vstack(samples), np.concatenate(sample_body), np.concatenate(mover)

        def depths(q):
            """Depth inside an obstacle (0 outside) and the obstacle mesh index of each sample."""

            pose[address] = q
            self._pose(pose)
            world = np.empty_like(samples)
            for body in moving:
                sel = sample_body == body
                world[sel] = self._world(body, samples[sel])
            depth, owner = np.zeros(len(world)), np.full(len(world), -1)
            dist, _ = tree.query(world, distance_upper_bound=CANDIDATE_M)
            candidates = np.where(np.isfinite(dist))[0]
            if not len(candidates):
                return depth, owner, world
            for i, near in zip(candidates, tree.query_ball_point(world[candidates], LOCAL_RADIUS_M)):
                if len(near) < 3:
                    continue
                local = tris[near]
                if solid_angles(world[i], local).sum() / (4 * np.pi) <= 0.25:
                    continue  # outside
                d = point_triangle_distance(world[i], local)
                k = int(np.argmin(d))
                depth[i], owner[i] = d[k], owners[near[k]]
            return depth, owner, world

        depth0, owner0, _ = depths(0.0)
        baseline = depth0 > 0  # already inside in the starting pose
        overlap = np.zeros(len(names))
        np.maximum.at(overlap, owner0[baseline], depth0[baseline])

        def pressing(q):
            depth, owner, world = depths(q)
            hit = ~baseline & (owner >= 0)
            hit[hit] = depth[hit] > overlap[owner[hit]] + PENETRATION_M
            return hit, depth, owner, world

        previous, q, found = 0.0, 0.0, None
        while abs(q) < 2 * np.pi - STEP_RAD:
            q += direction * STEP_RAD
            if pressing(q)[0].sum() >= MIN_POINTS:
                found = q
                break
            previous = q
        if found is None:
            raise RuntimeError(f"{joint_name}: no stop within a full turn")
        lo, hi = previous, found
        for _ in range(BISECTIONS):
            mid = (lo + hi) / 2
            if pressing(mid)[0].sum() >= MIN_POINTS:
                hi = mid
            else:
                lo = mid
        hit, depth1, owner, world = pressing(hi)
        # First touch: run each pressing point back along its sinking rate to zero depth.
        depth2, _, _ = depths(hi + direction * EXTRAPOLATE_RAD)
        idx = np.where(hit)[0]
        rate = (depth2[idx] - depth1[idx]) / EXTRAPOLATE_RAD
        ok = rate > MIN_SINK_RATE
        if ok.sum() >= MIN_POINTS:
            back = depth1[idx[ok]] / rate[ok]
            order = np.argsort(back)[::-1][:MIN_POINTS]  # the earliest touches
            q_stop = hi - direction * min(back[order[-1]], abs(hi))
            touching = idx[ok][order]
        else:
            q_stop, touching = (lo + hi) / 2, idx
        self._pose(np.zeros(self.model.nq))
        return Sweep(
            q_stop=float(q_stop),
            contact_radius_m=float(radius(world[touching]).mean()),
            obstacles=tuple(sorted({names[o] for o in owner[touching]})),
            movers=tuple(sorted({mover_names[m] for m in mover[touching]})),
            note=note,
        )


def find_stops(model, mesh_dir: Path, entries, hand: Mapping, motor_ids=None, *, progress=None) -> list[StopResult]:
    """Replay the homing sequence; motors not in motor_ids are posed from their current zeros."""

    stops = homing_stops(hand)
    by_motor = {entry.motor_id: entry for entry in entries}
    finder = HardStopFinder(model, mesh_dir)
    joint_of = {}
    for entry in entries:
        joint = model.joint(entry.urdf_joint).id
        joint_of[entry.motor_id] = joint
    qpos, held, zeros, results = np.zeros(model.nq), set(), {}, []
    pre_abduct = hand["homing"].get("middle_mcp_flex_pre_abduct_position_rad")
    for row in homing_rows(hand):
        motor = int(row[0])
        entry = by_motor.get(motor)
        if entry is None:
            continue
        if motor not in stops:
            raise ValueError(f"Motor {motor} has no row in the hand's homing table")
        if motor == MIDDLE_MCP_ABDUCT_MOTOR_ID and pre_abduct is not None and MIDDLE_MCP_FLEX_MOTOR_ID in by_motor:
            flex = by_motor[MIDDLE_MCP_FLEX_MOTOR_ID]
            zero = zeros.get(flex.motor_id, flex.zero)
            qpos[model.jnt_qposadr[joint_of[flex.motor_id]]] = flex.sign * flex.scale * (float(pre_abduct) - zero)
        stop, joint = stops[motor], joint_of[motor]
        if motor_ids is None or motor in motor_ids:
            if progress:
                progress(f"motor {motor} ({entry.urdf_joint})")
            sweep = finder.find(
                entry.urdf_joint, int(np.sign(entry.sign * stop.direction)),
                geared=abs(entry.scale - 1.0) > 1e-6, qpos=qpos, obstacles=finder.homing_obstacles(joint, held),
            )
            zeros[motor] = zero_from_stop(stop.reading, sweep.q_stop, entry.sign, entry.scale)
            q_stop = sweep.q_stop
            results.append(StopResult(
                motor_id=motor, joint=entry.urdf_joint, q_stop=q_stop, zero=zeros[motor],
                reading=stop.reading, contact_radius_m=sweep.contact_radius_m,
                obstacles=sweep.obstacles, movers=sweep.movers, note=sweep.note,
            ))
        else:
            q_stop = entry.sign * entry.scale * (stop.reading - entry.zero)
        qpos[model.jnt_qposadr[joint]] = q_stop
        held.add(joint)
    return results


def run(args: argparse.Namespace) -> int:
    import mujoco
    import yaml

    model = mujoco.MjModel.from_xml_path(str(args.model))
    entries = load_joint_map(args.joint_map)
    with args.hand_config.open("r", encoding="utf-8") as stream:
        hand = yaml.safe_load(stream)
    print(f"Hand calibration: {args.hand_config}\nModel: {args.model}")
    with tempfile.TemporaryDirectory() as tmp:
        if args.source is not None:
            mesh_dir = locate_urdf(args.source, Path(tmp)).parent.parent / "meshes"
        else:
            mesh_dir = args.model.parent / "meshes"
            print("Using the decimated model meshes; pass --source for full resolution.")
        results = find_stops(model, mesh_dir, entries, hand, set(args.motors) if args.motors else None,
                             progress=lambda what: print(f"  sweeping {what}", flush=True))
    old = {entry.motor_id: entry.zero for entry in entries}
    print(f"\n{'motor':>5} {'URDF joint':13s} {'stop (deg)':>10} {'reading':>8} {'zero':>8} {'change':>7}"
          "  contact")
    for r in results:
        change = math.degrees(r.zero - old[r.motor_id])
        print(f"{r.motor_id:>5} {r.joint:13s} {math.degrees(r.q_stop):+10.2f} {r.reading:+8.4f} {r.zero:+8.4f}"
              f" {change:+6.1f}d  {', '.join(r.movers)} -> {', '.join(r.obstacles)}"
              f" ({r.contact_radius_m * 1000:.0f} mm from axis) {r.note}")
    if args.write:
        write_zeros(args.joint_map, {r.motor_id: r.zero for r in results})
        print(f"\nSaved the zeros to {args.joint_map}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path,
                        help="URDF zip/folder for full-resolution meshes (default: the model's decimated meshes)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--joint-map", type=Path, default=DEFAULT_JOINT_MAP)
    parser.add_argument("--hand-config", type=Path, default=None,
                        help="hand YAML with the homing table that was used (default: as trimi_mirror.py)")
    parser.add_argument("--motors", type=int, nargs="+",
                        help="only these motor IDs (the others are posed from their current zeros)")
    parser.add_argument("--write", action="store_true", help="save the zeros to the joint map")
    args = parser.parse_args(argv)
    args.hand_config = (args.hand_config or default_hand_config()).expanduser()
    args.model = args.model.expanduser()
    args.joint_map = args.joint_map.expanduser()
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
