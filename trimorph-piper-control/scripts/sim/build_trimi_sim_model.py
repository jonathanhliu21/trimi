"""Build a MuJoCo model of the TriMI hand from its Onshape URDF export.

    python3 scripts/sim/build_trimi_sim_model.py ../TRIMI_URDF_v2.zip

The source may be the exported zip, its extracted folder, or the .urdf file.
Under --out (default assets/trimi_v2) this writes:

    meshes/*.stl     copies of the meshes, decimated where needed
    trimi_v2.urdf    the URDF with mesh paths and masses fixed
    trimi_v2.xml     the MJCF that scripts/sim/trimi_mirror.py loads

and prints what it found in the URDF and what it changed. The export cannot be
loaded into MuJoCo unmodified:

* The palm mesh has more than MuJoCo's 200,000-face STL limit. Large meshes are
  decimated by vertex clustering (0.5 mm cells by default).
* Onshape exported every part with a density of 1 kg/m^3 (no material
  assigned), so masses and inertias are about 1000x too small. They are scaled
  to --density (default 1000 kg/m^3) unless --mass-scale is given.
* There are no <collision> elements. Every moving link gets a collision geom
  from its own mesh (MuJoCo collides with its convex hull); hand geoms do not
  collide with each other.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path("assets/trimi_v2")
MODEL_NAME = "trimi_v2"
MUJOCO_MAX_STL_FACES = 200_000
DEFAULT_DECIMATE_ABOVE_FACES = 20_000
DEFAULT_CLUSTER_CELL_MM = 0.5
DEFAULT_DENSITY_KG_M3 = 1000.0
# Onshape writes a density of 1 kg/m^3 when a part has no material assigned.
UNASSIGNED_DENSITY_MAX_KG_M3 = 10.0

# contype/conaffinity bits: the hand collides with other objects, not itself.
HAND_CONTYPE, HAND_CONAFFINITY = 2, 1
VISUAL_GROUP, COLLISION_GROUP = 1, 3


# ---------------------------------------------------------------- STL meshes


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return welded (vertices [N, 3], faces [M, 3]) from a binary or ASCII STL."""

    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * count == len(data):
            record = np.dtype(
                [("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]
            )
            corners = np.frombuffer(data, record, count=count, offset=84)["v"]
            return _weld(corners.reshape(-1, 3).astype(np.float64))
    text = data.decode("ascii", errors="replace")
    values = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not values or len(values) % 3:
        raise ValueError(f"{path} is not a readable STL file")
    return _weld(np.asarray(values, dtype=np.float64))


def _weld(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices, inverse = np.unique(corners, axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    corners = vertices[faces].astype(np.float32)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.zeros(
        len(faces),
        dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")],
    )
    record["normal"] = normals
    record["v"] = corners
    with path.open("wb") as stream:
        stream.write(b"trimi sim mesh".ljust(80, b" "))
        stream.write(struct.pack("<I", len(faces)))
        stream.write(record.tobytes())


def cluster_decimate(
    vertices: np.ndarray, faces: np.ndarray, cell_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices that share a grid cell and drop collapsed faces."""

    cells = np.floor(vertices / cell_m).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = int(inverse.max()) + 1
    sums = np.zeros((count, 3))
    np.add.at(sums, inverse, vertices)
    merged = sums / np.bincount(inverse, minlength=count)[:, None]
    new_faces = inverse[faces]
    keep = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[keep]
    _, first = np.unique(np.sort(new_faces, axis=1), axis=0, return_index=True)
    return merged, new_faces[np.sort(first)]


def mesh_volume_m3(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed-volume sum; meaningful only for closed, consistently wound meshes."""

    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def is_closed(faces: np.ndarray) -> bool:
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


# ---------------------------------------------------------------- URDF source


def locate_urdf(source: Path, workdir: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for member in archive.namelist():
                if not (workdir / member).resolve().is_relative_to(workdir.resolve()):
                    raise ValueError(f"Unsafe path in {source}: {member}")
            archive.extractall(workdir)
        source = workdir
    if source.is_dir():
        candidates = sorted(source.rglob("*.urdf"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .urdf under {source}, found {len(candidates)}"
            )
        source = candidates[0]
    if source.suffix.lower() != ".urdf" or not source.is_file():
        raise FileNotFoundError(f"No URDF at {source}")
    return source


def find_mesh(urdf_path: Path, filename: str) -> Path:
    name = Path(filename.split("://", 1)[-1]).name
    for directory in (
        urdf_path.parent,
        urdf_path.parent / "meshes",
        urdf_path.parent.parent / "meshes",
    ):
        if (directory / name).is_file():
            return directory / name
    matches = sorted(urdf_path.parent.parent.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Mesh {filename} referenced by {urdf_path} not found")
    return matches[0]


@dataclass
class MeshReport:
    name: str
    faces_in: int
    faces_out: int
    mass_kg: float
    volume_m3: float | None


def convert_meshes(
    robot: ET.Element,
    urdf_path: Path,
    mesh_out: Path,
    *,
    decimate_above: int,
    cell_m: float,
) -> list[MeshReport]:
    """Copy every referenced mesh into mesh_out, decimating large ones."""

    mesh_out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, MeshReport] = {}
    for link in robot.findall("link"):
        mass_element = link.find("inertial/mass")
        mass = float(mass_element.get("value")) if mass_element is not None else 0.0
        for mesh in link.findall("./visual/geometry/mesh") + link.findall(
            "./collision/geometry/mesh"
        ):
            source = find_mesh(urdf_path, mesh.get("filename"))
            mesh.set("filename", source.name)
            if source.name in reports:
                continue
            vertices, faces = read_stl(source)
            volume = mesh_volume_m3(vertices, faces) if is_closed(faces) else None
            out_vertices, out_faces = vertices, faces
            cell = cell_m
            while len(out_faces) > decimate_above:
                out_vertices, out_faces = cluster_decimate(vertices, faces, cell)
                if len(out_faces) <= MUJOCO_MAX_STL_FACES:
                    break
                cell *= 1.5
            write_stl(mesh_out / source.name, out_vertices, out_faces)
            reports[source.name] = MeshReport(
                source.name, len(faces), len(out_faces), mass, volume
            )
    return list(reports.values())


def exported_density(reports: list[MeshReport]) -> float | None:
    """Median mass / volume over links whose mesh is closed and not tiny."""

    densities = [
        r.mass_kg / r.volume_m3
        for r in reports
        if r.volume_m3 is not None and r.volume_m3 > 1e-6 and r.mass_kg > 0
    ]
    return float(np.median(densities)) if densities else None


def scale_inertials(robot: ET.Element, scale: float) -> None:
    for inertial in robot.iter("inertial"):
        mass = inertial.find("mass")
        if mass is not None:
            mass.set("value", repr(float(mass.get("value")) * scale))
        inertia = inertial.find("inertia")
        if inertia is not None:
            for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                if inertia.get(key) is not None:
                    inertia.set(key, repr(float(inertia.get(key)) * scale))


# ---------------------------------------------------------------- MJCF build


def _spec_body_moves(body) -> bool:
    while body is not None and body.name != "world":
        if body.joints:
            return True
        body = body.parent
    return False


def build_spec(urdf_file: Path):
    """Load the fixed URDF and add a collision geom for every moving mesh."""

    import mujoco

    spec = mujoco.MjSpec.from_file(str(urdf_file))
    spec.modelname = MODEL_NAME
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    for geom in list(spec.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = VISUAL_GROUP
        # The fixed palm/camera meshes get none: their convex hull would fill
        # the space the wearer's hand occupies.
        if not _spec_body_moves(geom.parent):
            continue
        collision = geom.parent.add_geom()
        collision.name = f"{geom.meshname}_collision"
        collision.type = mujoco.mjtGeom.mjGEOM_MESH
        collision.meshname = geom.meshname
        collision.pos = geom.pos
        collision.quat = geom.quat
        collision.group = COLLISION_GROUP
        collision.contype = HAND_CONTYPE
        collision.conaffinity = HAND_CONAFFINITY
        collision.rgba = [0.9, 0.2, 0.2, 0.25]
    return spec


def check_saved_model(expected, path: Path, tolerance_m: float = 1e-6) -> None:
    """Reload a written MJCF and confirm every geom sits where it was built."""

    import mujoco

    saved = mujoco.MjModel.from_xml_path(str(path))
    if (saved.ngeom, saved.nbody, saved.njnt) != (
        expected.ngeom, expected.nbody, expected.njnt
    ):
        raise RuntimeError(f"{path} does not reload to the model that was built")
    before, after = mujoco.MjData(expected), mujoco.MjData(saved)
    mujoco.mj_forward(expected, before)
    mujoco.mj_forward(saved, after)
    error = float(np.abs(before.geom_xpos - after.geom_xpos).max(initial=0.0))
    if error > tolerance_m:
        raise RuntimeError(f"{path} moves geoms by up to {error * 1000:.2f} mm on reload")


def joint_report(model, data) -> list[str]:
    """Describe the hinge chains: three-joint fingers and a four-joint thumb."""

    import mujoco

    def name(kind, index):
        return mujoco.mj_id2name(model, kind, index)

    def child_joints(body):
        return [
            b for b in range(1, model.nbody)
            if model.body_parentid[b] == body and model.body_jntnum[b] > 0
        ]

    chains: list[list[int]] = []
    for joint in range(model.njnt):
        parent = model.body_parentid[model.jnt_bodyid[joint]]
        if model.body_jntnum[parent] > 0:
            continue
        chain, body = [joint], model.jnt_bodyid[joint]
        while len(children := child_joints(body)) == 1:
            body = children[0]
            chain.append(int(model.body_jntadr[body]))
        chains.append(chain)
    lines = [
        f"hinge joints: {model.njnt}; chains from the palm: "
        f"{sorted(len(c) for c in chains)} (TriMI motors: 3 + 3 + 3 + 4)"
    ]
    thumbs = [c for c in chains if len(c) == 4]
    fingers = [c for c in chains if len(c) == 3]
    if len(thumbs) == 1 and len(fingers) == 3:
        thumb_base = data.xanchor[thumbs[0][0]]
        fingers.sort(key=lambda c: np.linalg.norm(data.xanchor[c[0]] - thumb_base))
        for label, chain in zip(("index", "middle", "ring", "thumb"), fingers + thumbs):
            joints = ", ".join(name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in chain)
            lines.append(f"  {label:6s} (palm -> tip): {joints}")
    for joint in range(model.njnt):
        body = model.jnt_bodyid[joint]
        axis = data.xmat[body].reshape(3, 3) @ model.jnt_axis[joint]
        lines.append(
            f"    {name(mujoco.mjtObj.mjOBJ_JOINT, joint):14s} "
            f"link={name(mujoco.mjtObj.mjOBJ_BODY, body):24s} "
            f"anchor_mm={np.round(data.xanchor[joint] * 1000, 1)} axis={np.round(axis, 3)} "
            f"limited={'yes' if model.jnt_limited[joint] else 'no'}"
        )
    return lines


# ---------------------------------------------------------------- main


def build(args: argparse.Namespace) -> int:
    import mujoco

    out: Path = args.out.expanduser()
    if out.exists() and args.clean:
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        urdf_path = locate_urdf(args.source, Path(tmp))
        tree = ET.parse(urdf_path)
        robot = tree.getroot()
        if robot.tag != "robot":
            raise ValueError(f"{urdf_path} is not a URDF <robot>")
        reports = convert_meshes(
            robot, urdf_path, out / "meshes",
            decimate_above=args.decimate_above, cell_m=args.cluster_cell_mm / 1000.0,
        )
        print(f"URDF: {urdf_path.name} ({robot.get('name')})")
        print("\nMeshes:")
        for report in sorted(reports, key=lambda r: -r.faces_in):
            note = "" if report.faces_in == report.faces_out else f" -> {report.faces_out}"
            over = " (over MuJoCo's STL limit)" if report.faces_in > MUJOCO_MAX_STL_FACES else ""
            print(f"  {report.name:34s} {report.faces_in:7d} faces{note}{over}")

        density = exported_density(reports)
        if args.mass_scale is not None:
            scale = args.mass_scale
        elif density is not None and density < UNASSIGNED_DENSITY_MAX_KG_M3:
            scale = args.density / density
        else:
            scale = 1.0
        print("\nMass:")
        if density is not None:
            print(f"  density implied by exported mass / closed-mesh volume: {density:.3g} kg/m^3")
        print(f"  mass and inertia scale applied: {scale:.4g}")
        if scale != 1.0:
            scale_inertials(robot, scale)
        if robot.find("link/collision") is None:
            print("\nCollision: the URDF has no <collision> elements; using link meshes.")

        # fusestatic stays off: MjSpec.to_xml() drops the frames of fused
        # fixed links (the palm's 180 degree yaw and the thumb-tip part).
        mujoco_element = ET.Element("mujoco")
        mujoco_element.append(ET.Element("compiler", {
            "meshdir": "meshes", "discardvisual": "false",
            "fusestatic": "false", "balanceinertia": "true",
        }))
        robot.insert(0, mujoco_element)
        urdf_out = out / f"{MODEL_NAME}.urdf"
        tree.write(urdf_out, encoding="unicode")

    spec = build_spec(urdf_out)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("\nJoints:")
    for line in joint_report(model, data):
        print(f"  {line}")
    xml_out = out / f"{MODEL_NAME}.xml"
    xml_out.write_text(spec.to_xml())
    check_saved_model(model, xml_out)
    print(f"\nWrote {xml_out}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="URDF zip, its folder, or the .urdf file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output folder (default: %(default)s)")
    parser.add_argument("--clean", action="store_true", help="delete --out first")
    parser.add_argument("--decimate-above", type=int, default=DEFAULT_DECIMATE_ABOVE_FACES,
                        help="decimate meshes with more faces than this (default: %(default)s)")
    parser.add_argument("--cluster-cell-mm", type=float, default=DEFAULT_CLUSTER_CELL_MM,
                        help="vertex-clustering cell size (default: %(default)s mm)")
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--density", type=float, default=DEFAULT_DENSITY_KG_M3,
                             help="density to assume when the export has none "
                                  "(default: %(default)s kg/m^3)")
    scale_group.add_argument("--mass-scale", type=float,
                             help="multiply masses and inertias by this instead")
    args = parser.parse_args(argv)
    if not args.cluster_cell_mm > 0 or not args.density > 0 or args.decimate_above < 1:
        parser.error("--cluster-cell-mm, --density and --decimate-above must be positive")
    if args.mass_scale is not None and not args.mass_scale > 0:
        parser.error("--mass-scale must be positive")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(build(parse_args()))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
