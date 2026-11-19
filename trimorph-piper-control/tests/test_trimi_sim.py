import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - optional simulator dependency
    mujoco = None

from scripts.sim import build_trimi_sim_model as builder
from scripts.sim.trimi_mirror import (
    HandMirror,
    HomingStop,
    JointMapEntry,
    StopCheck,
    find_hand_port,
    load_joint_map,
    mirror_loop,
    parse_joint_map,
    write_zeros,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILT_MODEL = REPO_ROOT / "assets/trimi_v2/trimi_v2.xml"
JOINT_MAP = REPO_ROOT / "config/trimi_v2_joint_map.yaml"


def box_mesh(half=(0.01, 0.02, 0.005), center=(0.0, 0.0, 0.0)):
    corners = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float
    ) * half + center
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return corners, faces


# A root rotated 180 degrees about z, a fixed palm with an offset, and a finger
# whose tip part hangs off it through another fixed joint. Masses use the
# 1 kg/m^3 density of an Onshape export without materials.
SYNTHETIC_URDF = """<robot name="toy">
  <link name="root"/>
  <link name="palm">
    <inertial><mass value="3.2e-05"/><origin xyz="0 0 0"/>
      <inertia ixx="1e-13" ixy="0" ixz="0" iyy="1e-13" iyz="0" izz="1e-13"/></inertial>
    <visual><origin xyz="0 0 0"/><geometry><mesh filename="package://toy/meshes/palm.stl"/></geometry></visual>
  </link>
  <link name="finger">
    <inertial><mass value="4e-06"/><origin xyz="0 0 0"/>
      <inertia ixx="1e-13" ixy="0" ixz="0" iyy="1e-13" iyz="0" izz="1e-13"/></inertial>
    <visual><origin xyz="0 -0.02 0"/><geometry><mesh filename="package://toy/meshes/finger.stl"/></geometry></visual>
  </link>
  <link name="tip">
    <inertial><mass value="1e-06"/><origin xyz="0 0 0"/>
      <inertia ixx="1e-13" ixy="0" ixz="0" iyy="1e-13" iyz="0" izz="1e-13"/></inertial>
    <visual><origin xyz="0 0 0"/><geometry><mesh filename="package://toy/meshes/tip.stl"/></geometry></visual>
  </link>
  <joint name="fixed_palm" type="fixed"><origin xyz="0 0 0" rpy="0 0 3.14159265"/>
    <parent link="root"/><child link="palm"/></joint>
  <joint name="flex" type="continuous"><origin xyz="0 -0.03 0" rpy="0 0 0"/><axis xyz="-1 0 0"/>
    <parent link="palm"/><child link="finger"/></joint>
  <joint name="fixed_tip" type="fixed"><origin xyz="0 -0.05 0" rpy="0 0 0"/>
    <parent link="finger"/><child link="tip"/></joint>
</robot>
"""


class StlAndDecimationTests(unittest.TestCase):
    def test_binary_stl_round_trip_welds_vertices(self):
        vertices, faces = box_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.stl"
            builder.write_stl(path, vertices, faces)
            read_vertices, read_faces = builder.read_stl(path)
        self.assertEqual(read_vertices.shape, (8, 3))
        self.assertEqual(read_faces.shape, (12, 3))
        self.assertTrue(builder.is_closed(read_faces))
        self.assertAlmostEqual(
            builder.mesh_volume_m3(read_vertices, read_faces), 0.02 * 0.04 * 0.01, places=9
        )

    def test_ascii_stl_is_read(self):
        text = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n" \
               "vertex 0 1 0\nendloop\nendfacet\nendsolid t\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ascii.stl"
            path.write_text(text)
            vertices, faces = builder.read_stl(path)
        self.assertEqual((vertices.shape, faces.shape), ((3, 3), (1, 3)))

    def test_cluster_decimation_reduces_faces_and_keeps_extent(self):
        grid = np.linspace(0.0, 0.05, 101)
        xx, yy = np.meshgrid(grid, grid)
        vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        index = np.arange(xx.size).reshape(xx.shape)
        a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
        c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
        faces = np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
        out_vertices, out_faces = builder.cluster_decimate(vertices, faces, 0.002)
        self.assertLess(len(out_faces), len(faces) / 10)
        np.testing.assert_allclose(out_vertices.min(0)[:2], [0.0, 0.0], atol=0.001)
        np.testing.assert_allclose(out_vertices.max(0)[:2], [0.05, 0.05], atol=0.001)

    def test_exported_density_ignores_open_and_tiny_meshes(self):
        reports = [
            builder.MeshReport("a", 10, 10, 2e-5, 2e-5),
            builder.MeshReport("b", 10, 10, 3e-5, 3e-5),
            builder.MeshReport("open", 10, 10, 5.0, None),
            builder.MeshReport("tiny", 10, 10, 5.0, 1e-9),
        ]
        self.assertAlmostEqual(builder.exported_density(reports), 1.0)


@unittest.skipIf(mujoco is None, "mujoco is not installed")
class BuildModelTests(unittest.TestCase):
    def build_toy(self, tmp: Path) -> Path:
        source = tmp / "toy"
        (source / "urdf").mkdir(parents=True)
        (source / "meshes").mkdir()
        (source / "urdf/toy.urdf").write_text(SYNTHETIC_URDF)
        builder.write_stl(source / "meshes/palm.stl", *box_mesh((0.02, 0.02, 0.01)))
        builder.write_stl(source / "meshes/finger.stl", *box_mesh((0.005, 0.02, 0.005)))
        builder.write_stl(source / "meshes/tip.stl", *box_mesh((0.005, 0.005, 0.005)))
        out = tmp / "out"
        args = builder.parse_args([str(source), "--out", str(out)])
        self.assertEqual(builder.build(args), 0)
        return out

    def test_toy_export_builds_with_fixed_frames_masses_and_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.build_toy(Path(tmp))
            model = mujoco.MjModel.from_xml_path(str(out / "trimi_v2.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
        # The 180 degree root yaw puts the finger on +y; the tip keeps its offset.
        np.testing.assert_allclose(data.xpos[tip], [0.0, 0.08, 0.0], atol=1e-6)
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger")
        self.assertAlmostEqual(model.body_mass[finger], 4e-3, places=6)
        collision = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            for g in range(model.ngeom)
            if model.geom_contype[g] == builder.HAND_CONTYPE
        ]
        self.assertEqual(sorted(collision), ["finger", "tip"])
        self.assertEqual(model.nu, 0)

    def test_saved_model_check_rejects_moved_geoms(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.build_toy(Path(tmp))
            path = out / "trimi_v2.xml"
            model = mujoco.MjModel.from_xml_path(str(path))
            text = path.read_text().replace('pos="0 -0.05 0"', 'pos="0 -0.06 0"', 1)
            self.assertNotEqual(text, path.read_text())
            path.write_text(text)
            with self.assertRaisesRegex(RuntimeError, "moves geoms"):
                builder.check_saved_model(model, path)

    def test_zip_member_outside_the_folder_is_rejected(self):
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.urdf", "<robot/>")
            work = Path(tmp) / "work"
            work.mkdir()
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                builder.locate_urdf(archive, work)


class JointMapTests(unittest.TestCase):
    def test_checked_in_map_covers_all_motors_once(self):
        entries = load_joint_map(JOINT_MAP)
        self.assertEqual(sorted(e.motor_id for e in entries), list(range(13)))
        self.assertEqual(len({e.urdf_joint for e in entries}), 13)

    def test_invalid_entries_are_rejected(self):
        cases = {
            "sign must be": {0: {"urdf_joint": "a", "sign": 0.5}},
            "scale must be": {0: {"urdf_joint": "a", "scale": 0}},
            "zero must be": {0: {"urdf_joint": "a", "zero": float("nan")}},
            "no longer used": {0: {"urdf_joint": "a", "offset": 0.1}},
            "more than once": {0: {"urdf_joint": "a"}, 1: {"urdf_joint": "a"}},
            "urdf_joint": {0: {"sign": 1}},
        }
        for message, joints in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_joint_map({"joints": joints})
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_joint_map({})

    def test_writing_zeros_keeps_comments_signs_and_other_motors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.yaml"
            path.write_text(JOINT_MAP.read_text())
            write_zeros(path, {3: -1.23456, 12: 0.5})
            text = path.read_text()
            entries = {e.motor_id: e for e in load_joint_map(path)}
        original = {e.motor_id: e for e in load_joint_map(JOINT_MAP)}
        self.assertIn("q_urdf = sign * scale", text)
        self.assertAlmostEqual(entries[3].zero, -1.2346)
        self.assertAlmostEqual(entries[12].zero, 0.5)
        for motor_id, entry in original.items():
            if motor_id not in (3, 12):
                self.assertEqual(entries[motor_id], entry)
            self.assertEqual(entries[motor_id].sign, entry.sign)

    def test_writing_zeros_for_an_unlisted_motor_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.yaml"
            path.write_text(JOINT_MAP.read_text())
            with self.assertRaisesRegex(ValueError, r"motors \[40\].*40: 1.0000"):
                write_zeros(path, {0: 0.3, 40: 1.0})
            self.assertEqual(path.read_text(), JOINT_MAP.read_text())

    def test_hand_port_is_the_only_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "found none"):
                find_hand_port(directory)
            (directory / "usb-FTDI-if00").touch()
            self.assertEqual(find_hand_port(directory), str(directory / "usb-FTDI-if00"))
            (directory / "usb-other-if00").touch()
            with self.assertRaisesRegex(RuntimeError, "usb-FTDI-if00, usb-other-if00"):
                find_hand_port(directory)


TWO_JOINT_XML = """<mujoco><worldbody>
  <body name="a"><joint name="ja" axis="1 0 0"/><geom size="0.01"/>
    <body name="b" pos="0 0.05 0"><joint name="jb" axis="1 0 0"/><geom size="0.01"/></body>
  </body>
  <body name="free_body"><freejoint name="free"/><geom size="0.01"/></body>
</worldbody></mujoco>"""


@unittest.skipIf(mujoco is None, "mujoco is not installed")
class HandMirrorTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(TWO_JOINT_XML)
        self.data = mujoco.MjData(self.model)
        self.entries = (
            JointMapEntry(7, "jb", "B", -1.0, 2.0, 0.1),
            JointMapEntry(3, "ja", "A", 1.0, 1.0, 0.0),
        )

    def test_readings_in_hand_order_land_on_mapped_joints(self):
        mirror = HandMirror(self.model, self.entries, motor_ids=[3, 7])
        urdf = mirror.apply(self.data, [0.25, 0.5])
        # q = sign * scale * (reading - zero)
        np.testing.assert_allclose(urdf, [0.25, -2.0 * (0.5 - 0.1)])
        self.assertAlmostEqual(self.data.qpos[self.model.jnt_qposadr[0]], 0.25)
        self.assertAlmostEqual(self.data.qpos[self.model.jnt_qposadr[1]], -0.8)

    def test_stop_check_reports_how_far_past_the_homing_stop_each_joint_went(self):
        mirror = HandMirror(self.model, self.entries, motor_ids=[3, 7])
        stops = {3: HomingStop(0.5, 1), 7: HomingStop(-1.0, -1)}
        check = StopCheck(mirror, stops)
        for positions in ([0.4, -0.9], [0.52, -0.95], [0.3, -0.5]):
            check.update(positions)
        # motor 3 went 0.02 past its stop, motor 7 stayed 0.05 short of it
        zeros = check.zeros()
        self.assertAlmostEqual(zeros[3], 0.02)
        self.assertAlmostEqual(zeros[7], 0.15)
        # the new zero maps the hand-pushed stop where homing's stop maps now
        self.assertAlmostEqual(-2.0 * (-0.95 - zeros[7]), -2.0 * (-1.0 - 0.1))
        report = check.report()
        self.assertIn("1.1 deg past", report)
        self.assertIn("5.7 deg short of", report)

        untouched = StopCheck(mirror, stops)
        untouched.update([0.0, -0.95])
        self.assertEqual(list(untouched.zeros()), [7])
        self.assertIn("not pushed to its stop", untouched.report())
        with self.assertRaisesRegex(ValueError, r"\[7\]"):
            StopCheck(mirror, {3: HomingStop(0.5, 1)})

    def test_missing_motor_unknown_joint_and_bad_readings_fail(self):
        with self.assertRaisesRegex(ValueError, r"motor IDs \[9\]"):
            HandMirror(self.model, self.entries, motor_ids=[3, 9])
        with self.assertRaisesRegex(ValueError, "no joint 'nope'"):
            HandMirror(self.model, (JointMapEntry(3, "nope", "x", 1, 1, 0),), [3])
        with self.assertRaisesRegex(ValueError, "not a hinge"):
            HandMirror(self.model, (JointMapEntry(3, "free", "x", 1, 1, 0),), [3])
        mirror = HandMirror(self.model, self.entries, motor_ids=[3, 7])
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            mirror.to_urdf([0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            mirror.to_urdf([0.0, float("inf")])

    def test_loop_copies_each_reading_until_the_viewer_closes(self):
        import contextlib

        mirror = HandMirror(self.model, self.entries, motor_ids=[3, 7])
        readings = iter([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])

        class Hand:
            def read_pos(self):
                return next(readings)

        class Viewer:
            frames = 0

            def is_running(self):
                return self.frames < 3

            def lock(self):
                return contextlib.nullcontext()

            def sync(self):
                self.frames += 1

        sleeps = []
        frames = mirror_loop(
            Hand(), self.model, self.data, mirror, Viewer(),
            rate_hz=10.0, clock=lambda: 0.0, sleep=sleeps.append,
        )
        self.assertEqual(frames, 3)
        self.assertAlmostEqual(self.data.qpos[self.model.jnt_qposadr[0]], 0.3)
        np.testing.assert_allclose(sleeps, [0.1, 0.2, 0.3])


@unittest.skipIf(mujoco is None or not BUILT_MODEL.is_file(), "TriMI model not built")
class BuiltTrimiModelTests(unittest.TestCase):
    """Checks against assets/trimi_v2, built from the TRIMI_URDF_v2 export."""

    def test_joint_map_matches_the_urdf_chains(self):
        model = mujoco.MjModel.from_xml_path(str(BUILT_MODEL))
        mirror = HandMirror(model, load_joint_map(JOINT_MAP), list(range(13)))
        joint_of = {e.motor_id: e.urdf_joint for e in mirror.entries}

        def parent_joint(name):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            parent = model.body_parentid[model.jnt_bodyid[joint]]
            if not model.body_jntnum[parent]:
                return None
            return mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, model.body_jntadr[parent]
            )

        # Motors are numbered tip -> palm: (PIP, MCP, ABAD) and thumb (0, 1, 2, 3).
        for chain in ((4, 5, 6), (7, 8, 9), (10, 11, 12), (0, 1, 2, 3)):
            for distal, proximal in zip(chain, chain[1:]):
                self.assertEqual(parent_joint(joint_of[distal]), joint_of[proximal])
            self.assertIsNone(parent_joint(joint_of[chain[-1]]))

    def test_leader_flexion_curls_fingers_toward_the_palm_side(self):
        model = mujoco.MjModel.from_xml_path(str(BUILT_MODEL))
        data = mujoco.MjData(model)
        mirror = HandMirror(model, load_joint_map(JOINT_MAP), list(range(13)))
        tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("part_1", "_pip_link_l_v2_right", "_pip_link_l_v2_mirror")]
        mirror.apply(data, np.zeros(13))
        mujoco.mj_kinematics(model, data)
        open_z = data.xipos[tips, 2].copy()
        flexed = np.zeros(13)
        flexed[[4, 5, 7, 8, 10, 11]] = -1.5  # leader flexion is negative
        mirror.apply(data, flexed)
        mujoco.mj_kinematics(model, data)
        self.assertTrue(np.all(data.xipos[tips, 2] < open_z - 0.01))

    def test_geared_joints_turn_34_84_of_the_motor(self):
        model = mujoco.MjModel.from_xml_path(str(BUILT_MODEL))
        mirror = HandMirror(model, load_joint_map(JOINT_MAP), list(range(13)))
        # one radian of motor travel past each zero, in the leader's flexion direction
        delta = mirror.to_urdf(mirror.zero - 1.0) - mirror.to_urdf(mirror.zero)
        urdf = dict(zip(range(13), delta))
        for motor_id in (4, 5, 7, 8, 10, 11):
            self.assertAlmostEqual(urdf[motor_id], 34 / 84, places=4)
        for motor_id in (6, 9, 12):
            self.assertAlmostEqual(urdf[motor_id], 1.0)


if __name__ == "__main__":
    unittest.main()


from scripts.sim.sync_hard_stops import (  # noqa: E402
    HardStopFinder,
    find_stops,
    homing_stops,
    zero_from_stop,
)


class HomingStopTests(unittest.TestCase):
    def test_reading_at_stop_follows_home_offset_definition(self):
        hand = {
            "joint_signs": {0: 1.0, 4: -1.0},
            "homing": {
                "thumb": [[0, "Thumb joint 0", -0.03, 1]],
                "fingers": [[4, "Index PIP", 0.03, -1]],
            },
        }
        stops = homing_stops(hand)
        # read = (raw - (raw_stop + cad)) * sign, so at the stop read = -sign * cad
        self.assertAlmostEqual(stops[0].reading, 0.03)
        self.assertEqual(stops[0].direction, 1)
        self.assertAlmostEqual(stops[4].reading, 0.03)
        self.assertEqual(stops[4].direction, 1)

    def test_zero_puts_the_stop_reading_at_the_urdf_stop(self):
        zero = zero_from_stop(reading=1.57, q_stop=-1.5, sign=-1.0, scale=1.0)
        self.assertAlmostEqual(-1.0 * (1.57 - zero), -1.5)
        zero = zero_from_stop(reading=0.05, q_stop=-0.01, sign=-1.0, scale=34 / 84)
        self.assertAlmostEqual(-(34 / 84) * (0.05 - zero), -0.01)


def write_box(path, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    vertices, faces = box_mesh((hi - lo) / 2, (hi + lo) / 2)
    builder.write_stl(path, vertices, faces)


# A link swinging about z over a palm, with a tab fixed to it and a flap on a
# joint of its own. A wall at 30 degrees stops the tab: its outer corner
# (50 mm out, 0.2 mm short of the wall's face) touches at 30.23 degrees, the
# arm's own tip at 30.82; the flap would hit at 24.6 but is not carried. A
# plate on the palm overlaps the arm's top by 0.5 mm in the starting pose and
# only slides along it. A second finger holds a decoy wall at 12 degrees.
STOP_XML = """<mujoco>
  <compiler meshdir="{meshdir}"/>
  <asset>
    <mesh name="palm" file="palm.stl"/><mesh name="wall" file="wall.stl"/>
    <mesh name="arm" file="arm.stl"/><mesh name="decoy" file="decoy.stl"/>
    <mesh name="plate" file="plate.stl"/><mesh name="tab" file="tab.stl"/>
    <mesh name="flap" file="flap.stl"/>
  </asset>
  <worldbody>
    <body name="palm"><geom type="mesh" mesh="palm" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="plate" contype="0" conaffinity="0"/>
      <body name="wall" euler="0 0 30"><geom type="mesh" mesh="wall" contype="0" conaffinity="0"/></body>
      <body name="arm"><joint name="swing" axis="0 0 1"/>
        <geom type="mesh" mesh="arm" contype="0" conaffinity="0"/>
        <body name="tab"><geom type="mesh" mesh="tab" contype="0" conaffinity="0"/></body>
        <body name="flap"><joint name="flap_joint" axis="0 0 1"/>
          <geom type="mesh" mesh="flap" contype="0" conaffinity="0"/></body>
      </body>
      <body name="other" pos="0 0 0.05"><joint name="other_joint" axis="0 0 1"/>
        <body name="decoy_holder" pos="0 0 -0.05" euler="0 0 12">
          <geom type="mesh" mesh="decoy" contype="0" conaffinity="0"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


@unittest.skipIf(mujoco is None, "mujoco is not installed")
class HardStopFinderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        meshes = Path(self.tmp.name)
        write_box(meshes / "palm.stl", (-0.03, -0.03, -0.02), (0.0, 0.03, -0.01))
        write_box(meshes / "arm.stl", (0.015, -0.004, -0.004), (0.07, 0.004, 0.004))
        write_box(meshes / "tab.stl", (0.03, 0.004, -0.003), (0.05, 0.0048, 0.003))
        write_box(meshes / "flap.stl", (0.02, 0.004, -0.002), (0.075, 0.012, 0.002))
        write_box(meshes / "plate.stl", (0.02, -0.002, 0.0035), (0.06, 0.03, 0.006))
        # walls: thin radial plates lying beyond the arm's leading face
        write_box(meshes / "wall.stl", (0.02, 0.005, -0.006), (0.08, 0.009, 0.006))
        write_box(meshes / "decoy.stl", (0.02, 0.005, -0.006), (0.08, 0.009, 0.006))
        self.model = mujoco.MjModel.from_xml_string(STOP_XML.format(meshdir=meshes))
        self.finder = HardStopFinder(self.model, meshes)

    def tearDown(self):
        self.tmp.cleanup()

    def test_carried_parts_stop_at_first_touch_past_a_sliding_cad_overlap(self):
        sweep = self.finder.find("swing", +1, geared=False)
        self.assertEqual(sweep.obstacles, ("wall",))
        self.assertEqual(sweep.movers, ("tab",))
        self.assertAlmostEqual(np.degrees(sweep.q_stop), 30.23, delta=0.3)
        self.assertGreater(sweep.contact_radius_m, 0.04)

    def test_a_finger_already_homed_is_in_the_way(self):
        swing, other = self.model.joint("swing").id, self.model.joint("other_joint").id
        self.assertNotIn(self.model.body("decoy_holder").id, self.finder.homing_obstacles(swing, set()))
        sweep = self.finder.find("swing", +1, geared=False,
                                 obstacles=self.finder.homing_obstacles(swing, {other}))
        self.assertEqual(sweep.obstacles, ("decoy",))
        self.assertAlmostEqual(np.degrees(sweep.q_stop), 12.23, delta=0.3)


@unittest.skipIf(mujoco is None or not BUILT_MODEL.is_file(), "TriMI model not built")
class BuiltModelHardStopTests(unittest.TestCase):
    """Stops the real hand confirms, found on the built model's meshes."""

    def test_homing_stops(self):
        model = mujoco.MjModel.from_xml_path(str(BUILT_MODEL))
        hand = {
            "joint_signs": {0: 1.0, 9: 1.0, 12: 1.0},
            "homing": {"thumb": [[0, "Thumb joint 0", -0.03, 1]],
                       "fingers": [[9, "Middle ABAD", 0.0, 1], [12, "Ring ABAD", -1.57, 1]]},
        }
        results = {r.motor_id: r for r in find_stops(
            model, BUILT_MODEL.parent / "meshes", load_joint_map(JOINT_MAP), hand, {0, 9, 12})}
        # Thumb IP: its cover is modelled 0.5 mm into the motor case at 0 and only
        # slides along it; the stop is where the cover's other face meets the case.
        self.assertAlmostEqual(np.degrees(results[0].q_stop), 2.4, delta=1.0)
        # Ring: its ABAD link meets the palm near -90 deg (the proximal link would
        # cross the palm's thin hand band at -75, which the real hand passes).
        self.assertAlmostEqual(np.degrees(results[12].q_stop), -89.3, delta=1.0)
        self.assertEqual(results[12].movers, ("_mcp_link_m_mirror",))
        # Middle: the palm stop near -25 deg.
        self.assertAlmostEqual(np.degrees(results[9].q_stop), -24.9, delta=1.5)
        self.assertEqual(results[9].obstacles, ("DC15_A01_CASE_M_DUMMY",))
