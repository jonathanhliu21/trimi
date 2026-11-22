import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mujoco  # noqa: F401
except ImportError:  # the check needs MuJoCo; hardware-only installs skip these tests
    mujoco = None

MIDAS = Path("~/workspace/midas-piper-control").expanduser()


@unittest.skipIf(mujoco is None, "mujoco not installed")
class GravCompClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.grav_comp import grav_comp_clearance as G
        cls.G = G
        try:
            cls.scene = G.build_scene(0.75, 0.49)
        except FileNotFoundError as exc:  # no trimorph-piper-control workspace sim on this machine
            raise unittest.SkipTest(str(exc))
        cls.check = G.Checker(cls.scene)

    def vertices(self, geoms):
        """World mesh vertices of geoms at the pose last set."""
        out = []
        for g in geoms:
            verts, _ = self.G.mesh_arrays(self.scene.model, g)
            pos, rot = self.check._frame(g)
            out.append(pos + verts @ rot.T)
        return np.vstack(out)

    def test_desk_distance_matches_the_mesh_vertices(self):
        # Arm hanging toward the desk: MuJoCo 3.5's mj_geomDistance returns 0 for link5 here.
        c = self.check(np.array([-1.57, 1.6, -0.3, 0.0, 0.0, 0.0]))
        pts = self.vertices(self.check.front)
        lo, hi = np.array([0.021, -0.058, 0.0]), np.array([0.521, 0.918, 0.75])
        exact = np.linalg.norm(np.maximum(np.maximum(lo - pts, pts - hi), 0.0), axis=1).min()
        self.assertLess(exact, 0.25)
        self.assertLessEqual(c.env["desk"], exact + 1e-9)       # face samples can only come closer
        self.assertGreater(c.env["desk"], exact - 0.005)

    def test_face_is_the_furthest_point_behind_the_mounting_face(self):
        # Config 2 puts the J1 axis along world +x, so the mounting face is x = base x.
        for q, sign in (([0.0, 1.2, -1.2, 0.0, 0.0, 0.0], 1), ([0.0, 2.63, -0.57, 0.0, 0.0, 0.0], -1)):
            c = self.check(np.array(q))
            behind = (self.vertices(self.check.front)[:, 0] - self.scene.base_pos[0]).min()
            self.assertAlmostEqual(c.face, behind, delta=1e-9)
            self.assertEqual(np.sign(c.face), sign)

    def test_video_timeline_follows_the_capture_motion(self):
        from scripts.grav_comp import grav_comp_video as V
        start, q = np.zeros(6), np.array([0.3, 1.0, -0.8, 0.0, 0.2, 0.0])
        below, above = q - 0.1, q + 0.1
        it = self.G.Item("S2 gravity", "pose", q, [start, below, q, above, q], kind="capture")
        moves = [s for s in V.segments(it) if not np.array_equal(s.q0, s.q1)]
        self.assertEqual(len(moves), 4)
        for s, (a, b) in zip(moves, [(start, below), (below, q), (q, above), (above, q)]):
            np.testing.assert_allclose(s.q0, a)
            np.testing.assert_allclose(s.q1, b)
        holds = 2 * (V.STAGING_SETTLE_S + V.CAPTURE_SETTLE_S + V.CAPTURE_AVG_S)
        frames = V.timeline([it], speed=4.0, fps=30.0)
        self.assertAlmostEqual(frames[-1].t, (0.9 + 0.1) / V.SLEW_RAD_S + 0.2 / V.APPROACH_RAD_S + holds)
        np.testing.assert_allclose(frames[0].q, start)
        np.testing.assert_allclose(frames[-1].q, q)
        step = np.abs(np.diff([f.q for f in frames], axis=0)).max()
        self.assertLessEqual(step, V.SLEW_RAD_S * 4.0 / 30.0 + 1e-9)  # never faster than a slew
        self.assertEqual(V.playback([it], 4.0, 30.0), 1.0)           # 8 s: real time
        self.assertEqual(V.playback([it] * 20, 4.0, 30.0), 4.0)      # 159 s: x4 still lasts 40 s

    @unittest.skipUnless((MIDAS / "gravity_comp" / "poses.py").exists(), "midas-piper-control checkout missing")
    def test_stage_motions_follow_gravity_comp(self):
        gp, vd, mf, _, _ = self.G.load_gravity_comp(MIDAS)
        start = np.radians([10.0, 20.0, -30.0, 0.0, 0.0, 0.0])
        stages = self.G.stage_items(gp, vd, mf, start, 0.10, 0.30)
        self.assertEqual(len(stages["S2 gravity"]), len(gp.gravity_poses()) + 1)
        for name in ("S1 feedback", "S2 gravity", "S4a payload"):
            np.testing.assert_allclose(stages[name][0].path[0], start)    # slews out of the start pose
            np.testing.assert_allclose(stages[name][-1].path[-1], start)  # and back
        for it in stages["S5 friction"]:
            swing = np.abs(np.asarray(it.path[1:]) - it.q)
            self.assertAlmostEqual(swing[:, it.probe].max(), 0.30)
            self.assertEqual(np.delete(swing, it.probe, axis=1).max(), 0.0)
        # --viscous: measure_friction sweeps each joint right after its breakaway poses, in one run
        s5 = self.G.stage_items(gp, vd, mf, start, 0.10, 0.30, viscous=True)["S5 friction+viscous"]
        self.assertEqual([it.kind for it in s5],
                         sum((["probe"] * len(gp.friction_poses(j)) + ["sweep"] for j in range(6)), []))
        np.testing.assert_allclose(s5[0].path[0], start)
        for a, b in zip(s5, s5[1:]):
            np.testing.assert_allclose(b.path[0], a.path[-1])  # one continuous motion


if __name__ == "__main__":
    unittest.main()
