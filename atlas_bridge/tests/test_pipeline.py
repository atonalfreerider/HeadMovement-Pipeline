"""Run: <mamma env python> -m unittest discover -s tests -v   (from atlas_bridge/)"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
MAMMA = HERE.parents[1] / "mamma"

from atlas_bridge import align, cameras, sync, unity_export  # noqa: E402
from atlas_bridge.beats import zouk_grid  # noqa: E402

RIG = {"width": 1920, "height": 1080}  # session defaults for everything else


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.cams = cameras.build_rig(RIG)

    def test_rotations_are_proper(self):
        for c in self.cams:
            np.testing.assert_allclose(c.R_wc.T @ c.R_wc, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(c.R_wc), 1.0, places=12)

    def test_target_projects_to_principal_point_and_up_is_up(self):
        for c in self.cams:
            uv = c.project(np.array([[0, 0, 1.0], [0, 0, 1.5]]))
            np.testing.assert_allclose(uv[0], [c.cx, c.cy], atol=1e-6)
            self.assertLess(uv[1, 1], uv[0, 1], "higher point must be higher in the image (smaller v)")

    def test_coverage_default_rig(self):
        self.assertEqual(cameras.check_coverage(self.cams, stage_radius=1.2), [])

    def test_unity_conversion_is_not_a_mirror(self):
        self.assertAlmostEqual(np.linalg.det(cameras.STAGE_TO_UNITY), -1.0)  # handedness flip, by design
        p = cameras.stage_to_unity(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(p, [1.0, 3.0, 2.0])

    def test_pose_in_scaled_frame_stays_rigid(self):
        M = np.eye(4)
        M[:3, :3] = 2.5 * align.rotation_between(np.array([0, 0, 1.0]), np.array([0, 1.0, 0]))
        M[:3, 3] = [1, 2, 3]
        T = cameras.pose_in_frame(M, self.cams[0].R_wc, self.cams[0].C)
        np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], M[:3, :3] @ self.cams[0].C + M[:3, 3])

    @unittest.skipUnless((MAMMA / "capture" / "calibration.py").exists(), "mamma repo not found")
    def test_mamma_loader_contract(self):
        sys.path.insert(0, str(MAMMA))
        from capture.calibration import detect_up_axis, load_calibration

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calib.yaml"
            p.write_text(cameras.to_mamma_yaml(self.cams))
            calib = load_calibration(p)
        X = np.array([[0.3, -0.2, 1.1], [-0.5, 0.4, 0.2], [0.0, 0.0, 1.8]])
        for c in self.cams:
            mc = calib.cameras[c.name]
            Xh = np.c_[X, np.ones(len(X))]
            pc = (mc.T_cam_world @ Xh.T).T[:, :3]
            uv = (mc.intrinsics @ pc.T).T
            uv = uv[:, :2] / uv[:, 2:]
            np.testing.assert_allclose(uv, c.project(X), atol=1e-4)
            self.assertEqual((mc.width, mc.height), (c.width, c.height))
        up, conf = detect_up_axis(calib)
        self.assertEqual(up, "z")


class SyncTests(unittest.TestCase):
    def test_gcc_phat_sign_and_accuracy(self):
        rng = np.random.default_rng(0)
        ref = rng.standard_normal(sync.SR * 20).astype(np.float32)
        start = 3.2371
        sig = ref[int(start * sync.SR): int(start * sync.SR) + sync.SR * 8]
        sig = sig + 0.3 * rng.standard_normal(len(sig)).astype(np.float32)
        tau, conf = sync.gcc_phat(sig, ref)
        self.assertAlmostEqual(tau, start, delta=1.0 / sync.SR)
        self.assertGreater(conf, 5)

    def test_gcc_phat_negative_offset(self):
        rng = np.random.default_rng(1)
        full = rng.standard_normal(sync.SR * 20).astype(np.float32)
        ref = full[sync.SR * 2:]  # reference starts 2 s after the other source
        tau, _ = sync.gcc_phat(full[: sync.SR * 10], ref)
        self.assertAlmostEqual(tau, -2.0, delta=1.0 / sync.SR)

    def test_common_window(self):
        s = [sync.SourceSync("a", "", 0.0, 10.0, 9), sync.SourceSync("b", "", 1.5, 10.0, 9)]
        self.assertEqual(sync.common_window(s, {"a", "b"}), (1.5, 10.0))


class AlignTests(unittest.TestCase):
    def test_floor_recovery(self):
        rng = np.random.default_rng(2)
        # tilted, offset floor + clutter, in a made-up "scene" frame with scale 0.5
        n_true = np.array([0.2, 0.95, 0.1])
        n_true /= np.linalg.norm(n_true)
        u = np.cross(n_true, [1, 0, 0]); u /= np.linalg.norm(u)
        v = np.cross(n_true, u)
        floor = (rng.uniform(-3, 3, (4000, 1)) * u + rng.uniform(-3, 3, (4000, 1)) * v) + 0.7 * n_true
        clutter = rng.uniform(-3, 3, (1500, 3))
        pts = np.vstack([floor, clutter])
        n, d, _ = align.ransac_plane(pts, 0.02, up_hint=n_true)
        self.assertGreater(n @ n_true, 0.999)
        dancers = 0.7 * n_true + 0.5 * u + 0.4 * n_true  # 0.4 above floor
        M = align.scene_to_stage(n, d, dancers, scale=2.0)
        p = M @ np.r_[dancers, 1]
        np.testing.assert_allclose(p[:3], [0, 0, 0.8], atol=1e-3)  # 0.4 scene units * 2 = 0.8 m up


class ExportTests(unittest.TestCase):
    def _person(self, frames, height, x):
        j = np.zeros((frames, 127, 3))
        j[:, :, 2] = np.linspace(0.05, height, 127)[None]
        j[:, 10, 2] = j[:, 11, 2] = 0.05
        j[:, 15, 2] = height - 0.1
        j[:, :, 0] = x
        return j

    def test_roles_mapping_and_files(self):
        people = {3: unity_export.smplx_to_smpl24(self._person(40, 1.6, 0.5)),
                  7: unity_export.smplx_to_smpl24(self._person(40, 1.85, -0.5))}
        self.assertEqual(people[3].shape, (40, 24, 3))
        lead, follow, _ = unity_export.assign_roles(people, "auto", "auto")
        self.assertEqual((lead, follow), (7, 3))
        with tempfile.TemporaryDirectory() as d:
            beats = Path(d) / "b.json"
            beats.write_text(json.dumps(zouk_grid(90, 2)))
            out = unity_export.export_capture(Path(d) / "cap", people[lead], people[follow], 20.0, None, 0.0, beats)
            fig = json.loads((out / "figure1.json").read_text())
            meta = json.loads((out / "video_meta.json").read_text())
            man = json.loads((out / "capture.json").read_text())
        self.assertEqual(len(fig), 40)
        self.assertEqual(len(fig[0]), 24)
        self.assertAlmostEqual(fig[0][10]["y"], 0.0, places=4)  # feet on floor, y-up
        self.assertAlmostEqual(fig[0][0]["x"], -0.5)
        self.assertEqual(meta, {"duration": 2.0, "frame_count": 40})
        self.assertEqual(man["beats"], "zouk-time-analysis.json")

    def test_splat_scene_to_unity_roundtrip(self):
        from atlas_bridge.splat_io import studio, write_3dgs_ply
        rng = np.random.default_rng(4)
        xyz_stage, rgb = studio(rng, radius=4.5)
        # scene frame: y-up, 0.5 units per metre, offset - like an unscaled Atlas reconstruction
        S = np.eye(4)
        S[:3, :3] = 2.0 * align.rotation_between(np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
        S[:3, 3] = [0.4, -0.1, 0.0]
        inv = np.linalg.inv(S)
        xyz_scene = xyz_stage @ inv[:3, :3].T + inv[:3, 3]
        people = {0: self._person(10, 1.8, 0.3), 1: self._person(10, 1.6, -0.3)}
        with tempfile.TemporaryDirectory() as d:
            ply = Path(d) / "env.ply"
            write_3dgs_ply(ply, xyz_scene, rgb, sigma=0.01)
            back = align.read_ply_xyz(ply, max_points=10**7)
            np.testing.assert_allclose(back, xyz_scene, atol=1e-5)
            out = unity_export.export_capture(Path(d) / "cap", unity_export.smplx_to_smpl24(people[0]),
                                              unity_export.smplx_to_smpl24(people[1]), 20.0, None, 0.0, None,
                                              environment_splat=ply, scene_to_stage=S)
            man = json.loads((out / "capture.json").read_text())
        M = np.array(man["environment_splat"]["scene_to_unity"]).reshape(4, 4)  # row-major, as SplatPly.cs reads it
        u = back @ M[:3, :3].T + M[:3, 3]
        floor_z = unity_export.floor_offset([unity_export.smplx_to_smpl24(p) for p in people.values()])
        floor = u[: 60_000]
        np.testing.assert_allclose(floor[:, 1], -floor_z, atol=1e-4)  # floor is shifted with the dancers
        wall_r = np.hypot(u[60_000:, 0], u[60_000:, 2])
        np.testing.assert_allclose(wall_r, 4.5, atol=1e-3)

    def test_zouk_grid_matches_analyzer_pattern(self):
        g = zouk_grid(120, 2.0)
        self.assertEqual([b[1] for b in g[:8]], [1, 3, 3, 2, 1, 3, 2, 3])
        self.assertAlmostEqual(g[1][0], 0.25)


if __name__ == "__main__":
    unittest.main()
