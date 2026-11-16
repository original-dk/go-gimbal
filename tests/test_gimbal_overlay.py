"""CPU-only tests: no tracker, training data, CUDA or mjlab required."""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from supermimic.planner.gimbal import GimbalOverlay


class GimbalOverlayTests(unittest.TestCase):
    def test_fixed_mount_and_aim_follow_trunk_pose(self):
        g = GimbalOverlay(target=[1.5, .15, .2])
        np.testing.assert_allclose(g.solver.pos, [.05, 0, .13])
        for euler in ([0., 0., 0.], [.2, -.1, .7], [-.2, .3, -1.]):
            quat = np.empty(4)
            mujoco.mju_euler2Quat(quat, np.array(euler), 'xyz')
            g.update([1, -2, .45], quat)
            carrier = g.model.body('link_trunk').id
            world_target = g.data.xpos[carrier] + g.data.xmat[carrier].reshape(3, 3) @ g.target
            site = g.model.site('laser_origin').id
            origin = g.data.site_xpos[site]
            direction = g.data.site_xmat[site].reshape(3, 3)[:, 0]
            np.testing.assert_allclose(direction, (world_target-origin)/np.linalg.norm(world_target-origin), atol=1e-6)

    def test_live_checker_and_renamed_policy_root_are_preserved(self):
        g = GimbalOverlay()
        spec = mujoco.MjSpec.from_file(str(Path(__file__).resolve().parents[1]/'src/supermimic/robots/assets/rover/dobot.xml'))
        spec.joint('joint_fixed_world').name = 'robot/floating_base'
        source = spec.compile()
        # Use a textured source display model to exercise the live texture copy.
        textured = GimbalOverlay(source_model=g.model)
        np.testing.assert_array_equal(textured.model.tex_data, g.model.tex_data)
        pos = mujoco.MjData(source).qpos.copy()
        pos[:3] = [2, -1, .45]
        pos[3:7] = [1, 0, 0, 0]
        g.sync_policy(source, pos, np.zeros(source.nv))
        np.testing.assert_allclose(g.data.qpos[:7], pos[:7])

    def test_keys_limits_and_no_policy_model_changes(self):
        g = GimbalOverlay()
        self.assertFalse(g.key_callback(265))  # walking key is not consumed
        for _ in range(100):
            self.assertTrue(g.key_callback(ord('J')))
        g.update([0, 0, .45], [1, 0, 0, 0])
        self.assertAlmostEqual(g.angles[0], g.limits[0, 1])
        g.key_callback(ord('U'))
        g.update([0, 0, .45], [1, 0, 0, 0])
        np.testing.assert_array_equal(g.angles, [0, 0])
        xml = Path(__file__).resolve().parents[1]/'src/supermimic/robots/assets/rover/dobot.xml'
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        before = data.qpos.copy()
        scene = mujoco.MjvScene(model, maxgeom=1000)
        g.draw(scene)
        self.assertGreater(scene.ngeom, 0)
        g.sync_policy(model, data.qpos, data.qvel)
        self.assertEqual(model.nq, 19)
        self.assertEqual(model.nv, 18)
        np.testing.assert_array_equal(data.qpos, before)


if __name__ == '__main__':
    unittest.main()
