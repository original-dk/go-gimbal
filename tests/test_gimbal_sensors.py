"""Rendered perception test, independent of policy/checkpoints and CUDA."""
import unittest
import mujoco
import numpy as np
from supermimic.planner.gimbal import GimbalOverlay
from supermimic.planner.gimbal_runtime import GimbalSensors
from supermimic.planner import keys as K


class GimbalSensorTests(unittest.TestCase):
    def test_fusion_tracking_motion_loss_and_manual(self):
        g = GimbalOverlay()
        with GimbalSensors(g) as runtime:
            for pose, euler in [([0, 0, .4], [0., 0., 0.]),
                                ([.1, -.05, .4], [.08, -.08, .15])]:
                quat = np.empty(4)
                mujoco.mju_euler2Quat(quat, np.array(euler), 'xyz')
                g.update(pose, quat)
                start = runtime.next_scan+1
                for i in range(100):
                    runtime.update(now=start+i*.02)
                self.assertEqual(runtime.status, 'tracking')
                fusion = runtime.measurement[1]
                site = g.model.site('laser_origin').id
                origin = g.data.site_xpos[site]
                direction = g.data.site_xmat[site].reshape(3, 3)[:, 0]
                self.assertLess(np.linalg.norm(np.cross(np.array(fusion['point_world_m'])-origin, direction)), .003)
                self.assertEqual(runtime.rgb.shape, (480, 640, 3))
            # Turn away: no marker, no target truth fallback.
            mujoco.mju_euler2Quat(quat, np.array([0., 0., np.pi]), 'xyz')
            g.update([0, 0, .4], quat)
            runtime.update(now=runtime.next_scan+1)
            self.assertEqual(runtime.status, 'fusion unavailable')
            runtime.key_callback(294)
            runtime.update(now=runtime.next_scan+1)
            self.assertFalse(runtime.auto)
            runtime.draw(runtime.renderer.scene)

    def test_existing_walking_and_skill_key_edges(self):
        keys = K.Keys()
        keys.edge(K.UP, True, 0)
        self.assertEqual(keys.axes(), (1, 0, 0))
        keys.edge(K.UP, False, 0)
        keys.edge(K.LEFT, True, K.MOD_SHIFT)
        self.assertEqual(keys.axes(), (0, 1, 0))
        keys.edge(K.LEFT, False, 0)
        self.assertEqual(keys.axes(), (0, 0, 0))
        keys.edge(K.F6, True, 0)
        self.assertEqual(keys.pop_events(), ['skill:sit'])
