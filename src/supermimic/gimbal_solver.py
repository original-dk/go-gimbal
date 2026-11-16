"""Carrier-independent XYZ-to-gimbal angle solver (meters, radians, wxyz)."""
from pathlib import Path
import mujoco
import numpy as np


def rigid_pose(pos, quat):
    pos, quat = np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
    if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError('Expected finite pos[3] and nonzero quaternion[w,x,y,z]')
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
    return pos.copy(), rotation.reshape(3, 3)


class GimbalAngleSolver:
    """Fixed mounts are expressed in one carrier frame; no live carrier pose needed.

    Input XYZ is in the carrier frame, in meters (no camera dependency).
    solve returns [yaw, pitch] in radians, or None for an unreachable target.
    """

    def __init__(self, gimbal_pos, gimbal_quat,
                 xml=None):
        self.pos, self.rotation = rigid_pose(gimbal_pos, gimbal_quat)
        xml = xml or Path(__file__).resolve().parent / 'robots/assets/gimbal/gimbal.xml'
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.scratch = mujoco.MjData(self.model)
        joints = [self.model.joint(n).id for n in ('yaw_joint', 'pitch_joint')]
        self.qpos = self.model.jnt_qposadr[joints]
        self.limits = self.model.jnt_range[joints].copy()
        self.site = self.model.site('laser_origin').id
        self.mount_pos = self.pos.copy()
        self.mount_quat = np.asarray(gimbal_quat, dtype=float).copy()

    def solve(self, point_carrier):
        point = np.asarray(point_carrier, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        target = self.rotation.T @ (point - self.pos)
        seed = np.array([np.arctan2(target[1], target[0]),
                         np.arctan2(target[2], np.linalg.norm(target[:2]))])

        def residual(angles):
            self.scratch.qpos[self.qpos] = angles
            mujoco.mj_kinematics(self.model, self.scratch)
            origin = self.scratch.site_xpos[self.site]
            direction = self.scratch.site_xmat[self.site].reshape(3, 3)[:, 0]
            vector = target - origin
            distance = np.linalg.norm(vector)
            return np.ones(3)*10 if distance < 1e-6 else vector/distance-direction

        angles = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
        for _ in range(40):
            error = residual(angles)
            if np.linalg.norm(error) < 1e-7:
                return angles.copy()
            jacobian = np.column_stack([(residual(angles+np.eye(2)[i]*1e-5)-error)/1e-5 for i in range(2)])
            change = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
            angles = np.clip(angles+np.clip(change, -.2, .2), self.limits[:, 0], self.limits[:, 1])
        return angles.copy() if np.linalg.norm(residual(angles)) < 1e-4 else None
