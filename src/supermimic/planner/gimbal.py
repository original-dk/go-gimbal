"""Independent visual gimbal attached to the tracker trunk; no policy state changes."""
from pathlib import Path
from queue import SimpleQueue
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from supermimic.gimbal_solver import GimbalAngleSolver

ASSETS = Path(__file__).resolve().parents[1] / 'robots/assets/gimbal'


class GimbalOverlay:
    """J/L yaw, I/K pitch, U reset. Values are radians; target XYZ is trunk-local meters."""

    def __init__(self, yaw=0., pitch=0., target=None):
        # A separate combined display model gives the renderer valid mesh IDs.
        rover = ASSETS.parent / 'rover/dobot.xml'
        root = ET.parse(rover).getroot()
        root.find('compiler').set('meshdir', str(rover.parent / 'assets'))
        for mesh in ET.parse(ASSETS / 'components/gimbal_assets.xml').getroot():
            mesh.set('file', str(ASSETS / mesh.get('file')))
            root.find('asset').append(mesh)
        trunk = root.find("./worldbody/body[@name='link_trunk']")
        mount = ET.parse(ASSETS / 'overlay.xml').find("./worldbody/body/body[@name='gimbal_mount']")
        mount.remove(mount.find('include'))
        mount.append(ET.parse(ASSETS / 'components/gimbal_body.xml').getroot().find('body'))
        trunk.append(mount)
        ET.SubElement(root.find('worldbody'), 'geom', name='display_floor', type='plane',
                      size='20 20 .1', rgba='.55 .58 .6 1')
        ET.SubElement(root.find('worldbody'), 'light', pos='0 0 4', dir='0 0 -1')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        mount = self.model.body('gimbal_mount').id
        self.solver = GimbalAngleSolver(
            self.model.body_pos[mount], self.model.body_quat[mount], xml=ASSETS / 'gimbal.xml')
        joints = [self.model.joint(n).id for n in ('yaw_joint', 'pitch_joint')]
        self.qpos = self.model.jnt_qposadr[joints]
        self.limits = self.model.jnt_range[joints]
        self.angles = np.clip([yaw, pitch], self.limits[:, 0], self.limits[:, 1])
        self.events = SimpleQueue()
        self.option = mujoco.MjvOption()
        self.target = None
        self.status = 'manual'
        if target is not None:
            self.aim(target)

    def aim(self, point_carrier):
        """Accept a point already expressed in the trunk frame, not camera pixels."""
        angles = self.solver.solve(point_carrier)
        if angles is None:
            self.status = 'target unreachable'
            return False
        self.angles = angles
        self.target = np.asarray(point_carrier, dtype=float).copy()
        self.status = 'carrier XYZ target'
        return True

    def key_callback(self, key):
        if key in map(ord, 'IJKLU'):
            self.events.put(key)
            return True
        return False

    def update(self, trunk_pos, trunk_quat):
        while not self.events.empty():
            key = self.events.get()
            self.target = None
            self.status = 'manual'
            if key == ord('U'):
                self.angles[:] = 0
            else:
                axis = 0 if key in (ord('J'), ord('L')) else 1
                self.angles[axis] += np.deg2rad(2) * (1 if key in (ord('J'), ord('I')) else -1)
            self.angles = np.clip(self.angles, self.limits[:, 0], self.limits[:, 1])
        self.data.qpos[:3] = trunk_pos
        self.data.qpos[3:7] = trunk_quat
        self.data.qpos[self.qpos] = self.angles
        mujoco.mj_forward(self.model, self.data)

    def sync_policy(self, source_model, qpos, qvel):
        """Copy only policy joints by name; gimbal angles remain independent."""
        lookup = {source_model.joint(i).name.split('/')[-1]: i for i in range(source_model.njnt)}
        for j in range(self.model.njnt):
            name = self.model.joint(j).name
            if name not in lookup:
                continue
            k = lookup[name]
            nq, nv = (7, 6) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (1, 1)
            a, b = self.model.jnt_qposadr[j], source_model.jnt_qposadr[k]
            self.data.qpos[a:a+nq] = qpos[b:b+nq]
            a, b = self.model.jnt_dofadr[j], source_model.jnt_dofadr[k]
            self.data.qvel[a:a+nv] = qvel[b:b+nv]
        self.update(self.data.qpos[:3].copy(), self.data.qpos[3:7].copy())

    def draw(self, scene):
        if scene.ngeom >= scene.maxgeom:
            return
        site = self.model.site('laser_origin').id
        origin = self.data.site_xpos[site]
        direction = self.data.site_xmat[site].reshape(3, 3)[:, 0]
        endpoint = origin + 2 * direction
        if self.target is not None:
            carrier = self.model.body('link_trunk').id
            target_world = self.data.xpos[carrier] + self.data.xmat[carrier].reshape(3, 3) @ self.target
            endpoint = origin + np.linalg.norm(target_world-origin) * direction
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                          np.eye(3).ravel(), np.array([1., 0., 0., 1.]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., origin, endpoint)
        scene.ngeom += 1

    def label(self):
        yaw, pitch = np.rad2deg(self.angles)
        return f'Gimbal yaw {yaw:.1f} pitch {pitch:.1f} deg | {self.status} | J/L yaw, I/K pitch, U zero (visual only)'
