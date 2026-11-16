"""RGB + LiDAR fusion and independent visual aiming for planner teleop."""
import time
import mujoco
import numpy as np
from . import gimbal_perception as perception


class GimbalSensors:
    def __init__(self, overlay, enabled=True, auto=True, hz=15.):
        self.overlay = overlay
        self.enabled, self.auto, self.hz = enabled, auto, hz
        self.renderer = None
        self.scan = None
        self.measurement = (None, None)
        self.rgb = None
        self.next_scan = 0.
        self.goal = overlay.angles.copy() if overlay is not None else np.zeros(2)
        self.velocity = np.zeros(2)
        self.last_measurement = -np.inf
        self.status = 'waiting for fusion'
        self.toggle = False

    def __enter__(self):
        if self.enabled and self.overlay is not None:
            self.renderer = mujoco.Renderer(self.overlay.model, height=480, width=640, max_geom=5000)
        return self

    def __exit__(self, *exc):
        if self.renderer is not None:
            self.renderer.close()

    def key_callback(self, key):
        if key == 294:  # F5: no overlap with walking, speed, reset or skills
            self.toggle = not self.toggle
        elif key in map(ord, 'IJKLU'):
            self.auto = False

    def update(self, dt=.02, now=None):
        if self.renderer is None:
            return
        now = time.monotonic() if now is None else now
        if self.toggle:
            self.auto = not self.auto
            self.toggle = False
        g = self.overlay
        m, d = g.model, g.data
        if now >= self.next_scan:
            self.scan = perception.scan_lidar(m, d)
            self.renderer.update_scene(d, camera='fixed_camera')
            pixels = self.renderer.render().copy()
            self.measurement = perception.estimate_fusion(m, d, pixels, self.scan)
            detection, fusion = self.measurement
            if self.auto:
                angles = g.solver.solve(fusion['point_carrier_m']) if fusion.get('valid') else None
                if angles is not None:
                    self.goal = angles
                    self.last_measurement = now
                    self.status = 'tracking'
                else:
                    self.goal = g.angles.copy()
                    self.velocity[:] = 0
                    self.status = 'target unreachable' if fusion.get('valid') else 'fusion unavailable'
            self.rgb = self._hud(pixels, detection, fusion)
            self.next_scan = now + 1/self.hz
        if self.auto and now-self.last_measurement <= .25:
            error = self.goal-g.angles
            speed, acceleration = np.deg2rad(60), np.deg2rad(180)
            desired = np.sign(error)*np.minimum(speed, np.sqrt(2*acceleration*np.abs(error)))
            self.velocity += np.clip(desired-self.velocity, -acceleration*dt, acceleration*dt)
            move = self.velocity*dt
            arrived = (np.abs(move) >= np.abs(error)) & (move*error >= 0)
            g.angles += np.where(arrived, error, move)
            self.velocity[arrived] = 0
            g.angles = np.clip(g.angles, g.limits[:, 0], g.limits[:, 1])
            g.target = None
            g.data.qpos[g.qpos] = g.angles
            mujoco.mj_forward(m, d)
        elif self.auto and self.status == 'tracking':
            self.status = 'measurement stale'
            self.velocity[:] = 0
        elif not self.auto:
            self.goal = g.angles.copy()
            self.velocity[:] = 0
            self.status = 'manual'

    def _hud(self, rgb, detection, fusion):
        import cv2
        image = rgb.copy()
        text = 'Marker not visible'
        if detection is not None:
            x, y = detection['xy']
            text = f'Marker u={x:.1f} v={y:.1f}'
            cv2.drawMarker(image, (round(x), round(y)), (0, 255, 0))
        lines = [text]
        if fusion.get('valid'):
            lines += [f"Depth Z={fusion['depth_z_m']:.3f} m", 'Carrier XYZ: '+', '.join(f'{v:.3f}' for v in fusion['point_carrier_m'])]
        else:
            lines += ['Fusion: '+fusion.get('reason', 'unavailable')]
        lines += ['Aim: '+self.status+' | F5 auto/manual']
        for i, line in enumerate(lines):
            cv2.putText(image, line, (8, 22+22*i), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 255, 0), 1)
        return np.ascontiguousarray(image)  # viewer.set_images handles the vertical flip

    def draw(self, scene):
        if self.renderer is None:
            self.overlay.draw(scene)
            return
        if self.scan is not None:
            perception.draw_scan(scene, self.scan, 3.)
        perception.draw_laser(scene, perception.cast_laser(self.overlay.model, self.overlay.data))
