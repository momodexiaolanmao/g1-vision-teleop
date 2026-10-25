"""Kinematic root preview. Does not integrate dynamics or claim balance."""
import math
import mujoco
import numpy as np

from pose_retarget import rotate


class ShadowMotion:
    def __init__(self, model, data, config):
        self.model, self.data, self.config = model, data, config
        free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError("Shadow tracking needs exactly one free root joint")
        self.address = int(model.jnt_qposadr[free[0]])
        self.foot_geoms = [i for i in range(model.ngeom)
                           if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                           and "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                               int(model.geom_bodyid[i])) or "")]
        self.reset()

    def reset(self):
        self.home = self.data.qpos[self.address:self.address + 3].copy()
        self.command = np.r_[self.home, 0.0, 0.0]
        self.goal = self.command.copy()
        self.diagnostic = {"valid": False, "reason": "press C to calibrate body position"}

    def update(self, root):
        self.diagnostic = dict(root)
        if not root.get("valid"):
            self.hold()
            return
        max_lateral = float(self.config.get("max_lateral_m", 1.8))
        max_depth = float(self.config.get("max_depth_m", 1.2))
        self.goal[1] = self.home[1] + np.clip(root["lateral_m"], -max_lateral, max_lateral)
        self.goal[2] = self.home[2] + np.clip(root["vertical_m"], -0.65, 1.0)
        if self.config.get("estimate_depth", False) and root.get("depth_reliable"):
            distance = float(self.config.get("camera_distance_m", 2.5))
            self.goal[0] = self.home[0] + np.clip(distance * (1 - root["scale_ratio"]), -max_depth, max_depth)
        # Unwrap turns relative to the current root, allowing a continuous spin.
        yaw = float(root["yaw_rad"])
        self.goal[3] = self.command[3] + math.atan2(math.sin(yaw - self.command[3]),
                                                 math.cos(yaw - self.command[3]))
        self.goal[4] = np.clip(root["roll_rad"], -0.45, 0.45)

    def hold(self):
        self.goal = self.command.copy()

    def step(self, dt):
        smoothing = float(self.config.get("smoothing_seconds", 0.12))
        alpha = 1 if smoothing == 0 else 1 - math.exp(-dt / smoothing)
        delta = alpha * (self.goal - self.command)
        speed = float(self.config.get("max_speed_m_s", 1.8))
        angular_speed = math.radians(float(self.config.get("max_turn_deg_s", 240)))
        caps = np.array([speed, speed, speed, angular_speed, angular_speed]) * dt
        self.command += np.clip(delta, -caps, caps)
        self.data.qpos[self.address:self.address + 3] = self.command[:3]
        matrix = rotate("z", self.command[3]) @ rotate("x", self.command[4])
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, matrix.flatten())
        self.data.qpos[self.address + 3:self.address + 7] = quaternion

    def keep_feet_above_floor(self):
        if not self.config.get("floor_guard", True) or not self.foot_geoms:
            return
        bottom = min(self.data.geom_xpos[i, 2] - self.model.geom_size[i, 0] for i in self.foot_geoms)
        correction = max(0.0, 0.001 - bottom)
        if correction > 0:
            self.data.qpos[self.address + 2] += correction
            self.command[2] += correction
            mujoco.mj_forward(self.model, self.data)

    def status(self):
        return {**self.diagnostic, "position_m": self.command[:3].tolist(),
                "yaw_deg": math.degrees(self.command[3]), "roll_deg": math.degrees(self.command[4]),
                "depth_is_estimated": bool(self.config.get("estimate_depth", False)),
                "kinematic_preview": True}
