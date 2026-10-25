"""Position retargeting and actuator control for the bundled three-joint arm."""
from pathlib import Path
import math

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE_LOW = np.array([0.20, -0.25, 0.12])
WORKSPACE_HIGH = np.array([0.53, 0.25, 0.48])


def clamp_workspace(target):
    """Bound the target, including distance from the shoulder pivot."""
    point = np.clip(np.asarray(target, dtype=float), WORKSPACE_LOW, WORKSPACE_HIGH)
    pivot = np.array([0.0, 0.0, 0.18])
    offset = point - pivot
    distance = np.linalg.norm(offset)
    if distance > 0.565:
        point = pivot + offset * (0.565 / distance)
    return point


class Arm:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "arm.xml"))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool")
        self.target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = int(self.model.body_mocapid[self.target_body])
        self.limits = self.model.jnt_range.copy()
        self.jacobian = np.zeros((3, self.model.nv))
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.command = self.data.qpos.copy()
        self.goal = self.command.copy()
        self.home_target = self.data.site_xpos[self.site].copy()
        self.target = self.home_target.copy()
        self.data.mocap_pos[self.mocap_id] = self.target

    @property
    def tool_position(self):
        return self.data.site_xpos[self.site].copy()

    def set_target(self, target):
        point = np.asarray(target, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("Target must contain three finite coordinates")
        self.target = clamp_workspace(point)
        self.data.mocap_pos[self.mocap_id] = self.target
        self.ik_data.qpos[:] = self.command
        for _ in range(12):
            mujoco.mj_forward(self.model, self.ik_data)
            error = self.target - self.ik_data.site_xpos[self.site]
            if np.linalg.norm(error) < 0.002:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, self.jacobian, None, self.site)
            delta = self.jacobian.T @ np.linalg.solve(
                self.jacobian @ self.jacobian.T + 0.002**2 * np.eye(3), error
            )
            self.ik_data.qpos[:] = np.clip(
                self.ik_data.qpos + np.clip(delta, -0.12, 0.12),
                self.limits[:, 0], self.limits[:, 1],
            )
        self.goal = self.ik_data.qpos.copy()

    def hold(self):
        """Freeze the actuator setpoint when input is paused or lost."""
        self.goal = self.command.copy()
        self.target = self.tool_position
        self.data.mocap_pos[self.mocap_id] = self.target

    def step(self):
        goal = getattr(self, "goal", self.command)
        max_change = 1.2 * self.model.opt.timestep
        self.command += np.clip(goal - self.command, -max_change, max_change)
        self.data.ctrl[:] = self.command
        mujoco.mj_step(self.model, self.data)
        # mj_step may leave position-dependent output at the pre-integration state.
        mujoco.mj_forward(self.model, self.data)


class WristMapper:
    """Map shoulder-relative estimated wrist motion after a neutral calibration.

    Default mapping uses image x/y only. Optional world z is a monocular estimate.
    The camera stays fixed; this is relative motion control, not motion capture in
    a measured room coordinate frame.
    """
    def __init__(self, home, scale=0.35, depth=False):
        self.home = np.asarray(home, dtype=float).copy()
        self.scale = float(scale)
        self.depth = depth
        self.neutral = None
        self.filtered = self.home.copy()
        self.last_time = None

    def calibrate(self, sample):
        self.neutral = np.asarray(sample, dtype=float).copy()
        self.filtered = self.home.copy()
        self.last_time = None

    def update(self, sample, now):
        if self.neutral is None:
            return None
        delta = np.asarray(sample, dtype=float) - self.neutral
        # Image right -> robot y; image up -> robot z; camera-near -> robot x.
        offset = self.scale * np.array([
            -delta[2] if self.depth else 0.0,
            delta[0],
            -delta[1],
        ])
        desired = clamp_workspace(self.home + offset)
        elapsed = 1.0 / 30.0 if self.last_time is None else max(0.0, now - self.last_time)
        elapsed = min(elapsed, 0.1)
        alpha = 1.0 - math.exp(-elapsed / 0.12)
        next_point = self.filtered + alpha * (desired - self.filtered)
        difference = next_point - self.filtered
        magnitude = np.linalg.norm(difference)
        max_distance = 0.4 * elapsed
        if magnitude > max_distance and magnitude > 0:
            next_point = self.filtered + difference * max_distance / magnitude
        self.filtered = next_point
        self.last_time = now
        return self.filtered.copy()
