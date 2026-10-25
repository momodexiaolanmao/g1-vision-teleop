"""Calibrated camera joint references for the physical TWIST2 controller.

This adapter commands in-place body poses. It cannot recover a reliable walking
trajectory from root-local monocular landmarks. Forward kinematics below uses a
separate MjData; measured floating-base pose and velocity are never overwritten.
TWIST2 controller/checkpoint attribution and MIT license: see twist2_tracking.py.
"""
import math

import mujoco
import numpy as np

from twist2_tracking import DEFAULT_JOINT_POS, DEFAULT_REFERENCE, JOINT_NAMES, Twist2Tracking


class TwistCameraBalance:
    def __init__(self, model, data, config=None):
        self.model, self.data, self.config = model, data, dict(config or {})
        self.controller = Twist2Tracking(model, data, self.config)
        self.index = {name: index for index, name in enumerate(JOINT_NAMES)}
        self.blend_seconds = float(self.config.get("reference_blend_seconds", .12))
        self.max_offset = math.radians(float(self.config.get("max_reference_offset_deg", 60)))
        self.max_leg_offset = math.radians(float(self.config.get(
            "max_leg_reference_offset_deg", min(30., math.degrees(self.max_offset)))))
        self.height_limits = np.asarray(self.config.get("root_height_range_m", [.60, .92]), dtype=float)
        if (not np.isfinite(self.blend_seconds) or self.blend_seconds < 0
                or not np.isfinite(self.max_offset) or not 0 < self.max_offset <= math.pi
                or not np.isfinite(self.max_leg_offset) or not 0 < self.max_leg_offset <= self.max_offset
                or self.height_limits.shape != (2,) or not np.all(np.isfinite(self.height_limits))
                or not .45 < self.height_limits[0] < self.height_limits[1] < 1.2):
            raise ValueError("Invalid camera reference blending, offset or height limits")
        self.fk_data = mujoco.MjData(model)
        self.foot_geoms = self.controller.foot_geoms.copy()
        if len(self.foot_geoms) != 8:
            raise ValueError("TWIST2 camera height adjustment requires its eight foot contact spheres")
        self.base_foot_z = self._minimum_reference_foot_z(DEFAULT_JOINT_POS)
        self.joint_limits = model.jnt_range[self.controller.joints].copy()
        self.reset()

    @property
    def fallen(self):
        return self.controller.fallen

    @property
    def solves(self):
        return self.controller.solves

    @property
    def failures(self):
        return self.controller.failures

    def _minimum_reference_foot_z(self, angles):
        d = self.fk_data
        d.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
        d.qpos[self.controller.qadr] = angles
        mujoco.mj_forward(self.model, d)
        return min(float(d.geom_xpos[g, 2] - self.model.geom_size[g, 0]) for g in self.foot_geoms)

    def reset(self):
        """Reset input/controller memory without resetting the physical robot."""
        self.controller.reset()
        self.human_neutral = None
        self.reliable_mask = np.zeros(29, dtype=bool)
        self.last_valid_raw_goal = np.full(29, np.nan)
        self.tracking_enabled = False
        self.command = np.zeros(3)
        self.goal_offset = np.zeros(29)
        self.offset = np.zeros(29)
        self.reference = DEFAULT_REFERENCE.copy()
        self.controller.set_motion_reference(self.reference)

    def _mapped_vector(self, values, entries, fallback):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(entries),) or not np.all(np.isfinite(values)):
            raise ValueError("Camera mapping needs one finite angle per mapping entry")
        result = np.asarray(fallback, dtype=float).copy()
        seen = set()
        for value, entry in zip(values, entries):
            name = entry["rule"]["robot_joint"]
            if name not in self.index or name in seen:
                raise ValueError(f"Unknown or duplicate camera joint: {name}")
            result[self.index[name]] = value
            seen.add(name)
        return result

    def _valid_vector(self, entries, valid_mask=None):
        mask = np.ones(len(entries), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if mask.shape != (len(entries),):
            raise ValueError("Camera validity mask needs one flag per mapping entry")
        result = np.zeros(29, dtype=bool)
        for valid, entry in zip(mask, entries):
            name = entry["rule"]["robot_joint"]
            if name not in self.index:
                raise ValueError(f"Unknown camera joint: {name}")
            result[self.index[name]] = valid
        return result

    def _capture_fresh_input(self, mapped_values, entries, valid_mask=None):
        mapped = self._mapped_vector(mapped_values, entries, self.last_valid_raw_goal)
        self.reliable_mask = self._valid_vector(entries, valid_mask)
        first = self.reliable_mask & ~np.isfinite(self.human_neutral)
        self.human_neutral[first] = mapped[first]
        self.last_valid_raw_goal[self.reliable_mask] = mapped[self.reliable_mask]

    def calibrate(self, mapped_values, entries, valid_mask=None):
        self.human_neutral = np.full(29, np.nan)
        self.last_valid_raw_goal = np.full(29, np.nan)
        self._capture_fresh_input(mapped_values, entries, valid_mask)
        self.goal_offset[:] = 0
        self.tracking_enabled = False
        return True

    def allow_tracking(self, mapped=None, entries=None, valid_mask=None):
        if mapped is not None and self.human_neutral is not None:
            if entries is None:
                raise ValueError("Fresh camera values require their mapping entries")
            self._capture_fresh_input(mapped, entries, valid_mask)
        self.tracking_enabled = self.human_neutral is not None

    def set_reference(self, values, entries):
        # JointRobot calls this on every physics step, including while held.
        # Only a fresh, explicitly enabled camera input may resume tracking.
        if not self.tracking_enabled or self.human_neutral is None:
            return
        mapped = self._mapped_vector(values, entries, self.last_valid_raw_goal)
        self.last_valid_raw_goal[self.reliable_mask] = mapped[self.reliable_mask]
        defined = np.isfinite(self.human_neutral) & np.isfinite(self.last_valid_raw_goal)
        offset = np.zeros(29)
        offset[defined] = np.clip(self.last_valid_raw_goal[defined] - self.human_neutral[defined],
                                  -self.max_offset, self.max_offset)
        # Root-local camera estimates omit support-side COM shift and root speed.
        # Keep each leg's relative pose inside a conservative envelope, scaling
        # its six angles together so knee/hip/ankle coordination is preserved.
        for start in (0, 6):
            leg = offset[start:start+6]
            if not np.all(self.reliable_mask[[start, start+3, start+4]]):
                leg[:] = 0
            magnitude = float(np.max(abs(leg)))
            if magnitude > self.max_leg_offset:
                leg *= self.max_leg_offset / magnitude
        goal = np.clip(DEFAULT_JOINT_POS + offset, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.goal_offset = goal - DEFAULT_JOINT_POS

    def hold(self):
        self.tracking_enabled = False
        self.goal_offset[:] = 0

    def step(self):
        dt = self.model.opt.timestep
        alpha = 1. if self.blend_seconds == 0 else 1. - math.exp(-dt / self.blend_seconds)
        self.offset += alpha * (self.goal_offset - self.offset)
        if self.controller.physics_steps % self.controller.decimation == 0:
            angles = DEFAULT_JOINT_POS + self.offset
            current_foot_z = self._minimum_reference_foot_z(angles)
            height = np.clip(.8 + self.base_foot_z - current_foot_z, *self.height_limits)
            self.reference = np.r_[0., 0., height, 0., 0., 0., angles]
            self.controller.set_motion_reference(self.reference)
        self.controller.step()

    def status(self):
        result = self.controller.status()
        if not self.fallen:
            result["state"] = "TRACKING" if self.tracking_enabled else "HOLDING"
        result.update(camera_calibrated=self.human_neutral is not None,
                      camera_tracking_enabled=self.tracking_enabled,
                      camera_reference_valid_mask=self.reliable_mask.tolist(),
                      camera_neutral_defined_mask=(np.isfinite(self.human_neutral).tolist()
                                                   if self.human_neutral is not None else [False]*29),
                      velocity_command=self.command.tolist(),
                      camera_reference_offset_deg=np.degrees(self.offset).tolist(),
                      max_leg_reference_offset_deg=math.degrees(self.max_leg_offset),
                      solver_status="TWIST2 camera body-pose reference")
        return result
