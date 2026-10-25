"""TWIST2 reference-conditioned G1 control through NumPy and CPU ONNX Runtime.

Adapted from amazon-far/TWIST2, commit
d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a, by Yanjie Ze (2025).
Upstream code and checkpoint: MIT; retain policies/twist2_tracking/LICENSE.
https://github.com/amazon-far/TWIST2

Only motor torques are written during control. A reference is a command to the
learned controller, never a write to the floating base or measured joint state.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
JOINT_NAMES = tuple(
    [f"{side}_{joint}_joint" for side in ("left", "right")
     for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{joint}_joint" for side in ("left", "right")
       for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                     "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
DEFAULT_JOINT_POS = np.array(
    [-.2, 0, 0, .4, -.2, 0] * 2 + [0, 0, 0]
    + [0, .4, 0, 1.2, 0, 0, 0] + [0, -.4, 0, 1.2, 0, 0, 0], dtype=np.float64)
INITIAL_JOINT_POS = DEFAULT_JOINT_POS.copy()
INITIAL_JOINT_POS[[16, 23]] = [.2, -.2]
DEFAULT_REFERENCE = np.r_[0., 0., .8, 0., 0., 0., DEFAULT_JOINT_POS]
KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [150] * 3
              + [40, 40, 40, 40, 4, 4, 4] * 2, dtype=np.float64)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [4] * 3
              + [5, 5, 5, 5, .2, .2, .2] * 2, dtype=np.float64)
EXPECTED_MODEL_MASS_KG = 33.342142


def quat_to_euler_wxyz(quaternion):
    """Upstream quatToEuler convention, with no external rotation dependency."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array([
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
        np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    ])


class Twist2Tracking:
    """The released 35-reference / 1432-observation / 29-action controller.

    ``set_motion_reference`` accepts local root vx/vy, root height, roll/pitch,
    local yaw rate, then the 29 joint angles in ``JOINT_NAMES`` order. ``step``
    performs one 1 ms physical step. The supplied upstream launcher uses 100 Hz
    policy inference; a different integer decimation can be selected explicitly.
    """
    def __init__(self, model, data, config=None):
        import onnxruntime as ort
        self.model, self.data, self.config = model, data, dict(config or {})
        if (model.nq != 36 or model.nv != 35 or model.nu != 29
                or model.neq or model.nmocap or np.any(model.body_gravcomp)
                or not np.allclose(model.opt.gravity, [0, 0, -9.81])):
            raise ValueError("TWIST2 requires a free-base, unsupported 29-joint gravity model")
        if not np.isclose(model.body_mass.sum(), EXPECTED_MODEL_MASS_KG, atol=1e-5, rtol=0):
            raise ValueError("TWIST2 needs its supplied 33.342142 kg model, not the current Unitree model")
        self.joints = np.array([int(model.actuator_trnid[a, 0]) for a in range(29)])
        self.names = tuple(model.joint(int(j)).name for j in self.joints)
        if self.names != JOINT_NAMES:
            raise ValueError("TWIST2 actuator joint order does not match the released checkpoint")
        if (np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
                or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE)
                or not np.allclose(model.actuator_gainprm[:, 0], 1)
                or np.any(model.actuator_biasprm)
                or not np.allclose(model.actuator_gear[:, 0], 1)):
            raise ValueError("TWIST2 requires direct, unit-gear torque motors")
        self.qadr = model.jnt_qposadr[self.joints].copy()
        self.vadr = model.jnt_dofadr[self.joints].copy()
        self.pelvis = model.body("pelvis").id
        self.feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
        self.foot_geoms = [g for g in range(model.ngeom)
                           if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                           and model.geom_bodyid[g] in self.feet]
        model.opt.timestep = .001
        self.frequency = float(self.config.get("policy_frequency", 100))
        if not np.isfinite(self.frequency) or not 0 < self.frequency <= 1000:
            raise ValueError("policy_frequency must be finite and between 0 and 1000 Hz")
        ratio = 1 / (self.frequency * model.opt.timestep)
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError("policy_frequency must divide the 1000 Hz physics frequency")
        self.decimation, self.period = int(round(ratio)), 1 / self.frequency
        self.kp, self.kd = KP.copy(), KD.copy()
        # Upstream unusually uses its KP vector as torque clipping limits.
        # Also enforce actual MJCF joint and actuator force/control limits.
        self.torque_low, self.torque_high = -KP.copy(), KP.copy()
        for a, j in enumerate(self.joints):
            if model.jnt_actfrclimited[j]:
                self.torque_low[a] = max(self.torque_low[a], model.jnt_actfrcrange[j, 0])
                self.torque_high[a] = min(self.torque_high[a], model.jnt_actfrcrange[j, 1])
            if model.actuator_forcelimited[a]:
                self.torque_low[a] = max(self.torque_low[a], model.actuator_forcerange[a, 0])
                self.torque_high[a] = min(self.torque_high[a], model.actuator_forcerange[a, 1])
            if model.actuator_ctrllimited[a]:
                self.torque_low[a] = max(self.torque_low[a], model.actuator_ctrlrange[a, 0])
                self.torque_high[a] = min(self.torque_high[a], model.actuator_ctrlrange[a, 1])
        self.policy_path = Path(self.config.get("policy_path", ROOT / "policies/twist2_tracking/policy.onnx"))
        source_path = self.policy_path.parent / "SOURCE.json"
        if source_path.exists():
            source = json.loads(source_path.read_text(encoding="utf-8"))
            expected = source.get("policy_sha256")
            if expected and hashlib.sha256(self.policy_path.read_bytes()).hexdigest() != expected:
                raise ValueError("TWIST2 checkpoint checksum does not match its source manifest")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(self.config.get("intra_op_num_threads", 1))
        options.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(self.policy_path), sess_options=options,
                                          providers=["CPUExecutionProvider"])
        inputs, outputs = self.policy.get_inputs(), self.policy.get_outputs()
        if (len(inputs) != 1 or len(inputs[0].shape) != 2 or inputs[0].shape[-1] != 1432
                or len(outputs[0].shape) != 2 or outputs[0].shape[-1] != 29):
            raise ValueError("Unexpected TWIST2 ONNX interface; expected [batch,1432] -> [batch,29]")
        self.input_name = inputs[0].name
        self.reference = DEFAULT_REFERENCE.copy()
        self.reset()

    def reset(self):
        """Clear controller memory only; do not alter any measured physics state."""
        self.history = deque((np.zeros(127, dtype=np.float64) for _ in range(10)), maxlen=10)
        self.last_action = np.zeros(29, dtype=np.float64)
        self.target = DEFAULT_JOINT_POS.copy()
        self.last_torque = np.zeros(29)
        self.last_observation = np.zeros((1, 1432), dtype=np.float32)
        self.physics_steps, self.solves, self.failures = 0, 0, 0
        self.solve_time_ms = self.max_solve_time_ms = self.max_torque_ratio = 0.
        self.fallen = False
        self.origin = self.data.qpos[:2].copy()

    def set_motion_reference(self, reference):
        values = np.asarray(reference, dtype=np.float64)
        if values.shape != (35,) or not np.all(np.isfinite(values)):
            raise ValueError("TWIST2 reference needs exactly 35 finite values")
        self.reference = values.copy()

    def hold(self):
        """Request the trained neutral standing reference, without resetting physics."""
        self.set_motion_reference(DEFAULT_REFERENCE)

    def _update(self):
        started = time.perf_counter()
        d = self.data
        velocity = d.qvel[self.vadr].copy()
        velocity[[4, 5, 10, 11]] = 0
        current = np.r_[self.reference, d.qvel[3:6] * .25,
                        quat_to_euler_wxyz(d.qpos[3:7])[:2],
                        d.qpos[self.qadr] - DEFAULT_JOINT_POS,
                        velocity * .05, self.last_action]
        # Oldest-to-newest history, excluding current until the vector is built.
        vector = np.r_[current, np.asarray(self.history).ravel(), self.reference]
        self.history.append(current.copy())
        self.last_observation = vector.astype(np.float32)[None, :]
        raw = np.asarray(self.policy.run(None, {self.input_name: self.last_observation})[0][0])
        if raw.shape != (29,) or not np.all(np.isfinite(raw)):
            self.failures += 1
            raise RuntimeError("TWIST2 policy produced invalid motor targets")
        self.last_action = raw.copy()  # Upstream history retains the unclipped output.
        self.target = DEFAULT_JOINT_POS + .5 * np.clip(raw, -10, 10)
        self.solve_time_ms = 1000 * (time.perf_counter() - started)
        self.max_solve_time_ms = max(self.max_solve_time_ms, self.solve_time_ms)
        self.solves += 1

    def step(self):
        m, d = self.model, self.data
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            self.failures += 1
            raise RuntimeError("TWIST2 physics state is non-finite")
        if self.physics_steps % self.decimation == 0:
            self._update()
        torque = self.kp * (self.target - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
        self.last_torque = np.clip(torque, self.torque_low, self.torque_high)
        d.ctrl[:] = self.last_torque
        allowed = np.maximum(np.abs(self.torque_low), np.abs(self.torque_high))
        self.max_torque_ratio = max(self.max_torque_ratio, float(np.max(np.abs(self.last_torque) / allowed)))
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        self.physics_steps += 1
        upright = float(d.xmat[self.pelvis].reshape(3, 3)[2, 2])
        self.fallen |= bool(d.qpos[2] < .45 or upright < math.cos(math.radians(50)))

    def status(self):
        from physics_benchmark import measure
        clearances = []
        for body in self.feet:
            heights = [float(self.data.geom_xpos[g, 2] - self.model.geom_size[g, 0])
                       for g in self.foot_geoms if self.model.geom_bodyid[g] == body]
            clearances.append(min(heights) if heights else None)
        measured = measure(self.model, self.data)
        return {"state": "FALLEN" if self.fallen else "TRACKING", "fallen": self.fallen,
                "solver_status": "TWIST2 general motion tracker", "qp_solves": self.solves,
                "qp_failures": self.failures, "solver_backend": "TWIST2 ONNX CPU",
                "solve_ms": self.solve_time_ms, "max_solve_ms": self.max_solve_time_ms,
                "max_torque_limit_ratio": self.max_torque_ratio, "limited_joints": [],
                "feet_planted": False, "kinematic_preview": False,
                "policy_frequency_hz": self.frequency, "physics_frequency_hz": 1000,
                "motion_reference": self.reference.tolist(), "foot_clearance_m": clearances,
                "root_xy_displacement_m": float(np.linalg.norm(self.data.qpos[:2] - self.origin)),
                **measured}
