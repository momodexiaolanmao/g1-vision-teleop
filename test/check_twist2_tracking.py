"""Headless TWIST2 physical walking validation, using the upstream reference.

The numerical observation convention follows Yanjie Ze's MIT-licensed TWIST2
(d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a). Reference data is loaded with an
allow-list unpickler; no general-purpose pickle.load/loads is used.
"""
import argparse
import io
import json
from pathlib import Path
import pickle
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twist2_tracking import (DEFAULT_REFERENCE, EXPECTED_MODEL_MASS_KG,
                            INITIAL_JOINT_POS, JOINT_NAMES, Twist2Tracking,
                            quat_to_euler_wxyz)


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Read the official numeric reference without permitting arbitrary globals."""
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name in ("_reconstruct", "scalar"):
                return getattr(np._core.multiarray, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return np._core.numeric._frombuffer
        raise pickle.UnpicklingError(f"Reference contains a forbidden global: {module}.{name}")


def quat_multiply_xyzw(a, b):
    return np.r_[a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3]),
                 a[3]*b[3] - np.dot(a[:3], b[:3])]


def inverse_rotate_xyzw(q, vector):
    v = np.asarray(vector)
    return (v * (2*q[3]**2 - 1) - np.cross(q[:3], v) * (2*q[3])
            + q[:3] * np.dot(q[:3], v) * 2)


def slerp_xyzw(a, b, fraction):
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        result = (1-fraction)*a + fraction*b
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1-fraction)*theta)*a + np.sin(fraction*theta)*b) / np.sin(theta)


def rotation_vector_xyzw(q):
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    if length < 1e-12:
        return 2*q[:3]
    return q[:3] * (2*np.arctan2(length, q[3]) / length)


class RecordedMotion:
    """Minimal NumPy equivalent of the upstream MotionLib reference calculation."""
    def __init__(self, path):
        with Path(path).open("rb") as stream:
            document = NumpyOnlyUnpickler(stream).load()
        if not isinstance(document, dict):
            raise ValueError("Reference must be a motion dictionary")
        self.fps = float(document["fps"])
        if not np.isfinite(self.fps) or not 0 < self.fps <= 1000:
            raise ValueError("Reference has an invalid frame rate")
        self.root_pos = np.asarray(document["root_pos"], dtype=np.float64)
        self.root_rot = np.asarray(document["root_rot"], dtype=np.float64)
        self.joint_pos = np.asarray(document["dof_pos"], dtype=np.float64)
        self.frames = len(self.root_pos)
        if (self.frames < 3 or self.root_pos.shape != (self.frames, 3)
                or self.root_rot.shape != (self.frames, 4)
                or self.joint_pos.shape != (self.frames, 29)
                or not all(np.all(np.isfinite(x)) for x in
                           (self.root_pos, self.root_rot, self.joint_pos))):
            raise ValueError("Reference needs finite root XYZ, XYZW rotation and 29 joint frames")
        norms = np.linalg.norm(self.root_rot, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("Reference contains a zero quaternion")
        self.root_rot = self.root_rot / norms[:, None]
        self.duration = (self.frames - 1) / self.fps
        self.dt = 1 / self.fps
        self.root_vel = np.gradient(self.root_pos, self.dt, axis=0)
        self.root_ang_vel = np.empty((self.frames, 3))
        for index in range(self.frames):
            previous, following = max(0, index-1), min(self.frames-1, index+1)
            conjugate = self.root_rot[previous].copy()
            conjugate[:3] *= -1
            relative = quat_multiply_xyzw(self.root_rot[following], conjugate)
            self.root_ang_vel[index] = rotation_vector_xyzw(relative) / ((following-previous)*self.dt)

    def frame_at(self, seconds):
        seconds = float(np.clip(seconds, 0, self.duration))
        frame = seconds * self.fps
        first = min(int(frame), self.frames-1)
        last = min(first+1, self.frames-1)
        fraction = frame-first
        root = (1-fraction)*self.root_pos[first] + fraction*self.root_pos[last]
        quat = slerp_xyzw(self.root_rot[first], self.root_rot[last], fraction)
        angles = (1-fraction)*self.joint_pos[first] + fraction*self.joint_pos[last]
        return root, quat, angles, first

    def reference_at(self, seconds):
        root, quat, angles, first = self.frame_at(seconds)
        euler = quat_to_euler_wxyz(quat[[3, 0, 1, 2]])
        return np.r_[inverse_rotate_xyzw(quat, self.root_vel[first])[:2], root[2],
                     euler[:2], inverse_rotate_xyzw(quat, self.root_ang_vel[first])[2], angles]


def validate_loader_rejects_code():
    # This tests the allow list without constructing or running an executable payload.
    forbidden = pickle.dumps(eval, protocol=4)
    try:
        NumpyOnlyUnpickler(io.BytesIO(forbidden)).load()
    except pickle.UnpicklingError:
        return True
    raise AssertionError("Reference loader permitted a non-NumPy global")


def run_case(model_path, duration, motion=None, frequency=100):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Initialization is the only place in this benchmark that sets physical pose.
    if motion is None:
        data.qpos[:] = np.r_[0., 0., .793, 1., 0., 0., 0., INITIAL_JOINT_POS]
    else:
        root, quat, angles, _ = motion.frame_at(0)
        data.qpos[:] = np.r_[root, quat[[3, 0, 1, 2]], angles]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    controller = Twist2Tracking(model, data, {"policy_frequency": frequency})
    initial = data.qpos[:3].copy()
    saved_qpos, saved_qvel, saved_time = data.qpos.copy(), data.qvel.copy(), data.time
    controller.set_motion_reference(DEFAULT_REFERENCE)
    controller.hold()
    controller.reset()
    np.testing.assert_array_equal(data.qpos, saved_qpos)
    np.testing.assert_array_equal(data.qvel, saved_qvel)
    assert data.time == saved_time
    samples, inference_times = [], []
    max_external = max_body_force = max_actual_force_ratio = max_other_contact = 0.
    max_control_ratio = 0.
    maximum_clearance = np.zeros(2)
    total_path = 0.
    previous_xy = data.qpos[:2].copy()
    started = time.perf_counter()
    steps = round(duration / model.opt.timestep)
    actual_limits = np.maximum(np.abs(controller.torque_low), np.abs(controller.torque_high))
    for step in range(steps):
        if step % controller.decimation == 0:
            reference = DEFAULT_REFERENCE if motion is None else motion.reference_at(data.time + .02)
            controller.set_motion_reference(reference)
        old_solves = controller.solves
        controller.step()
        if controller.solves != old_solves:
            inference_times.append(controller.solve_time_ms)
        max_external = max(max_external, float(np.max(np.abs(data.qfrc_applied))))
        max_body_force = max(max_body_force, float(np.max(np.abs(data.xfrc_applied))))
        max_control_ratio = max(max_control_ratio, float(np.max(np.abs(data.ctrl) / actual_limits)))
        max_actual_force_ratio = max(max_actual_force_ratio,
                                    float(np.max(np.abs(data.qfrc_actuator[controller.vadr]) / actual_limits)))
        total_path += float(np.linalg.norm(data.qpos[:2] - previous_xy))
        previous_xy = data.qpos[:2].copy()
        if step % 50 == 0 or step == steps-1:
            state = controller.status()
            samples.append({"time": float(data.time), **state})
            maximum_clearance = np.maximum(maximum_clearance, state["foot_clearance_m"])
            max_other_contact = max(max_other_contact, state["other_floor_contacts"])
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise AssertionError("Non-finite simulated robot state")
    elapsed = time.perf_counter() - started
    settled = [sample for sample in samples if sample["time"] >= 1]
    weight = float(model.body_mass.sum() * 9.81)
    one_foot_samples = [sample for sample in settled
                        if sum(load > .15*weight for load in sample["foot_loads_n"]) == 1]
    supported_samples = [sample for sample in settled if sum(sample["foot_loads_n"]) > .5*weight]
    result = {
        "case": "constant_standing_reference" if motion is None else "upstream_recorded_walking_reference",
        "model_path": str(model_path.resolve()),
        "simulated_seconds": float(data.time), "wall_seconds": elapsed,
        "model_mass_kg": float(model.body_mass.sum()), "gravity": model.opt.gravity.tolist(),
        "initial_root_xyz": initial.tolist(), "final_root_xyz": data.qpos[:3].tolist(),
        "root_xy_displacement_m": float(np.linalg.norm(data.qpos[:2] - initial[:2])),
        "root_xy_path_length_m": total_path,
        "minimum_height_m": min(sample["height_m"] for sample in samples),
        "maximum_tilt_deg": max(sample["tilt_deg"] for sample in samples),
        "maximum_foot_clearance_m": maximum_clearance.tolist(),
        "single_support_fraction": len(one_foot_samples) / max(1, len(settled)),
        "loaded_support_fraction": len(supported_samples) / max(1, len(settled)),
        "maximum_other_floor_contacts": int(max_other_contact),
        "maximum_external_generalized_force": max_external,
        "maximum_external_body_force": max_body_force,
        "maximum_control_limit_ratio": max_control_ratio,
        "maximum_actuator_force_limit_ratio": max_actual_force_ratio,
        "policy_frequency_hz": frequency, "policy_inferences": controller.solves,
        "inference_ms_median": float(np.median(inference_times)),
        "inference_ms_p95": float(np.percentile(inference_times, 95)),
        "fallen": controller.fallen, "policy_failures": controller.failures,
        "samples": samples,
    }
    if motion is not None:
        result["reference_frames"] = motion.frames
        result["reference_fps"] = motion.fps
        result["reference_duration_s"] = motion.duration
        result["reference_root_xy_displacement_m"] = float(np.linalg.norm(
            motion.frame_at(duration)[0][:2] - motion.root_pos[0, :2]))
    result["passed"] = bool(
        not controller.fallen and not controller.failures and max_external == 0
        and max_body_force == 0 and max_control_ratio <= 1+1e-6
        and max_actual_force_ratio <= 1+1e-6 and max_other_contact == 0
        and result["minimum_height_m"] > .45 and result["maximum_tilt_deg"] < 50
        and result["loaded_support_fraction"] > .85
        and (motion is None or (result["root_xy_displacement_m"] > .5
                                and max(maximum_clearance) > .02
                                and result["single_support_fraction"] > .1)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "robots/unitree_g1_twist2/scene.xml")
    parser.add_argument("--motion", type=Path, default=ROOT / "fixtures/twist2_walk.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/unitree_g1_twist2_validation.json")
    parser.add_argument("--standing-seconds", type=float, default=10)
    parser.add_argument("--walking-seconds", type=float, default=None)
    parser.add_argument("--policy-frequency", type=int, default=100)
    args = parser.parse_args()
    assert validate_loader_rejects_code()
    motion = RecordedMotion(args.motion)
    walking_seconds = min(30., motion.duration-.02) if args.walking_seconds is None else args.walking_seconds
    if walking_seconds <= 0 or args.standing_seconds <= 0:
        parser.error("Validation durations must be positive")
    cases = [run_case(args.model, args.standing_seconds, frequency=args.policy_frequency),
             run_case(args.model, walking_seconds, motion, args.policy_frequency)]
    report = {"controller": "TWIST2 general motion tracker, NumPy/ONNX CPU",
              "upstream_commit": "d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a",
              "expected_model_mass_kg": EXPECTED_MODEL_MASS_KG,
              "joint_order": list(JOINT_NAMES), "restricted_reference_loader": True,
              "passed": all(case["passed"] for case in cases), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for case in cases:
        print(json.dumps({key: value for key, value in case.items() if key != "samples"}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
