"""Regression: gravity, real foot support, bounded motors and pose following."""
import json
from pathlib import Path
import time
import unittest

import mujoco
import numpy as np

from joint_mapping import JointRobot
from physics_benchmark import measure
from validate_unitree_g1 import pose

ROOT = Path(__file__).absolute().parent


class StandingTests(unittest.TestCase):
    def robot(self):
        return JointRobot(ROOT / "robots/unitree_g1/scene_physics.xml",
                          ROOT / "unitree_g1_physics_mapping.json")

    def test_stands_and_follows_with_real_ground_support(self):
        robot = self.robot()
        self.assertTrue(robot.calibrate(pose()))
        self.assertEqual(robot.model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(robot.model.neq, 0)
        self.assertEqual(robot.model.nmocap, 0)
        np.testing.assert_allclose(robot.model.opt.gravity, [0, 0, -9.81])
        records, joint_samples = [], []
        started = time.perf_counter()
        for step in range(15000):
            seconds = step * robot.model.opt.timestep
            if step % 15 == 0:
                observation = pose(max(0, seconds - 4), dancing=seconds >= 4)
                targets = robot.targets_from_pose(observation)
                if targets is not None:
                    robot.set_targets(targets)
            robot.step()
            if step % 50 == 0:
                records.append({"time": seconds, **measure(robot.model, robot.data)})
                joint_samples.append(robot.data.qpos[7:].copy())
            self.assertTrue(np.all(np.isfinite(robot.data.qpos)))
            self.assertLess(np.max(np.abs(robot.data.qfrc_applied)), 1e-12)
            self.assertLess(np.max(np.abs(robot.data.xfrc_applied)), 1e-12)
            self.assertTrue(np.all(robot.data.ctrl <= robot.model.actuator_ctrlrange[:, 1] + 1e-6))
            self.assertTrue(np.all(robot.data.ctrl >= robot.model.actuator_ctrlrange[:, 0] - 1e-6))
        report = {"duration_s": robot.data.time, "wall_s": time.perf_counter() - started,
                  "controller": robot.balance.status(), "samples": records,
                  "joint_span_deg": np.degrees(np.ptp(joint_samples, axis=0)).tolist()}
        (ROOT / "artifacts/unitree_g1_physics_validation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        settled = [r for r in records if r["time"] > 2]
        self.assertGreater(min(r["height_m"] for r in settled), 0.60)
        self.assertLess(max(r["tilt_deg"] for r in settled), 12)
        self.assertEqual(max(r["other_floor_contacts"] for r in settled), 0)
        weight = robot.model.body_mass.sum() * 9.81
        self.assertGreater(sum(sum(r["foot_loads_n"]) > 0.65 * weight for r in settled) / len(settled), 0.95)
        spans = np.degrees(np.ptp(joint_samples, axis=0))
        self.assertGreater(float(np.max(spans[15:])), 30)
        self.assertGreater(float(np.max(spans[:12])), 5)

    def test_input_hold_keeps_balance_and_root_is_not_reset(self):
        robot = self.robot()
        for step in range(3000):
            if step > 500:
                robot.hold()
            robot.step()
        before = robot.data.qpos[:7].copy()
        before_time = robot.data.time
        self.assertTrue(robot.calibrate(pose()))
        np.testing.assert_array_equal(robot.data.qpos[:7], before)
        self.assertEqual(robot.data.time, before_time)
        result = measure(robot.model, robot.data)
        self.assertGreater(result["height_m"], 0.65)
        self.assertLess(result["tilt_deg"], 8)
        self.assertGreater(sum(result["foot_loads_n"]), 250)
        self.assertFalse(robot.balance.fallen)

    def test_gravity_really_takes_over_when_motors_are_disabled(self):
        robot = self.robot()
        for _ in range(1500):
            robot.step()
        standing_height = float(robot.data.qpos[2])
        robot.data.ctrl[:] = 0
        for _ in range(1500):
            mujoco.mj_step(robot.model, robot.data)
        self.assertLess(robot.data.qpos[2], standing_height - 0.25)
        self.assertEqual(np.max(np.abs(robot.data.qfrc_applied)), 0)

    def test_large_upper_body_motion_and_small_push_recover(self):
        robot = self.robot()
        robot.calibrate(pose())
        snapshots = []
        stages = [
            {},
            {"left_shoulder_roll": 1.35, "right_shoulder_roll": -1.35},
            {"left_shoulder_pitch": -2.5, "right_shoulder_pitch": -2.5,
             "left_elbow_flex": 0.4, "right_elbow_flex": 0.4},
            {"waist_yaw": 0.9, "waist_roll": 0.35, "waist_pitch": 0.4,
             "left_knee_flex": 1.5, "right_knee_flex": 1.5,
             "left_hip_pitch": -0.75, "right_hip_pitch": -0.75},
            {"left_knee_flex": 1.4, "left_hip_pitch": -0.8},
            {},
        ]
        for step in range(15000):
            seconds = step * robot.model.opt.timestep
            if step % 15 == 0:
                targets = robot.targets_from_pose(pose(**stages[min(5, int(seconds // 5))]))
                if targets is not None:
                    robot.set_targets(targets)
            if 26 <= seconds < 26.1:
                robot.data.xfrc_applied[robot.balance.pelvis, 0] = 20
            else:
                robot.data.xfrc_applied[:] = 0
            robot.step()
            if step % 50 == 0:
                snapshots.append({"time": seconds, **measure(robot.model, robot.data)})
        (ROOT / "artifacts/unitree_g1_physics_stress.json").write_text(json.dumps({
            "duration_s": robot.data.time, "intentional_push": "20 N for 0.1 s at pelvis",
            "controller": robot.balance.status(), "samples": snapshots,
        }, indent=2), encoding="utf-8")
        self.assertFalse(robot.balance.fallen)
        self.assertLess(max(s["tilt_deg"] for s in snapshots), 12)
        self.assertGreater(min(s["height_m"] for s in snapshots), 0.60)
        self.assertEqual(max(s["other_floor_contacts"] for s in snapshots), 0)
        self.assertEqual(robot.balance.failures, 0)


if __name__ == "__main__":
    unittest.main()
