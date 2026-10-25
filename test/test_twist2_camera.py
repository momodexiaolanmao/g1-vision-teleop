"""Physical camera-reference stepping, holds and calibrated joint-order tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twist2_camera import TwistCameraBalance
from twist2_tracking import DEFAULT_JOINT_POS, JOINT_NAMES


def make_balance():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/unitree_g1_twist2/scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return TwistCameraBalance(model, data)


ENTRIES = [{"rule": {"robot_joint": name}} for name in JOINT_NAMES]


class CameraReferenceTests(unittest.TestCase):
    def test_hidden_knee_at_calibration_reveals_without_false_offset(self):
        balance = make_balance()
        mask = np.ones(29, dtype=bool)
        mask[3] = False
        initial = np.zeros(29)
        initial[3] = .4  # JointRobot's invalid-channel fallback is not human data.
        balance.calibrate(initial, ENTRIES, mask)
        self.assertTrue(np.isnan(balance.human_neutral[3]))
        before = balance.data.qpos.copy()
        straight = np.zeros(29)
        balance.allow_tracking(straight, ENTRIES, np.ones(29, dtype=bool))
        balance.set_reference(straight, ENTRIES)
        self.assertEqual(balance.human_neutral[3], 0.)
        self.assertEqual(balance.goal_offset[3], 0.)
        np.testing.assert_array_equal(balance.data.qpos, before)

    def test_first_reliable_wrist_establishes_neutral_without_jump(self):
        balance = make_balance()
        mask = np.ones(29, dtype=bool)
        mask[20] = False
        balance.calibrate(np.zeros(29), ENTRIES, mask)
        wrist = np.zeros(29)
        wrist[20] = .7
        balance.allow_tracking(wrist, ENTRIES, np.ones(29, dtype=bool))
        balance.set_reference(wrist, ENTRIES)
        self.assertEqual(balance.goal_offset[20], 0.)
        wrist[20] = .9
        balance.allow_tracking(wrist, ENTRIES, np.ones(29, dtype=bool))
        balance.set_reference(wrist, ENTRIES)
        self.assertAlmostEqual(balance.goal_offset[20], .2)
        mask[20] = False
        wrist[20] = -1.  # Ignore a lost joint's robot-command fallback.
        balance.allow_tracking(wrist, ENTRIES, mask)
        balance.set_reference(wrist, ENTRIES)
        self.assertAlmostEqual(balance.goal_offset[20], .2)

    def test_lost_essential_leg_points_settle_leg_while_arm_continues(self):
        balance = make_balance()
        balance.calibrate(np.zeros(29), ENTRIES)
        mapped = np.zeros(29)
        mapped[[0, 3, 1, 7]] = [-.3, .45, .15, .15]
        balance.allow_tracking(mapped, ENTRIES, np.ones(29, dtype=bool))
        balance.set_reference(mapped, ENTRIES)
        for _ in range(500):
            balance.step()
        mask = np.ones(29, dtype=bool)
        mask[3] = False
        mapped[15] = -.3
        before = balance.data.qpos.copy()
        balance.allow_tracking(mapped, ENTRIES, mask)
        balance.set_reference(mapped, ENTRIES)
        np.testing.assert_array_equal(balance.data.qpos, before)
        np.testing.assert_array_equal(balance.goal_offset[:6], 0)
        self.assertAlmostEqual(balance.goal_offset[15], -.3)
        for _ in range(1500):
            balance.step()
        self.assertLess(float(np.max(abs(balance.offset[:6]))), 1e-5)
        self.assertFalse(balance.fallen)
        self.assertEqual(float(np.max(abs(balance.data.qfrc_applied))), 0.)
        self.assertEqual(float(np.max(abs(balance.data.xfrc_applied))), 0.)

    def test_calibration_hold_and_reset_do_not_reposition_the_robot(self):
        balance = make_balance()
        for _ in range(1000):
            balance.step()
        data = balance.data
        qpos, qvel, timestamp = data.qpos.copy(), data.qvel.copy(), data.time
        order = np.arange(29)[::-1]
        entries = [ENTRIES[i] for i in order]
        neutral = np.linspace(-.15, .15, 29)
        self.assertTrue(balance.calibrate(neutral[order], entries))
        np.testing.assert_allclose(balance.human_neutral, neutral)
        balance.allow_tracking()
        moved = neutral.copy()
        moved[3] += .5
        balance.set_reference(moved[order], entries)
        self.assertAlmostEqual(balance.goal_offset[3], .5)
        balance.hold()
        balance.set_reference(moved[order], entries)
        np.testing.assert_array_equal(balance.goal_offset, np.zeros(29))
        self.assertFalse(balance.tracking_enabled)
        balance.reset()
        self.assertIsNone(balance.human_neutral)
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        self.assertEqual(data.time, timestamp)

    def test_fk_height_sign_uses_the_reference_data_only(self):
        balance = make_balance()
        actual = balance.data.qpos.copy()
        balance.calibrate(np.zeros(29), ENTRIES)
        balance.allow_tracking()
        # Both knees and hips bend to a physically shorter support configuration.
        moved = np.zeros(29)
        moved[[0, 6]] = -.4
        moved[[3, 9]] = .8
        balance.set_reference(moved, ENTRIES)
        for _ in range(600):
            balance.step()
        self.assertLess(balance.reference[2], .8)
        self.assertGreaterEqual(balance.reference[2], .6)
        # FK's floating base is zero; measured physical height remains independent.
        self.assertEqual(balance.fk_data.qpos[2], 0)
        self.assertGreater(balance.data.qpos[2], .45)
        self.assertFalse(balance.fallen)
        self.assertEqual(actual.shape, balance.data.qpos.shape)

    def test_alternating_camera_leg_references_create_physical_foot_lifts(self):
        balance = make_balance()
        balance.calibrate(np.zeros(29), ENTRIES)
        samples, joint_positions = [], []
        maximum_ctrl_ratio = maximum_force_ratio = 0.
        limits = np.maximum(abs(balance.controller.torque_low), abs(balance.controller.torque_high))
        held_offset = None
        for step in range(20000):
            seconds = balance.data.time
            if step % 33 == 0:
                mapped = np.zeros(29)
                if 2 <= seconds < 14:
                    phase = int((seconds - 2) / .6)
                    hip, knee = (0, 3) if phase % 2 == 0 else (6, 9)
                    mapped[hip], mapped[knee] = -.65, 1.
                    # A physical one-leg raise also moves the support hip laterally.
                    # Both hip-roll references shift the pelvis toward the stance foot.
                    mapped[[1, 7]] = .25 if phase % 2 == 0 else -.25
                    # Gentle asymmetric arm movement exercises full-body reference input.
                    mapped[15] = -.25 * np.sin(1.3 * seconds)
                    mapped[22] = .25 * np.sin(1.3 * seconds)
                    balance.allow_tracking()
                elif seconds >= 14:
                    if held_offset is None:
                        held_offset = balance.offset.copy()
                    balance.hold()
                balance.set_reference(mapped, ENTRIES)
            balance.step()
            if step % 50 == 0:
                samples.append({"time": balance.data.time, **balance.status()})
                joint_positions.append(balance.data.qpos[balance.controller.qadr].copy())
            maximum_ctrl_ratio = max(maximum_ctrl_ratio, float(np.max(abs(balance.data.ctrl) / limits)))
            maximum_force_ratio = max(maximum_force_ratio, float(np.max(
                abs(balance.data.qfrc_actuator[balance.controller.vadr]) / limits)))
            self.assertTrue(np.all(np.isfinite(balance.data.qpos)))
            self.assertEqual(float(np.max(abs(balance.data.qfrc_applied))), 0.)
            self.assertEqual(float(np.max(abs(balance.data.xfrc_applied))), 0.)
        samples.append({"time": balance.data.time, **balance.status()})
        active = [s for s in samples if 3 <= s["time"] < 14]
        settled = [s for s in samples if s["time"] > 1]
        weight = float(balance.model.body_mass.sum() * 9.81)
        clearances = np.max([s["foot_clearance_m"] for s in active], axis=0)
        single = [s for s in active if sum(load > .15*weight for load in s["foot_loads_n"]) == 1]
        spans = np.degrees(np.ptp(joint_positions, axis=0))
        report = {
            "input": "synthetic mapped camera angles at approximately 30 Hz",
            "camera_reference_root_velocity": [0., 0., 0.],
            "requested_leg_offsets_rad": {"hip_pitch": -.65, "knee": 1.,
                                          "bilateral_support_hip_roll": .25},
            "max_leg_reference_offset_deg": 30.,
            "physics_seconds": float(balance.data.time),
            "alternating_reference_seconds": [2., 14.], "hold_seconds": [14., 20.],
            "left_right_clearance_m": clearances.tolist(),
            "single_support_fraction": len(single) / len(active),
            "minimum_height_m": min(s["height_m"] for s in samples),
            "maximum_tilt_deg": max(s["tilt_deg"] for s in samples),
            "maximum_other_floor_contacts": max(s["other_floor_contacts"] for s in samples),
            "loaded_support_fraction": sum(sum(s["foot_loads_n"]) > .5*weight for s in settled) / len(settled),
            "maximum_control_limit_ratio": maximum_ctrl_ratio,
            "maximum_actuator_force_limit_ratio": maximum_force_ratio,
            "final_root_xy_displacement_m": balance.status()["root_xy_displacement_m"],
            "joint_span_deg": spans.tolist(), "fallen": balance.fallen,
            "final_state": balance.status(), "samples": samples,
        }
        report["passed"] = bool(not balance.fallen and min(clearances) > .02
                                and report["single_support_fraction"] > .1
                                and report["maximum_other_floor_contacts"] == 0
                                and maximum_ctrl_ratio <= 1+1e-6
                                and maximum_force_ratio <= 1+1e-6)
        path = ROOT / "artifacts/unitree_g1_twist2_camera_validation.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "final_state")}))
        self.assertFalse(balance.fallen)
        self.assertTrue(report["passed"])
        self.assertFalse(balance.tracking_enabled)
        np.testing.assert_allclose(balance.offset, 0, atol=1e-8)
        self.assertEqual(balance.status()["state"], "HOLDING")
        self.assertGreater(float(np.max(abs(held_offset))), .1)

    def test_joint_robot_pose_mapping_bypasses_footstep_gate_and_holds(self):
        from joint_mapping import JointRobot
        from validate_unitree_g1 import pose
        profile = json.loads((ROOT / "unitree_g1_physics_mapping.json").read_text(encoding="utf-8"))
        profile["control_mode"] = "tracking"
        profile["tracking"] = {"reference_blend_seconds": .12, "max_leg_reference_offset_deg": 30}
        for rule in profile["rules"]:
            rule["actuator"] = rule["robot_joint"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            robot = JointRobot(ROOT / "robots/unitree_g1_twist2/scene.xml", path)
            self.assertIsNone(robot.footstep)
            self.assertTrue(robot.calibrate(pose()))
            samples = []
            for step in range(20000):
                seconds = robot.data.time
                if step % 33 == 0:
                    if seconds < 14:
                        overrides = {}
                        if seconds >= 2:
                            left = int((seconds-2)/.6) % 2 == 0
                            side = "left" if left else "right"
                            overrides = {side + "_hip_pitch": -.65, side + "_knee_flex": 1.,
                                         "left_hip_roll": .25 if left else -.25,
                                         "right_hip_roll": .25 if left else -.25}
                        targets = robot.targets_from_pose(pose(**overrides))
                        self.assertIsNotNone(targets)
                        robot.set_targets(targets)
                    else:
                        robot.hold()
                robot.step()
                if step % 50 == 0:
                    samples.append({"time": robot.data.time, **robot.balance.status()})
            active = [s for s in samples if 3 <= s["time"] < 14]
            clearances = np.max([s["foot_clearance_m"] for s in active], axis=0)
            report = {"input": "validate.pose -> JointRobot camera mapping -> TWIST2",
                      "footstep_gate_present": robot.footstep is not None,
                      "left_right_clearance_m": clearances.tolist(),
                      "minimum_height_m": min(s["height_m"] for s in samples),
                      "maximum_tilt_deg": max(s["tilt_deg"] for s in samples),
                      "fallen": robot.balance.fallen,
                      "final_camera_tracking_enabled": robot.balance.tracking_enabled,
                      "final_root_xy_displacement_m": robot.balance.status()["root_xy_displacement_m"],
                      "final_state": robot.balance.status()["state"],
                      "samples": samples}
            report["passed"] = bool(not robot.balance.fallen and min(clearances) > .02
                                    and max(s["other_floor_contacts"] for s in samples) == 0
                                    and max(s["external_force_max"] for s in samples) == 0
                                    and max(s["body_force_max"] for s in samples) == 0
                                    and not robot.balance.tracking_enabled
                                    and robot.balance.status()["state"] == "HOLDING")
            (ROOT / "artifacts/unitree_g1_twist2_camera_mapping_validation.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({key: value for key, value in report.items() if key != "samples"}))
            self.assertTrue(report["passed"])
            self.assertEqual(robot.balance.status()["state"], "HOLDING")
            self.assertFalse(robot.balance.tracking_enabled)


if __name__ == "__main__":
    unittest.main()
