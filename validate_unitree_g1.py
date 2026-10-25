"""Exercise full-body geometry, root motion, holds and the actual G1 model."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import time

import cv2
import mujoco
import numpy as np

from joint_mapping import JointRobot
from pose_retarget import BodyRetarget, IDS, rotate, CHANNEL_LABELS
from tracking_session import TrackingSession
from teleop import write_live_status

ROOT = Path(__file__).absolute().parent
A = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])


def pose(seconds=0, dancing=False, turn=0, lateral=0, jump=0, depth_scale=1, **overrides):
    """A small ideal anatomical rig, not recorded camera data."""
    params = {}
    if dancing:
        params = {"waist_yaw": 0.32 * math.sin(1.1 * seconds), "waist_roll": 0.18 * math.sin(1.5 * seconds),
                  "waist_pitch": 0.17 * math.cos(1.2 * seconds)}
        for sign, side in ((1, "left"), (-1, "right")):
            phase = seconds + sign * 0.7
            for key, value in (("shoulder_pitch", -0.6 - 0.65 * math.sin(phase)),
                               ("shoulder_roll", sign * (0.3 + 0.3 * math.sin(phase * 1.3))),
                               ("shoulder_yaw", 0.35 * math.sin(phase * 0.8)),
                               ("elbow_flex", 0.65 + 0.45 * math.sin(phase * 1.4)),
                               ("wrist_roll", 0.45 * math.sin(phase * 1.2)),
                               ("wrist_pitch", 0.25 * math.cos(phase * 1.6)),
                               ("wrist_yaw", 0.25 * math.sin(phase * 1.8)),
                               ("hip_pitch", -0.25 + 0.25 * math.sin(phase * 1.4)),
                               ("hip_roll", sign * (0.12 + 0.1 * math.cos(phase))),
                               ("hip_yaw", 0.2 * math.sin(phase)),
                               ("knee_flex", 0.5 + 0.35 * math.sin(phase * 1.4)),
                               ("ankle_pitch", 0.15 * math.sin(phase * 1.5)),
                               ("ankle_roll", 0.12 * math.cos(phase * 1.3))):
                params[f"{side}_{key}"] = value
    params.update(overrides)
    v = lambda key: params.get(key, 0.0)
    waist = rotate("z", v("waist_yaw")) @ rotate("x", v("waist_roll")) @ rotate("y", v("waist_pitch"))
    positions = {}
    for sign, side in ((1, "left"), (-1, "right")):
        hip = np.array([0., sign * 0.12, 0.])
        shoulder = waist @ np.array([0., sign * 0.21, 0.52])
        arm = waist @ rotate("y", v(f"{side}_shoulder_pitch")) @ rotate("x", v(f"{side}_shoulder_roll")) @ rotate("z", v(f"{side}_shoulder_yaw"))
        elbow = shoulder + arm @ np.array([0., 0., -0.27])
        forearm = arm @ rotate("y", -v(f"{side}_elbow_flex"))
        wrist = elbow + forearm @ np.array([0., 0., -0.24])
        wrist_frame = forearm @ rotate("y", math.pi / 2)
        hand = wrist_frame @ rotate("x", v(f"{side}_wrist_roll")) @ rotate("y", v(f"{side}_wrist_pitch")) @ rotate("z", v(f"{side}_wrist_yaw"))
        leg = rotate("y", v(f"{side}_hip_pitch")) @ rotate("x", v(f"{side}_hip_roll")) @ rotate("z", v(f"{side}_hip_yaw"))
        knee = hip + leg @ np.array([0., 0., -0.37])
        shank = leg @ rotate("y", v(f"{side}_knee_flex"))
        ankle = knee + shank @ np.array([0., 0., -0.36])
        foot = shank @ rotate("y", v(f"{side}_ankle_pitch")) @ rotate("x", v(f"{side}_ankle_roll"))
        for name, point in (("hip", hip), ("shoulder", shoulder), ("elbow", elbow), ("wrist", wrist),
                            ("index", wrist + hand @ np.array([0.11, 0.025, 0.])),
                            ("pinky", wrist + hand @ np.array([0.11, -0.025, 0.])),
                            ("thumb", wrist + hand @ np.array([0.07, 0.04, 0.015])),
                            ("knee", knee), ("ankle", ankle),
                            ("heel", ankle + foot @ np.array([-0.04, 0., -0.06])),
                            ("foot_index", ankle + foot @ np.array([0.15, 0., -0.06]))):
            positions[f"{side}_{name}"] = point
    for name in IDS:
        if name not in positions:
            positions[name] = waist @ np.array([0.06, 0.02 if "left" in name else -0.02, 0.72])
    world, image = [], []
    root_rotation = rotate("z", turn)
    for name in IDS:
        point = A @ root_rotation @ positions[name]
        world.append(SimpleNamespace(x=point[0], y=point[1], z=point[2], visibility=1., presence=1.))
        image.append(SimpleNamespace(x=0.5 + (point[0] * depth_scale + lateral) / 2.2 / (1280 / 720),
                                     y=0.5 + (point[1] * depth_scale - jump) / 2.2,
                                     z=point[2] / 2.2, visibility=1., presence=1.))
    return SimpleNamespace(landmarks=image, world_landmarks=world, arrived=time.monotonic(),
                           frame=np.zeros((720, 1280, 3), dtype=np.uint8))


def robot():
    return JointRobot(ROOT / "robots/unitree_g1/scene.xml", ROOT / "unitree_g1_mapping.json")


class RetargetTests(unittest.TestCase):
    def setUp(self):
        self.mapper = BodyRetarget()
        self.assertTrue(self.mapper.calibrate(pose()))

    def test_all_29_channels_recover_bent_full_body_pose(self):
        channels, root = self.mapper.evaluate(pose(0.7, True))
        self.assertEqual(set(channels), set(CHANNEL_LABELS))
        self.assertTrue(all(c["valid"] for c in channels.values()), channels)
        self.assertTrue(root["valid"])

    def test_shoulder_knee_and_elbow_directions(self):
        channels, _ = self.mapper.evaluate(pose(left_shoulder_pitch=-math.pi / 3,
            left_shoulder_roll=0.4, right_shoulder_roll=-0.4, left_elbow_flex=math.pi / 2,
            left_hip_pitch=-0.7, left_knee_flex=0.8))
        for channel, expected in (("left_shoulder_pitch", -60), ("left_shoulder_roll", math.degrees(.4)),
                                  ("right_shoulder_roll", -math.degrees(.4)), ("left_elbow_flex", 90),
                                  ("left_hip_pitch", math.degrees(-.7)), ("left_knee_flex", math.degrees(.8))):
            self.assertAlmostEqual(channels[channel]["angle"], expected, places=6)

    def test_waist_and_wrist_rotations(self):
        channels, _ = self.mapper.evaluate(pose(waist_yaw=.3, waist_roll=.2, waist_pitch=-.25,
            left_elbow_flex=.5, left_wrist_roll=.4, left_wrist_pitch=.2, left_wrist_yaw=-.3))
        for channel, expected in (("waist_yaw", .3), ("waist_roll", .2), ("waist_pitch", -.25),
                                  ("left_wrist_roll", .4), ("left_wrist_pitch", .2), ("left_wrist_yaw", -.3)):
            self.assertAlmostEqual(channels[channel]["angle"], math.degrees(expected), places=5)

    def test_global_turn_preserves_internal_joint_angles(self):
        before, _ = self.mapper.evaluate(pose(.7, True))
        after, root = self.mapper.evaluate(pose(.7, True, turn=2.1))
        self.assertAlmostEqual(root["yaw_rad"], 2.1)
        for name in before:
            self.assertAlmostEqual(before[name]["angle"], after[name]["angle"], places=5)

    def test_body_position_and_jump_are_relative_to_calibration(self):
        _, root = self.mapper.evaluate(pose(lateral=.3, jump=.2, depth_scale=1.1))
        self.assertGreater(root["lateral_m"], .1)
        self.assertGreater(root["vertical_m"], .1)
        self.assertLess(root["scale_ratio"], 1)

    def test_straight_arm_yaw_is_held_not_fabricated(self):
        channels, _ = self.mapper.evaluate(pose())
        self.assertFalse(channels["left_shoulder_yaw"]["valid"])
        self.assertTrue(channels["left_elbow_flex"]["valid"])

    def test_hidden_hips_do_not_disable_visible_elbow(self):
        observation = pose(.4, True)
        for name in ("left_hip", "right_hip"):
            observation.landmarks[IDS[name]].visibility = 0.1
        channels, root = self.mapper.evaluate(observation)
        self.assertTrue(channels["left_elbow_flex"]["valid"])
        self.assertTrue(channels["left_shoulder_pitch"]["valid"])
        self.assertFalse(channels["waist_yaw"]["valid"])
        self.assertFalse(root["valid"])

    def test_degenerate_hand_plane_only_holds_wrist(self):
        observation = pose(.4, True)
        observation.world_landmarks[IDS["left_pinky"]] = copy.copy(observation.world_landmarks[IDS["left_index"]])
        channels, _ = self.mapper.evaluate(observation)
        self.assertFalse(channels["left_wrist_roll"]["valid"])
        self.assertTrue(channels["left_elbow_flex"]["valid"])

    def test_implausible_hip_twist_is_held(self):
        channels, _ = self.mapper.evaluate(pose(left_hip_yaw=math.radians(175), left_knee_flex=.5))
        self.assertFalse(channels["left_hip_yaw"]["valid"])
        self.assertEqual(channels["left_hip_yaw"]["reason"], "ambiguous rotation; HOLD")
        self.assertTrue(channels["left_knee_flex"]["valid"])

    def test_offscreen_foot_with_high_confidence_is_held(self):
        observation = pose(.4, True)
        observation.landmarks[IDS["left_foot_index"]].y = 1.1
        channels, _ = self.mapper.evaluate(observation)
        self.assertFalse(channels["left_ankle_pitch"]["valid"])
        self.assertTrue(channels["right_ankle_pitch"]["valid"])


class G1Tests(unittest.TestCase):
    def test_calibrated_live_status_serializes_numpy_geometry(self):
        bot = robot()
        observation = pose(.5, True, lateral=.2, turn=.6)
        bot.calibrate(pose())
        bot.set_targets(bot.targets_from_pose(observation))
        bot.step()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "status.json"
            write_live_status(output, bot, TrackingSession(enabled=True, ready=True), observation, True)
            status = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(status["calibrated"])
            self.assertTrue(status["root_tracking"]["valid"])
            self.assertEqual(len(status["actual_deg"]), 29)

    def test_full_model_shadow_and_limits(self):
        bot = robot()
        self.assertEqual((bot.model.nq, bot.model.nv, bot.model.nu, len(bot.entries)), (36, 35, 29, 29))
        self.assertTrue(bot.calibrate(pose()))
        targets = bot.targets_from_pose(pose(.5, True))
        bot.set_targets(targets)
        for _ in range(1000):
            bot.step()
        np.testing.assert_allclose(bot.command, targets, atol=1e-5)
        for value, entry in zip(bot.command, bot.entries):
            self.assertTrue(entry["limits"][0] <= value <= entry["limits"][1])
            self.assertAlmostEqual(value, bot.data.qpos[entry["qpos"]])
        self.assertTrue(np.isfinite(bot.data.qpos).all())

    def test_pause_freezes_joints_and_root_cancelling_old_goals(self):
        bot = robot()
        bot.calibrate(pose())
        bot.set_targets(bot.targets_from_pose(pose(.5, True, lateral=.5, jump=.3, turn=1.2)))
        for _ in range(40):
            bot.step()
        bot.hold()
        snapshot = bot.data.qpos.copy()
        for _ in range(100):
            bot.step()
        np.testing.assert_allclose(bot.data.qpos, snapshot, atol=1e-10)

    def test_occluded_ankle_holds_only_affected_leg_channels(self):
        bot = robot()
        bot.calibrate(pose())
        bot.set_targets(bot.targets_from_pose(pose(.5, True)))
        for _ in range(30):
            bot.step()
        old = bot.command.copy()
        observation = pose(1.3, True)
        observation.landmarks[IDS["left_ankle"]].presence = .1
        targets = bot.targets_from_pose(observation)
        for i, diagnostic in enumerate(bot.pose_diagnostics):
            if diagnostic["joint"] == "left_knee_joint":
                self.assertFalse(diagnostic["valid"])
                self.assertEqual(targets[i], old[i])
            if diagnostic["joint"] == "right_knee_joint":
                self.assertTrue(diagnostic["valid"])
                self.assertNotAlmostEqual(targets[i], old[i])


def render_validation():
    bot = robot()
    bot.calibrate(pose())
    low, high = bot.command.copy(), bot.command.copy()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .7]
    camera.distance, camera.azimuth, camera.elevation = 3.2, 145, -10
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    visual_options = mujoco.MjvOption()
    visual_options.geomgroup[0] = 0
    # Three distinct poses prove the real model is drawn with upper/lower body
    # targets, plus a deterministic video for inspecting the mapping visually.
    writer = cv2.VideoWriter(str(output / "unitree_g1_shadow_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (960, 720))
    if not writer.isOpened():
        raise RuntimeError("Could not create validation video")
    try:
        with mujoco.Renderer(bot.model, height=720, width=960) as renderer:
            for frame in range(160):
                t = frame / 20
                observation = pose(t, True, turn=.4 * math.sin(t * .6), lateral=.15 * math.sin(t), jump=.06 * max(0, math.sin(t * 2)))
                bot.set_targets(bot.targets_from_pose(observation))
                for _ in range(25):
                    bot.step()
                low, high = np.minimum(low, bot.command), np.maximum(high, bot.command)
                renderer.update_scene(bot.data, camera=camera, scene_option=visual_options)
                image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(image, "UNITREE G1 | synthetic full-body shadow test", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 244, 250), 1, cv2.LINE_AA)
                cv2.putText(image, "29 joints + moving root | kinematic preview", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (130, 240, 190), 1, cv2.LINE_AA)
                writer.write(image)
                if frame in (35, 80, 135):
                    cv2.imwrite(str(output / f"unitree_g1_pose_{frame}.png"), image)
    finally:
        writer.release()
    report = {"model": "Official Unitree G1 29dof", "mode": "kinematic shadow preview", "synthetic_input": True,
              "simulated_seconds": bot.data.time, "mapped_joints": len(bot.entries),
              "moving_joints": int(sum(np.degrees(high - low) > 2)),
              "motion_span_deg": {e["rule"]["robot_joint"]: float(np.degrees(h - l))
                                  for e, l, h in zip(bot.entries, low, high)},
              "unit_tests": "passed", "physical_balance": "not evaluated in kinematic mode"}
    (output / "unitree_g1_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    render_validation()
