"""Regression checks for visible arms being blocked by unrelated hidden hips."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import threading

import numpy as np

from joint_mapping import JointRobot, LANDMARK_IDS
from tracking_session import TrackingSession
from teleop import CameraPose, Observation
from subject_tracker import SubjectTracker

ROOT = Path(__file__).resolve().parent


def visible_pose():
    points = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0)
              for _ in range(33)]
    positions = {
        "left_shoulder": [-0.2, 0, 0], "left_hip": [-0.2, 0.5, 0],
        "left_elbow": [-0.2 - np.sqrt(3) / 4, 0.25, 0],
        "right_shoulder": [0.2, 0, 0], "right_hip": [0.2, 0.5, 0],
        "right_elbow": [0.5, 0, 0], "right_wrist": [0.5, 0.3, 0],
    }
    for name, values in positions.items():
        point = points[LANDMARK_IDS[name]]
        point.x, point.y, point.z = values
    return SimpleNamespace(landmarks=points, world_landmarks=points,
                           frame=np.zeros((480, 640, 3), dtype=np.uint8))


class TrackingInputTests(unittest.TestCase):
    def setUp(self):
        self.robot = JointRobot(ROOT / "arm.xml", ROOT / "joint_mapping.json")

    def test_hidden_hips_do_not_block_visible_right_elbow(self):
        observation = visible_pose()
        for name in ("left_hip", "right_hip"):
            observation.landmarks[LANDMARK_IDS[name]].visibility = 0.1
        targets = self.robot.targets_from_pose(observation)
        self.assertIsNotNone(targets, "Visible elbow must remain controllable when hips are out of view")
        np.testing.assert_allclose(targets[:2], self.robot.command[:2])
        self.assertAlmostEqual(np.degrees(targets[2]), 72.0)
        self.robot.set_targets(targets)
        for _ in range(1500):
            self.robot.step()
        self.assertAlmostEqual(np.degrees(self.robot.data.qpos[2]), 72.0, delta=0.6)

    def test_occluded_wrist_only_holds_elbow(self):
        observation = visible_pose()
        observation.landmarks[LANDMARK_IDS["right_wrist"]].visibility = 0.1
        targets = self.robot.targets_from_pose(observation)
        self.assertIsNotNone(targets, "Visible shoulders must remain controllable")
        self.assertAlmostEqual(targets[2], self.robot.command[2])
        np.testing.assert_allclose(np.degrees(targets[:2]), [0, -70], atol=1e-9)

    def test_all_hidden_returns_no_input(self):
        observation = visible_pose()
        for point in observation.landmarks:
            point.visibility = 0.1
        self.assertIsNone(self.robot.targets_from_pose(observation))

    def test_held_joint_cancels_previous_moving_goal(self):
        observation = visible_pose()
        self.robot.set_targets(self.robot.targets_from_pose(observation))
        for _ in range(20):
            self.robot.step()
        held = self.robot.command[2]
        observation.landmarks[LANDMARK_IDS["right_wrist"]].presence = 0.1
        targets = self.robot.targets_from_pose(observation)
        self.assertIsNotNone(targets)
        self.robot.set_targets(targets)
        for _ in range(100):
            self.robot.step()
        self.assertAlmostEqual(self.robot.command[2], held)


class TrackingSessionTests(unittest.TestCase):
    def test_recalibration_waits_for_full_pose_even_when_arm_is_visible(self):
        session = TrackingSession(enabled=True, ready=True)
        session.request_start()
        self.assertEqual(session.advance(True, 0., start_ready=False), (False, False))
        self.assertTrue(session.start_pending)
        self.assertFalse(session.enabled)
        self.assertEqual(session.advance(True, .1, start_ready=True), (True, True))

    def test_c_pressed_before_pose_is_ready_is_remembered(self):
        session = TrackingSession()
        session.request_start()
        self.assertEqual(session.advance(False, 0.0), (False, False))
        self.assertTrue(session.start_pending)
        self.assertEqual(session.advance(True, 0.1), (True, True))
        self.assertFalse(session.start_pending)

    def test_one_missing_frame_holds_then_recovers(self):
        session = TrackingSession(enabled=True, ready=True)
        self.assertEqual(session.advance(False, 0.0), (False, False))
        self.assertEqual(session.advance(True, 0.03), (False, True))
        self.assertFalse(session.lost_latched)

    def test_persistent_loss_holds_and_requires_explicit_restart(self):
        session = TrackingSession(enabled=True, ready=True)
        session.advance(False, 0.0, stream_fresh=False)
        self.assertEqual(session.advance(False, 0.8, stream_fresh=False), (False, False))
        self.assertTrue(session.lost_latched)
        self.assertEqual(session.advance(True, 0.9), (False, False))
        session.request_start()
        self.assertEqual(session.advance(True, 1.0), (True, True))

    def test_visible_pose_recovers_after_long_occlusion_with_live_frames(self):
        session = TrackingSession(enabled=True, ready=True)
        self.assertEqual(session.advance(False, 0.0), (False, False))
        self.assertEqual(session.advance(False, 2.0), (False, False))
        self.assertEqual(session.advance(True, 2.1), (False, True))
        self.assertTrue(session.enabled)
        self.assertFalse(session.lost_latched)

    def test_pause_cancels_pending_start(self):
        session = TrackingSession()
        session.request_start()
        session.toggle_pause(False)
        self.assertEqual(session.advance(True, 0.0), (False, False))
        self.assertFalse(session.enabled)

    def test_mapping_controls_visible_elbow_after_start_with_hidden_hips(self):
        robot = JointRobot(ROOT / "arm.xml", ROOT / "joint_mapping.json")
        session = TrackingSession()
        observation = visible_pose()
        for name in ("left_hip", "right_hip"):
            observation.landmarks[LANDMARK_IDS[name]].presence = 0.1
        session.request_start()
        targets = robot.targets_from_pose(observation)
        restart, can_track = session.advance(targets is not None, 0.0)
        self.assertTrue(restart and can_track)
        robot.reset()
        targets = robot.targets_from_pose(observation)
        robot.set_targets(targets)
        for _ in range(1500):
            robot.step()
        self.assertAlmostEqual(np.degrees(robot.data.qpos[2]), 72.0, delta=0.6)


class CameraSubjectRequestTests(unittest.TestCase):
    def setUp(self):
        self.source=CameraPose.__new__(CameraPose)
        self.source.lock=threading.Lock()
        self.source.subject=SubjectTracker()
        self.source.subject_revision=0
        self.source.subject_request=None
        self.source.observation=None
        self.source.error=''

    def test_reacquire_supersedes_queued_lock(self):
        self.source.subject_request=(0, (np.array([.5,.3]), .2, .1))
        self.source.reacquire_subject()
        _, revision=self.source._select_subject([], 1.)
        self.assertFalse(self.source.subject.locked)
        self.assertEqual(revision,1)

    def test_late_provisional_frame_cannot_publish_after_reacquire(self):
        old=Observation(0.,0.,None,None,[],subject_revision=0)
        self.source.reacquire_subject()
        self.source._publish_observation(old)
        self.assertIsNone(self.source.latest()[0])
        fresh=Observation(.1,.1,None,None,[],subject_revision=1)
        self.source._publish_observation(fresh)
        self.assertIs(self.source.latest()[0],fresh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
