"""Check the hands-free sampling workflow without a camera or GUI."""
import json
from pathlib import Path
import tempfile
import unittest

from check_live_tracking import build_report
from sampling_session import SamplingSession


def snapshot(timestamp, angle=0):
    return {
        "updated_at_unix": timestamp,
        "enabled": True,
        "input_valid": True,
        "state": "TRACKING (custom joint mapping)",
        "robot_model": "test robot",
        "frame_age_ms": 30,
        "rules": [{"valid": True, "joint": "left_elbow_joint",
                   "human_angle_deg": angle, "target_deg": angle}],
        "command_deg": [angle],
        "actual_deg": [angle],
    }


class SamplingSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "reports" / "sample.json"
        self.session = SamplingSession(duration=20, output=self.output)

    def test_waits_for_calibration_without_collecting_or_writing(self):
        self.assertEqual(self.session.state, "WAITING_CALIBRATION")
        self.assertIn("walk into camera", self.session.display(100))
        self.assertFalse(self.session.update(100, snapshot(100)))
        self.assertEqual(self.session.samples, [])
        self.assertFalse(self.output.exists())

    def test_three_seconds_settling_is_outside_twenty_second_sample_window(self):
        self.assertTrue(self.session.start(10))
        self.assertEqual(self.session.display(10),
                         "CALIBRATED: stand still - sampling starts in 3s")
        self.assertFalse(self.session.update(12.999, snapshot(1)))
        self.assertEqual(self.session.samples, [])
        self.assertFalse(self.session.update(13, snapshot(2)))
        self.assertEqual(self.session.state, "SAMPLING")
        self.assertEqual(self.session.display(13),
                         "SAMPLING: stand still - 20s remaining")
        self.assertFalse(self.session.update(32.999, snapshot(3)))
        self.assertEqual(self.session.display(32.999),
                         "SAMPLING: stand still - 1s remaining")
        self.assertTrue(self.session.update(33, snapshot(4)))
        self.assertEqual([s["updated_at_unix"] for s in self.session.samples], [2, 3])
        self.assertEqual(self.session.report["seconds"], 20)

    def test_only_unique_snapshots_in_the_active_window_are_retained(self):
        self.session.start(0)
        self.session.update(2, snapshot(10))
        self.session.update(3, snapshot(11))
        self.session.update(4, snapshot(12))
        self.session.update(5, snapshot(11))
        self.session.update(6, {"state": "missing timestamp"})
        self.session.update(7, snapshot(float("nan")))
        self.session.update(23, snapshot(13))
        self.session.update(24, snapshot(14))
        self.assertEqual([s["updated_at_unix"] for s in self.session.samples], [11, 12])

    def test_finished_message_and_one_time_completion_event_persist(self):
        self.session.start(0)
        self.session.update(3, snapshot(1))
        self.assertTrue(self.session.update(23))
        saved = self.output.read_bytes()
        self.assertFalse(self.session.update(24, snapshot(2)))
        self.assertFalse(self.session.update(1000, snapshot(3)))
        self.assertEqual(self.session.state, "FINISHED")
        self.assertEqual(self.session.display(1000), "SAMPLING FINISHED - you may move")
        self.assertEqual(self.output.read_bytes(), saved)
        self.assertFalse(self.session.start(1000))

    def test_completed_report_contains_collected_joint_statistics_and_workflow(self):
        self.session.start(0)
        self.session.update(3, snapshot(1, 0))
        self.session.update(4, snapshot(2, 5))
        self.session.update(5, snapshot(3, 10))
        self.session.update(23)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(report["sampling_completed"])
        self.assertEqual(report["fresh_samples"], 3)
        self.assertEqual(report["workflow_note"],
                         "handsfree sampling after calibration; ground truth movement needs operator")
        self.assertEqual(report["joints"]["left_elbow_joint"]["actual_motion_deg"], 10)
        self.assertTrue(report["joints"]["left_elbow_joint"]["motion_observed"])

    def test_reset_preserves_saved_report_until_the_next_sampling_finishes(self):
        self.session.start(0)
        self.session.update(3, snapshot(1))
        self.session.update(23)
        completed = self.output.read_bytes()
        self.session.reset()
        self.assertEqual(self.session.state, "WAITING_CALIBRATION")
        self.assertEqual(self.session.samples, [])
        self.assertEqual(self.output.read_bytes(), completed)
        self.assertTrue(self.session.start(30))
        self.assertFalse(self.session.start(31))
        self.session.update(33, snapshot(2))
        self.assertEqual(self.output.read_bytes(), completed)
        self.assertTrue(self.session.update(53))
        self.assertEqual(self.session.report["fresh_samples"], 1)

    def test_snapshot_is_a_stable_copy_of_control_state(self):
        self.session.start(0)
        value = snapshot(1, 5)
        self.session.update(3, value)
        value["actual_deg"][0] = 100
        self.session.update(23)
        self.assertEqual(self.session.samples[0]["actual_deg"], [5])

    def test_empty_sample_window_still_finishes_without_claiming_observed_motion(self):
        self.session.start(0)
        self.assertTrue(self.session.update(23))
        self.assertEqual(self.session.report["fresh_samples"], 0)
        self.assertEqual(self.session.report["joints"], {})
        self.assertTrue(self.session.report["sampling_completed"])

    def test_report_builder_has_no_output_or_cli_side_effects(self):
        report = build_report([snapshot(1, 1), snapshot(2, 7)], 20)
        self.assertEqual(report["seconds"], 20)
        self.assertEqual(report["joints"]["left_elbow_joint"]["actual_motion_deg"], 6)
        self.assertFalse(self.output.exists())

    def test_brief_right_step_between_status_snapshots_is_reported(self):
        self.session.start(0)
        self.session.update(3, snapshot(1))
        stats = {"height_m": .78, "tilt_deg": 2., "foot_loads_n": [170., 170.],
                 "other_floor_contacts": 0, "external_force_max": 0.,
                 "body_force_max": 0., "max_torque_limit_ratio": .8,
                 "foot_clearance_m": [0., 0.], "fallen": False, "phase": "STEPPING"}
        self.session.observe_physics(3., 10., stats)
        stats["foot_clearance_m"] = [.001, .027]
        stats["foot_loads_n"] = [340., 0.]
        self.session.observe_physics(3.064, 10.064, stats)
        stats["foot_clearance_m"] = [0., 0.]
        stats["foot_loads_n"] = [170., 170.]
        self.session.observe_physics(3.066, 10.066, stats)
        self.session.update(3.5, snapshot(2))
        self.session.update(23)
        self.assertEqual(self.session.report['continuous_physics']['samples'], 3)
        self.assertEqual(self.session.report['physical_steps']['foot_clearance_max_m'], [.001, .027])
        self.assertEqual(self.session.report['physical_steps']['foot_unloaded_samples'], [0, 1])

    def test_physics_measurements_respect_window_reset_and_simulation_time(self):
        self.session.start(0)
        stats = {"height_m": .78, "tilt_deg": 2., "foot_loads_n": [170., 170.],
                 "other_floor_contacts": 0, "external_force_max": 0.,
                 "body_force_max": 0., "max_torque_limit_ratio": .8,
                 "foot_clearance_m": [0., 0.], "fallen": False, "phase": "STANDING"}
        self.session.observe_physics(2.99, 1., stats)
        self.session.observe_physics(3., 2., stats)
        self.session.observe_physics(3.1, 2., stats)
        self.session.observe_physics(23., 3., stats)
        self.session.update(23)
        self.assertEqual(self.session.report['continuous_physics']['samples'], 1)
        self.session.reset()
        self.session.start(30.)
        self.session.update(53.)
        self.assertNotIn('continuous_physics', self.session.report)

    def test_actual_walking_and_tracking_models_supply_finite_physics_report(self):
        from joint_mapping import JointRobot
        root=Path(__file__).resolve().parent
        for model,mapping in (('unitree_g1/scene_physics.xml','unitree_g1_walking_mapping.json'),
                              ('unitree_g1_twist2/scene.xml','unitree_g1_tracking_mapping.json')):
            with self.subTest(mapping=mapping):
                robot=JointRobot(root/'robots'/model,root/mapping)
                self.session.reset()
                self.session.start(0.)
                for _ in range(100):
                    robot.step()
                    self.session.observe_robot(3.+robot.data.time,robot)
                self.session.update(23.)
                physics=self.session.report['continuous_physics']
                self.assertEqual(physics['samples'],100)
                self.assertLessEqual(physics['max_sample_gap_s'],robot.model.opt.timestep+1e-10)
                self.assertEqual(physics['max_external_force'],0.)
                self.assertEqual(physics['max_applied_body_force'],0.)
                # Strict JSON rejects NaN/Inf; report extrema must be measured.
                json.dumps(self.session.report,allow_nan=False)


if __name__ == "__main__":
    unittest.main()
