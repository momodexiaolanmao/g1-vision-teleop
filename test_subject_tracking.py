"""Foreground selection stays with the same torso when detector order changes."""
import copy
import unittest

from subject_tracker import SubjectTracker
from validate_unitree_g1 import pose


class SubjectTests(unittest.TestCase):
    def test_before_calibration_a_new_foreground_person_can_be_selected(self):
        front,background=self.people()
        tracker=SubjectTracker()
        self.assertEqual(tracker.select([background],16/9),0)
        self.assertEqual(tracker.select([background,front],16/9),1)
    def people(self):
        front=pose().landmarks
        background=copy.deepcopy(front)
        for point in background:
            point.x=.88+.55*(point.x-.5)
            point.y=.48+.55*(point.y-.48)
        return front,background

    def test_largest_central_subject_wins_independent_of_result_order(self):
        front,background=self.people()
        self.assertEqual(SubjectTracker().select([background,front],16/9),1)
        self.assertEqual(SubjectTracker().select([front,background],16/9),0)

    def test_reordering_and_background_only_frames_never_switch_subject(self):
        front,background=self.people()
        tracker=SubjectTracker()
        self.assertEqual(tracker.select([front,background],16/9),0)
        tracker.lock()
        self.assertEqual(tracker.select([background,front],16/9),1)
        self.assertIsNone(tracker.select([background],16/9))
        self.assertEqual(tracker.select([background,front],16/9),1)

    def test_small_translation_and_hidden_hips_keep_visible_arm_input(self):
        front,background=self.people()
        tracker=SubjectTracker()
        tracker.select([front,background],16/9)
        tracker.lock()
        shifted=copy.deepcopy(front)
        for point in shifted:
            point.x+=.015
        for index in (23,24):
            shifted[index].visibility=.1
        self.assertEqual(tracker.select([background,shifted],16/9),1)

    def test_reacquisition_requires_explicit_reset(self):
        front,background=self.people()
        tracker=SubjectTracker()
        tracker.select([front],16/9)
        tracker.lock()
        for _ in range(60):
            self.assertIsNone(tracker.select([background],16/9))
        tracker.reset()
        self.assertEqual(tracker.select([background],16/9),0)

    def test_hidden_hips_cannot_disguise_a_small_background_person(self):
        front,_=self.people()
        tracker=SubjectTracker()
        tracker.select([front],16/9)
        tracker.lock()
        background=copy.deepcopy(front)
        center_y=tracker.center[1]
        for point in background:
            point.x=.515+.25*(point.x-.5)
            point.y=center_y+.25*(point.y-center_y)
        for index in (23,24):
            background[index].visibility=.1
        self.assertIsNone(tracker.select([background],16/9))
        self.assertEqual(tracker.state,'SELECTED PERSON LOST - HOLD')
        self.assertEqual(tracker.select([background,front],16/9),1)

    def test_lock_to_uses_the_calibrated_observations_shoulder_width(self):
        front,background=self.people()
        tracker=SubjectTracker()
        target=tracker._candidate(front,16/9)
        tracker.select([background],16/9)
        tracker.lock_to(*target)
        self.assertEqual(tracker.select([front],16/9),0)

    def test_low_confidence_shoulders_do_not_establish_a_track(self):
        front,_=self.people()
        for index in (11,12):
            front[index].visibility=.1
        self.assertIsNone(SubjectTracker().select([front],16/9))
