"""The real retargeting seam: bad reconstruction must not become a command."""
import copy
import math
import unittest

import numpy as np

from pose_retarget import BodyRetarget, IDS
from validate_unitree_g1 import pose


class PoseQualityTests(unittest.TestCase):
    def test_world_depth_drift_without_image_motion_holds_joint_angles(self):
        from validate_unitree_g1 import pose
        from pose_retarget import BodyRetarget,IDS
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_elbow_flex=.8,left_shoulder_pitch=-.4)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        for frame in range(1,41):
            observation=pose(left_elbow_flex=.8,left_shoulder_pitch=-.4)
            observation.arrived=frame/30.
            observation.world_landmarks[IDS['left_elbow']].z+=.06*frame/40
            observation.world_landmarks[IDS['left_wrist']].z-=.04*frame/40
            channels,_=mapper.evaluate(observation)
            self.assertTrue(channels['left_elbow_flex']['valid'])
            self.assertAlmostEqual(channels['left_elbow_flex']['angle'],
                                   baseline['left_elbow_flex']['angle'],places=6)

    def mapper(self):
        mapper = BodyRetarget()
        mapper.quality_config = {"enabled": True, "max_angular_speed_deg_s": 650,
                                 "neutral_correction": True}
        observation = pose()
        observation.arrived = 10.
        self.assertTrue(mapper.calibrate(observation))
        return mapper

    def test_missing_observation_gap_cannot_release_same_image_depth_drift(self):
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_elbow_flex=.8,left_shoulder_pitch=-.4)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        for frame in range(1,11):
            observation=copy.deepcopy(initial)
            observation.arrived=frame/30.
            observation.world_landmarks[IDS['left_elbow']].z+=.006*frame
            observation.world_landmarks[IDS['left_wrist']].z-=.004*frame
            mapper.evaluate(observation)
        missing,_=mapper.evaluate(None)
        cached,_=mapper.evaluate(None)
        self.assertIs(missing,cached)
        self.assertFalse(missing['left_elbow_flex']['valid'])
        for frame in range(1,6):
            resumed=copy.deepcopy(observation)
            resumed.arrived=1.5+frame/30.
            channels,_=mapper.evaluate(resumed)
            self.assertTrue(channels['left_elbow_flex']['valid'])
            self.assertAlmostEqual(channels['left_elbow_flex']['angle'],
                                   baseline['left_elbow_flex']['angle'],places=6)

    def test_partial_bone_rejection_does_not_rebase_accepted_image_or_angles(self):
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_elbow_flex=.8,left_shoulder_pitch=-.4)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        for stamp in (.2,.4,.6,.8):
            bad=copy.deepcopy(initial)
            bad.arrived=stamp
            elbow,wrist=(bad.world_landmarks[IDS['left_'+n]] for n in ('elbow','wrist'))
            elbow.z+=.06
            for axis in ('x','y','z'):
                setattr(wrist,axis,getattr(elbow,axis)+.08*(getattr(wrist,axis)-getattr(elbow,axis)))
            channels,_=mapper.evaluate(bad)
            self.assertFalse(channels['left_elbow_flex']['valid'])
            if channels['left_shoulder_pitch']['valid']:
                self.assertAlmostEqual(channels['left_shoulder_pitch']['angle'],
                                       baseline['left_shoulder_pitch']['angle'],places=6)
        resumed=copy.deepcopy(initial)
        resumed.arrived=1.8
        resumed.world_landmarks[IDS['left_elbow']].z+=.06
        resumed.world_landmarks[IDS['left_wrist']].z-=.04
        channels,_=mapper.evaluate(resumed)
        for name in ('left_shoulder_pitch','left_elbow_flex'):
            self.assertTrue(channels[name]['valid'])
            self.assertAlmostEqual(channels[name]['angle'],baseline[name]['angle'],places=6)

    def test_visible_motion_after_gap_still_requires_three_fresh_samples(self):
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_shoulder_pitch=-.3,left_elbow_flex=.7)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        mapper.evaluate(None)
        for frame in range(1,4):
            observation=pose(left_shoulder_pitch=-1.3,left_elbow_flex=.7)
            observation.arrived=1.+frame/30.
            channels,_=mapper.evaluate(observation)
            self.assertTrue(channels['left_shoulder_pitch']['valid'])
            if frame<3:
                self.assertAlmostEqual(channels['left_shoulder_pitch']['angle'],
                                       baseline['left_shoulder_pitch']['angle'],places=6)
            else:
                self.assertLess(abs(channels['left_shoulder_pitch']['angle']+math.degrees(1.3)),2.)

    def test_hidden_hips_do_not_allow_depth_drift_to_move_visible_arms(self):
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_elbow_flex=.8,left_shoulder_pitch=-.4)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        for frame in range(1,41):
            observation=pose(left_elbow_flex=.8,left_shoulder_pitch=-.4)
            observation.arrived=frame/30.
            for name in ('left_hip','right_hip'):
                observation.landmarks[IDS[name]].visibility=.1
            observation.world_landmarks[IDS['left_elbow']].z+=.06*frame/40
            observation.world_landmarks[IDS['left_wrist']].z-=.04*frame/40
            channels,_=mapper.evaluate(observation)
            for name in ('left_shoulder_pitch','left_elbow_flex'):
                self.assertTrue(channels[name]['valid'])
                self.assertAlmostEqual(channels[name]['angle'],baseline[name]['angle'],places=6)

    def test_real_arm_motion_remains_responsive_when_hips_are_hidden(self):
        mapper=self.mapper()
        for frame in range(1,25):
            observation=pose(left_shoulder_pitch=-frame*.05,left_elbow_flex=.6)
            observation.arrived=10.+frame/30.
            for name in ('left_hip','right_hip'):
                observation.landmarks[IDS[name]].visibility=.1
            channels,_=mapper.evaluate(observation)
            self.assertTrue(channels['left_shoulder_pitch']['valid'])
        # Three confirmations can add up to roughly 0.1 s of motion latency.
        self.assertLess(abs(channels['left_shoulder_pitch']['angle']+math.degrees(1.2)),math.degrees(.15)+.5)
        for frame in range(25,29):
            observation=copy.deepcopy(observation)
            observation.arrived=10.+frame/30.
            channels,_=mapper.evaluate(observation)
        # The final subthreshold image movement may retain one trajectory step.
        self.assertLess(abs(channels['left_shoulder_pitch']['angle']+math.degrees(1.2)),math.degrees(.05)+.1)

    def test_sub_ten_pixel_image_noise_cannot_release_world_depth_drift(self):
        for width,height in ((1280,720),(640,960)):
            with self.subTest(size=(width,height)):
                mapper=BodyRetarget(quality_config={'enabled':True})
                initial=pose(left_elbow_flex=.8,left_shoulder_pitch=-.4)
                initial.frame=np.zeros((height,width,3),dtype=np.uint8)
                initial.arrived=0.
                self.assertTrue(mapper.calibrate(initial))
                baseline,_=mapper.evaluate(initial)
                for frame in range(1,16):
                    observation=pose(left_elbow_flex=1.2,left_shoulder_pitch=-.4)
                    observation.frame=initial.frame
                    observation.landmarks=copy.deepcopy(initial.landmarks)
                    observation.landmarks[IDS['left_wrist']].x+=8/width
                    observation.arrived=frame/30.
                    channels,_=mapper.evaluate(observation)
                    self.assertTrue(channels['left_elbow_flex']['valid'])
                    self.assertAlmostEqual(channels['left_elbow_flex']['angle'],
                                           baseline['left_elbow_flex']['angle'],places=6)

    def test_real_image_displacement_is_accepted_on_third_fresh_sample(self):
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_shoulder_pitch=-.3,left_elbow_flex=.7)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        for frame in range(1,4):
            observation=pose(left_shoulder_pitch=-.8,left_elbow_flex=.7)
            observation.arrived=frame/30.
            channels,_=mapper.evaluate(observation)
            self.assertTrue(channels['left_shoulder_pitch']['valid'])
            if frame<3:
                self.assertAlmostEqual(channels['left_shoulder_pitch']['angle'],
                                       baseline['left_shoulder_pitch']['angle'],places=6)
                # Re-reading a captured observation cannot count as confirmation.
                mapper.evaluate(observation)
                duplicate=copy.deepcopy(observation)
                repeated,_=mapper.evaluate(duplicate)
                self.assertAlmostEqual(repeated['left_shoulder_pitch']['angle'],
                                       baseline['left_shoulder_pitch']['angle'],places=6)
            else:
                self.assertLess(abs(channels['left_shoulder_pitch']['angle']+math.degrees(.8)),2.)

    def test_isolated_image_displacement_cannot_release_a_depth_jump(self):
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_shoulder_pitch=-.3,left_elbow_flex=.7)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        for frame in range(1,8):
            observation=pose(left_shoulder_pitch=-1.3,left_elbow_flex=.7)
            if frame%2==0:
                observation.landmarks=copy.deepcopy(initial.landmarks)
            observation.arrived=frame/30.
            channels,_=mapper.evaluate(observation)
            self.assertTrue(channels['left_shoulder_pitch']['valid'])
            self.assertAlmostEqual(channels['left_shoulder_pitch']['angle'],
                                   baseline['left_shoulder_pitch']['angle'],places=6)

    def test_slow_image_motion_accumulates_from_last_accepted_reference(self):
        mapper=BodyRetarget(quality_config={'enabled':True})
        initial=pose(left_shoulder_pitch=-.4,left_elbow_flex=.7)
        initial.arrived=0.
        self.assertTrue(mapper.calibrate(initial))
        baseline,_=mapper.evaluate(initial)
        prior=initial
        for frame in range(1,41):
            observation=pose(left_shoulder_pitch=-.4-frame*.01,left_elbow_flex=.7)
            observation.arrived=frame/30.
            # Each frame is below the pixel floor; their accumulated movement
            # must still become a new supported pose.
            a,b=(o.landmarks[IDS['left_wrist']] for o in (observation,prior))
            self.assertLess(math.hypot((a.x-b.x)*1280,(a.y-b.y)*720),10)
            channels,_=mapper.evaluate(observation)
            self.assertTrue(channels['left_shoulder_pitch']['valid'])
            prior=observation
        self.assertGreater(baseline['left_shoulder_pitch']['angle']-channels['left_shoulder_pitch']['angle'],15.)

    def test_single_frame_depth_flip_holds_only_affected_limb(self):
        mapper = self.mapper()
        normal = pose(left_shoulder_pitch=-.3, left_elbow_flex=.7)
        normal.arrived = 10.033
        mapper.evaluate(normal)
        bad = copy.deepcopy(normal)
        bad.arrived = 10.066
        bad.world_landmarks[IDS['left_elbow']].z += .65
        bad.world_landmarks[IDS['left_wrist']].z += .65
        channels, _ = mapper.evaluate(bad)
        self.assertFalse(channels['left_shoulder_pitch']['valid'])
        self.assertTrue(channels['right_elbow_flex']['valid'])
        normal.arrived = 10.099
        channels, _ = mapper.evaluate(normal)
        self.assertTrue(channels['left_shoulder_pitch']['valid'])
        self.assertLess(abs(channels['left_shoulder_pitch']['angle'] + math.degrees(.3)), 2)

    def test_neutral_body_bias_is_removed_by_calibration(self):
        mapper = self.mapper()
        neutral = pose(waist_pitch=.12, waist_roll=.09,
                       left_shoulder_roll=.1, right_shoulder_roll=-.1)
        neutral.arrived = 11.
        self.assertTrue(mapper.calibrate(neutral))
        neutral = copy.deepcopy(neutral)
        neutral.arrived = 11.033
        channels, _ = mapper.evaluate(neutral)
        self.assertLess(abs(channels['waist_pitch']['angle']), 1.)
        self.assertLess(abs(channels['left_shoulder_roll']['angle']), 1.)

    def test_real_fast_motion_is_not_frozen(self):
        mapper = self.mapper()
        for i in range(1, 25):
            observation = pose(left_shoulder_pitch=-i * .05, left_elbow_flex=.6)
            observation.arrived = 10. + i / 30
            channels, _ = mapper.evaluate(observation)
            self.assertTrue(channels['left_shoulder_pitch']['valid'])
        self.assertLess(abs(channels['left_shoulder_pitch']['angle'] + math.degrees(1.2)), math.degrees(.15)+.5)
        for i in range(25,29):
            observation=copy.deepcopy(observation)
            observation.arrived=10.+i/30.
            channels,_=mapper.evaluate(observation)
        self.assertLess(abs(channels['left_shoulder_pitch']['angle']+math.degrees(1.2)),math.degrees(.05)+.1)

    def test_collapsed_forearm_reconstruction_cannot_drive_elbow_or_wrist(self):
        mapper = self.mapper()
        bad = pose(left_elbow_flex=.5)
        bad.arrived = 10.033
        elbow, wrist = bad.world_landmarks[13], bad.world_landmarks[15]
        for axis in ('x','y','z'):
            setattr(wrist,axis,getattr(elbow,axis)+.08*(getattr(wrist,axis)-getattr(elbow,axis)))
        channels,_=mapper.evaluate(bad)
        self.assertFalse(channels['left_elbow_flex']['valid'])
        self.assertFalse(channels['left_wrist_pitch']['valid'])
        self.assertTrue(channels['right_elbow_flex']['valid'])


if __name__ == '__main__':
    unittest.main()
