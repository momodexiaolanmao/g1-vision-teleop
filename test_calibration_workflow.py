import unittest
from tracking_session import CalibrationGate
from validate_unitree_g1 import pose

class CalibrationWorkflowTests(unittest.TestCase):
    def test_countdown_does_not_calibrate_at_computer(self):
        gate=CalibrationGate(delay=8,stable_seconds=1.5)
        gate.begin(10)
        observation=pose()
        for now in (10,12,17.9):
            self.assertFalse(gate.ready(observation,True,now))
        self.assertFalse(gate.ready(observation,True,18))
        self.assertFalse(gate.ready(observation,True,19))
        self.assertTrue(gate.ready(observation,True,19.6))

    def test_walking_into_frame_restarts_stillness_wait(self):
        gate=CalibrationGate(delay=0,stable_seconds=1)
        still=pose()
        self.assertFalse(gate.ready(still,True,0))
        moving=pose()
        for p in moving.landmarks:
            p.x+=.04
        self.assertFalse(gate.ready(moving,True,.8))
        self.assertFalse(gate.ready(moving,True,1.1))
        self.assertTrue(gate.ready(moving,True,1.9))

    def test_hidden_feet_and_raised_arms_prevent_calibration(self):
        gate=CalibrationGate(delay=0,stable_seconds=.5)
        hidden=pose()
        hidden.landmarks[27].visibility=.1
        self.assertFalse(gate.ready(hidden,True,0))
        self.assertFalse(gate.ready(hidden,True,2))
        raised=pose(left_shoulder_pitch=-1.5)
        raised.landmarks[15].y=raised.landmarks[11].y-.1
        self.assertFalse(gate.ready(raised,True,3))
        self.assertFalse(gate.ready(raised,True,4))

if __name__=='__main__':
    unittest.main()
