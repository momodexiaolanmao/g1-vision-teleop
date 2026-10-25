"""A Windows diagnostic-file lock must not interrupt robot control."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from joint_mapping import JointRobot
from teleop import write_live_status

ROOT=Path(__file__).resolve().parent


class LiveStatusTests(unittest.TestCase):
    def test_locked_snapshot_preserves_old_report_and_control_then_retries(self):
        robot=JointRobot(ROOT/'arm.xml',ROOT/'joint_mapping.json')
        session=SimpleNamespace(enabled=True,start_pending=False,
                                status=lambda *args:'TRACKING')
        qpos,ctrl,command=robot.data.qpos.copy(),robot.data.ctrl.copy(),robot.command.copy()
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'live_status.json'
            previous={'state':'previous complete snapshot'}
            destination.write_text(json.dumps(previous),encoding='utf-8')
            with patch.object(Path,'replace',side_effect=PermissionError('Windows reader lock')):
                report=write_live_status(destination,robot,session,None,True)
            self.assertIn('diagnostic_write_error',report)
            self.assertEqual(json.loads(destination.read_text(encoding='utf-8')),previous)
            np.testing.assert_array_equal(robot.data.qpos,qpos)
            np.testing.assert_array_equal(robot.data.ctrl,ctrl)
            np.testing.assert_array_equal(robot.command,command)
            replacement=write_live_status(destination,robot,session,None,True)
            saved=json.loads(destination.read_text(encoding='utf-8'))
            self.assertEqual(saved,replacement)
            self.assertEqual(saved['state'],'TRACKING')
            self.assertNotIn('diagnostic_write_error',saved)


if __name__=='__main__':
    unittest.main()
