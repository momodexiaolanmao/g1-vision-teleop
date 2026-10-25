"""Run explicit public regression groups and write a machine-readable result."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
QUICK=['test_tracking_input','test_pose_quality','test_subject_tracking',
       'test_calibration_workflow','test_sampling_session','test_live_status',
       'test.test_twist2_camera']
FULL=['test_g1_physics','test_footsteps']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='Add longer physical regressions')
    args=parser.parse_args()
    (ROOT/'artifacts').mkdir(exist_ok=True)
    # Keep test scratch files in the writable project, including restricted
    # desktop sessions where the Windows system temp directory is unavailable.
    scratch=ROOT/'artifacts/test-temp'
    scratch.mkdir(exist_ok=True)
    tempfile.tempdir=str(scratch)
    names=QUICK+(FULL if args.full else [])
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'group':'full' if args.full else 'quick','modules':names,
            'tests_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'skipped':[{'test':str(t),'reason':why} for t,why in result.skipped],
            'successful':result.wasSuccessful(),'python':sys.version.split()[0]}
    (ROOT/'artifacts/public_regression_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
