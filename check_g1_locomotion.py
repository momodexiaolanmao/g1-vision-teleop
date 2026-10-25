"""Deterministic real-contact locomotion validation, not a camera accuracy test."""
import json
from pathlib import Path
import time

import numpy as np

from joint_mapping import JointRobot
from g1_locomotion import G1Locomotion
from validate_unitree_g1 import pose

ROOT=Path(__file__).absolute().parent


def run(upper=False, strength=.5):
    robot=JointRobot(ROOT/'robots/unitree_g1/scene_physics.xml', ROOT/'unitree_g1_physics_mapping.json')
    controller=G1Locomotion(robot.model,robot.data,{'imitation_strength':strength})
    robot.calibrate(pose())
    records=[]
    started=time.perf_counter()
    for step in range(15000):
        t=step*.002
        controller.set_velocity([0 if t<3 or t>25 else .3, 0,0])
        if upper and t>3 and step%15==0:
            targets=robot.targets_from_pose(pose(t-3,dancing=True))
            if targets is not None:
                robot.set_targets(targets)
                controller.set_reference(robot.goal,robot.entries)
        controller.step()
        if step%50==0:
            records.append({'time':t,**controller.status()})
        if controller.fallen:
            break
    report={'upper_body_follow':upper,'duration_s':robot.data.time,
            'wall_s':time.perf_counter()-started,'samples':records,
            'final':controller.status(),'distance_m':float(np.linalg.norm(robot.data.qpos[:2]))}
    (ROOT/f'artifacts/unitree_g1_locomotion_{"upper" if upper else "baseline"}.json').write_text(
        json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
    assert not controller.fallen,'Robot fell'
    assert controller.max_torque_ratio<=1.00001
    assert all(r['other_floor_contacts']==0 for r in records if r['time']>2)
    for side in (0,1):
        assert max(r['foot_clearance_m'][side] for r in records)>.01,'No actual foot lift'
        assert sum(r['foot_loads_n'][side]<5 for r in records)>5,'No single-support phase'
    assert np.max(abs(robot.data.xfrc_applied))==0
    assert np.max(abs(robot.data.qfrc_applied))==0


if __name__=='__main__':
    run(False)
    run(True)
