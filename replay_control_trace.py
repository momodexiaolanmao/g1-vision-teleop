"""Replay recorded mapped goals and intents through unmodified physics steps."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from joint_mapping import JointRobot

ROOT=Path(__file__).absolute().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/unitree_g1_control_replay_check.json')
    args=parser.parse_args()
    metadata=None
    rows=[]
    for line in args.trace.read_text(encoding='utf-8').splitlines():
        try:
            row=json.loads(line)
        except json.JSONDecodeError:
            continue  # The active recorder may not have completed its last line.
        if row.get('type')=='metadata':
            metadata=row
        elif row.get('type')=='input':
            rows.append(row)
    if metadata is None or not rows:
        raise ValueError('A metadata record and at least one input are required')
    profile=ROOT/'artifacts/control_replay_profile.json'
    profile.write_text(json.dumps(metadata['mapping'],ensure_ascii=False),encoding='utf-8')
    robot=JointRobot(metadata['model'],profile)
    # Initialize once at the recorded starting state. During playback, only
    # motor torques and normal forward dynamics move the free base.
    robot.data.qpos[:]=metadata['qpos']
    robot.data.qvel[:]=metadata['qvel']
    robot.data.time=metadata['time']
    mujoco.mj_forward(robot.model,robot.data)
    robot.balance.reset()
    dt=robot.model.opt.timestep
    largest_deviation=0.
    states=[]
    previous_phase=None
    for index,row in enumerate(rows):
        if row['time']<robot.data.time-dt/2:
            raise ValueError('Trace contains an explicit reset; replay each run separately')
        steps=round((row['time']-robot.data.time)/dt)
        for _ in range(steps):
            robot.step()
        largest_deviation=max(largest_deviation,float(np.max(abs(robot.data.qpos-np.array(row['qpos'])))))
        robot.goal=np.array(row['goal'])
        robot.balance.origin=np.array(row['origin'])
        robot.balance.origin_yaw=row['origin_yaw']
        robot.balance.update_intent(row['step_active'],row['root_intent'])
        if robot.balance.phase!=previous_phase or index%30==0:
            previous_phase=robot.balance.phase
            states.append(robot.balance.status())
    states.append(robot.balance.status())
    report={'trace':args.trace.name,'input_records':len(rows),
            'duration_s':robot.data.time-metadata['time'],
            'any_recorded_fall':any(r['fallen'] for r in rows),
            'any_replayed_fall':any(s['fallen'] for s in states),
            'max_qpos_deviation_from_recording':largest_deviation,
            'phase_changes':[s['phase'] for s in states],
            'max_tilt_deg':max(s['tilt_deg'] for s in states),
            'external_force_max':max(s['external_force_max'] for s in states),
            'final':robot.balance.status()}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
