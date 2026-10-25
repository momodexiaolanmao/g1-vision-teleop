"""Render real forward dynamics driven by synthetic human foot lifts."""
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from joint_mapping import JointRobot
from validate_unitree_g1 import pose

ROOT=Path(__file__).absolute().parent


def main():
    robot=JointRobot(ROOT/'robots/unitree_g1/scene_physics.xml',ROOT/'unitree_g1_walking_mapping.json')
    observation=pose()
    observation.arrived=0.
    robot.calibrate(observation)
    output=ROOT/'artifacts'
    writer=cv2.VideoWriter(str(output/'unitree_g1_stepping_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(960,720))
    if not writer.isOpened():
        raise RuntimeError('Cannot save stepping preview')
    camera=mujoco.MjvCamera()
    camera.distance,camera.azimuth,camera.elevation=2.65,145,-13
    options=mujoco.MjvOption()
    options.geomgroup[0]=0
    samples=[]
    saved_pose=False
    try:
        with mujoco.Renderer(robot.model,height=720,width=960) as renderer:
            for frame in range(320):
                t=frame/20
                values={}
                if 3<t<13:
                    side='left' if int((t-3)/.6)%2==0 else 'right'
                    values={side+'_hip_pitch':-.65,side+'_knee_flex':1.,
                            'left_shoulder_pitch':-.5-.3*np.sin(t),
                            'right_shoulder_roll':-.3-.2*np.sin(t)}
                observation=pose(**values)
                observation.arrived=t
                targets=robot.targets_from_pose(observation)
                if targets is not None:
                    robot.set_targets(targets)
                for _ in range(25):
                    robot.step()
                state=robot.balance.status()
                samples.append(state)
                camera.lookat[:]=np.r_[robot.data.qpos[:2],.68]
                renderer.update_scene(robot.data,camera=camera,scene_option=options)
                frame_image=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                text=['G1 | physical steps + upper-body follow | synthetic input',
                      'Gravity, foot contacts and joint torque limits',
                      f"Foot height: {100*state['foot_clearance_m'][0]:.1f} / {100*state['foot_clearance_m'][1]:.1f} cm",
                      f"Foot load: {state['foot_loads_n'][0]:.0f} / {state['foot_loads_n'][1]:.0f} N | tilt {state['tilt_deg']:.1f} deg"]
                for row,message in enumerate(text):
                    cv2.putText(frame_image,message,(18,28+26*row),cv2.FONT_HERSHEY_SIMPLEX,.56,(235,245,248),1,cv2.LINE_AA)
                writer.write(frame_image)
                if 4<t<13 and max(state['foot_clearance_m'])>.013 and not saved_pose:
                    cv2.imwrite(str(output/'unitree_g1_stepping_pose.png'),frame_image)
                    saved_pose=True
    finally:
        writer.release()
    report={'input':'synthetic human landmarks, not recorded user motion','duration_s':robot.data.time,
            'any_fall':any(s['fallen'] for s in samples),
            'max_foot_clearance_m':np.max([s['foot_clearance_m'] for s in samples],axis=0).tolist(),
            'max_tilt_deg':max(s['tilt_deg'] for s in samples),
            'root_external_force_max':max(s['external_force_max'] for s in samples),
            'video':'unitree_g1_stepping_demo.mp4'}
    (output/'unitree_g1_stepping_demo_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
