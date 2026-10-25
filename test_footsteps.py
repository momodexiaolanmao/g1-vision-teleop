"""Camera-to-physics step integration and stationary/no-input regressions."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import mujoco

from footstep_intent import FootstepIntent
from joint_mapping import JointRobot
from validate_unitree_g1 import pose

ROOT=Path(__file__).absolute().parent


class FootstepTests(unittest.TestCase):
    def robot(self):
        return JointRobot(ROOT/'robots/unitree_g1/scene_physics.xml',ROOT/'unitree_g1_walking_mapping.json')

    def test_stationary_recorded_camera_does_not_start_steps(self):
        path=ROOT/'fixtures/stationary_feet.json'
        if not path.exists():
            self.skipTest('Private camera landmark fixture is intentionally not distributed')
        traces=json.loads(path.read_text())
        detector=FootstepIntent()
        ready=False
        counts=0
        for trace in traces:
            points=[SimpleNamespace(x=0,y=0,z=0,visibility=0,presence=0) for _ in range(33)]
            for i,p in trace['feet'].items():
                points[int(i)]=SimpleNamespace(x=p[0],y=p[1],z=p[2],visibility=p[3],presence=p[4])
            world=[SimpleNamespace(x=0,y=0,z=0) for _ in range(33)]
            for i,p in trace['leg_image'].items():
                points[int(i)]=SimpleNamespace(x=p[0],y=p[1],z=p[2],visibility=p[3],presence=p[4])
            for i,p in trace['leg_world'].items():
                world[int(i)]=SimpleNamespace(x=p[0],y=p[1],z=p[2])
            observation=SimpleNamespace(arrived=trace['time'],landmarks=points,world_landmarks=world)
            if not ready:
                ready=detector.calibrate(observation,.22)
            else:
                self.assertFalse(detector.update(observation),detector.status())
                counts+=1
        self.assertGreater(counts,100)

    def test_common_camera_height_shift_does_not_start_steps(self):
        detector=FootstepIntent()
        initial=pose()
        self.assertTrue(detector.calibrate(initial,.22))
        for stamp in np.arange(0,1,.03):
            observation=pose()
            observation.arrived=float(stamp)
            for point in observation.landmarks[27:]:
                point.y-=.04
            self.assertFalse(detector.update(observation),detector.status())

    def test_one_foot_image_error_without_leg_bend_does_not_start_steps(self):
        detector=FootstepIntent()
        self.assertTrue(detector.calibrate(pose(),.22))
        for stamp in np.arange(0,1,.03):
            observation=pose()
            observation.arrived=float(stamp)
            for index in (28,30,32):
                observation.landmarks[index].y-=.04
            self.assertFalse(detector.update(observation),detector.status())

    def test_confirmed_short_step_survives_grounded_interval_then_expires(self):
        detector=FootstepIntent()
        self.assertTrue(detector.calibrate(pose(),.52/2.2))
        active=[]
        for stamp in np.arange(0,1.8,.03):
            values={'left_hip_pitch':-.65,'left_knee_flex':1.} if stamp<.2 else {}
            observation=pose(**values)
            observation.arrived=float(stamp)
            active.append((stamp,detector.update(observation)))
        self.assertTrue(all(value for stamp,value in active if .21<=stamp<=1.0))
        self.assertFalse(active[-1][1])

    def test_short_alternating_steps_enter_walk_and_settle_with_real_physics(self):
        for duration in (.2,.3):
            with self.subTest(lift_duration_s=duration):
                robot=self.robot()
                self.assertTrue(robot.calibrate(pose()))
                first_step=None
                records=[]
                for step in range(7000):
                    t=step*.002
                    if step%15==0:
                        values={}
                        if 2<=t<8 and (t-2)%.65<duration:
                            side='left' if int((t-2)/.65)%2==0 else 'right'
                            values={side+'_hip_pitch':-.65,side+'_knee_flex':1.}
                        observation=pose(**values)
                        observation.arrived=t
                        targets=robot.targets_from_pose(observation)
                        self.assertIsNotNone(targets)
                        robot.set_targets(targets)
                    robot.step()
                    if first_step is None and robot.balance.phase=='STEPPING':
                        first_step=t
                    if step%25==0:
                        records.append({'time':t,**robot.balance.status()})
                    self.assertFalse(robot.balance.fallen,robot.balance.status())
                self.assertIsNotNone(first_step,'Short visible steps never entered locomotion')
                self.assertLess(first_step,4.,'Starting took more than two seconds of short steps')
                walking=[s for s in records if s['phase']=='STEPPING']
                for side in (0,1):
                    self.assertGreater(max(s['foot_clearance_m'][side] for s in walking),.006)
                    self.assertGreater(sum(s['foot_loads_n'][side]<5 for s in walking),5)
                self.assertEqual(robot.balance.phase,'STANDING')
                self.assertFalse(robot.footstep.active)
                self.assertLess(np.linalg.norm(robot.balance.command),.025)
                self.assertEqual(max(s['other_floor_contacts'] for s in records),0)

    def test_hidden_foot_or_unclear_leg_cancels_confirmed_request_immediately(self):
        for hidden in ('right_heel','right_knee'):
            detector=FootstepIntent()
            self.assertTrue(detector.calibrate(pose(),.52/2.2))
            for stamp in np.arange(0,.3,.03):
                observation=pose(left_hip_pitch=-.65,left_knee_flex=1.)
                observation.arrived=float(stamp)
                detector.update(observation)
            self.assertTrue(detector.active)
            from pose_retarget import IDS
            observation.landmarks[IDS[hidden]].visibility=.1
            observation.arrived=.33
            self.assertFalse(detector.update(observation))
            observation=pose()
            observation.arrived=.36
            self.assertFalse(detector.update(observation),'Hidden input left a live step request')
        detector.hold()
        self.assertFalse(detector.active)

    def test_short_pause_preserves_return_direction_and_active_cycle(self):
        robot=self.robot()
        robot.calibrate(pose())
        targets=robot.targets_from_pose(pose())
        robot.set_targets(targets)
        for _ in range(1000):
            robot.step()
        robot.balance.update_intent(True,{'valid':True})
        returning=False
        for _ in range(4000):
            robot.step()
            loco=robot.balance.locomotion
            yaw=loco.origin_yaw
            local_x=np.dot(robot.data.qpos[:2]-loco.origin,[np.cos(yaw),np.sin(yaw)])
            if robot.balance.phase=='STEPPING' and loco.command[0]<-.25 and .05<local_x<.20:
                returning=True
                break
        self.assertTrue(returning,'Robot never began a bounded return step')
        before=loco.intent_phase
        robot.balance.update_intent(False,{'valid':True})
        for _ in range(120):
            robot.step()
        self.assertEqual(robot.balance.phase,'SETTLING')
        robot.balance.update_intent(True,{'valid':True})
        for _ in range(30):
            robot.step()
        self.assertEqual(robot.balance.phase,'STEPPING')
        self.assertGreater(loco.intent_phase,before,'A short pause restarted the walking cycle')
        self.assertLess(loco.command[0],-.25,'A short pause replaced the return step with a forward start')
        self.assertFalse(robot.balance.fallen)

    def test_recorded_right_steps_keep_nonzero_velocity_and_real_clearance(self):
        path=ROOT/'artifacts/step_entry_control_20261012.jsonl'
        if not path.exists():
            self.skipTest('Recorded field regression is unavailable in this package')
        with path.open(encoding='utf-8') as trace:
            metadata=json.loads(next(trace))
            rows=[]
            for line in trace:
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('type')!='input':
                    continue
                if row['time']>45.:
                    break
                rows.append(row)
        profile=ROOT/'artifacts/step_velocity_regression_profile.json'
        profile.write_text(json.dumps(metadata['mapping']),encoding='utf-8')
        robot=JointRobot(metadata['model'],profile)
        robot.data.qpos[:]=metadata['qpos']
        robot.data.qvel[:]=metadata['qvel']
        robot.data.time=metadata['time']
        mujoco.mj_forward(robot.model,robot.data)
        robot.balance.reset()
        dt=robot.model.opt.timestep
        minimum_active_speed=np.inf
        maximum_local_x=0.
        maximum_tilt=0.
        first_fall=None
        right_windows=[{'start_s':16.876,'end_s':19.050,'max_clearance_m':np.zeros(2),
                        'minimum_speed_m_s':np.inf},
                       {'start_s':31.216,'end_s':33.292,'max_clearance_m':np.zeros(2),
                        'minimum_speed_m_s':np.inf}]
        for row in rows:
            for _ in range(round((row['time']-robot.data.time)/dt)):
                robot.step()
                loco=robot.balance.locomotion
                yaw=robot.balance.origin_yaw
                local_x=np.dot(robot.data.qpos[:2]-robot.balance.origin,[np.cos(yaw),np.sin(yaw)])
                maximum_local_x=max(maximum_local_x,abs(local_x))
                tilt=np.degrees(np.arccos(np.clip(robot.data.xmat[robot.balance.pelvis].reshape(3,3)[2,2],-1,1)))
                maximum_tilt=max(maximum_tilt,tilt)
                if robot.balance.fallen and first_fall is None:
                    first_fall=robot.data.time
                if robot.balance.phase=='STEPPING' and robot.data.time-robot.balance.phase_since>.04:
                    speed=abs(loco.command[0])
                    minimum_active_speed=min(minimum_active_speed,speed)
                    clearances=np.array([min(robot.data.geom_xpos[g,2]-robot.model.geom_size[g,0]
                                            for g in loco.foot_geoms if robot.model.geom_bodyid[g]==foot)
                                         for foot in loco.feet])
                    for window in right_windows:
                        if window['start_s']<=robot.data.time<window['end_s']:
                            window['max_clearance_m']=np.maximum(window['max_clearance_m'],clearances)
                            window['minimum_speed_m_s']=min(window['minimum_speed_m_s'],speed)
            robot.goal=np.array(row['goal'])
            robot.balance.origin=np.array(row['origin'])
            robot.balance.origin_yaw=row['origin_yaw']
            robot.balance.update_intent(row['step_active'],row['root_intent'])
        windows=[{**w,'max_clearance_m':w['max_clearance_m'].tolist()} for w in right_windows]
        report={'input':'actual recorded field goals and intents; initial qpos/qvel applied once',
                'trace':path.name,'input_records':len(rows),'duration_s':robot.data.time,
                'minimum_active_forward_speed_m_s':float(minimum_active_speed),
                'maximum_local_x_from_origin_m':float(maximum_local_x),'maximum_tilt_deg':float(maximum_tilt),
                'first_fall_s':first_fall,'right_action_windows':windows,'final':robot.balance.status()}
        (ROOT/'artifacts/unitree_g1_step_velocity_real_replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        self.assertIsNone(first_fall,report)
        self.assertLess(maximum_tilt,20.,report)
        self.assertLess(maximum_local_x,.5,report)
        self.assertGreaterEqual(minimum_active_speed,.25,report)
        for window in windows:
            for clearance in window['max_clearance_m']:
                self.assertGreater(clearance,.006,report)

    def test_repeated_human_foot_lifts_produce_real_contact_steps(self):
        robot=self.robot()
        neutral=pose()
        neutral.arrived=0.
        self.assertTrue(robot.calibrate(neutral))
        records=[]
        for step in range(15000):
            t=step*.002
            if step%15==0:
                values={}
                if 3<t<24:
                    side='left' if int((t-3)/.6)%2==0 else 'right'
                    values={side+'_hip_pitch':-.65,side+'_knee_flex':1.0,
                            'left_shoulder_pitch':-.5-.3*np.sin(t),
                            'right_shoulder_roll':-.3-.2*np.sin(t)}
                observation=pose(**values)
                observation.arrived=t
                targets=robot.targets_from_pose(observation)
                if targets is not None:
                    robot.set_targets(targets)
            robot.step()
            if step%25==0:
                records.append({'time':t,'intent':robot.footstep.status(),**robot.balance.status()})
            self.assertFalse(robot.balance.fallen,robot.balance.status())
        report={'duration_s':robot.data.time,'samples':records,'final':robot.balance.status()}
        (ROOT/'artifacts/unitree_g1_footstep_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        for side in (0,1):
            self.assertGreater(max(s['foot_clearance_m'][side] for s in records),.006)
            self.assertGreater(sum(s['foot_loads_n'][side]<5 for s in records if 5<s['time']<24),5)
        self.assertFalse(robot.footstep.active)
        self.assertLess(np.linalg.norm(robot.data.qpos[:2]),.5)
        self.assertLess(np.linalg.norm(robot.balance.command),.025)
        self.assertEqual(max(s['other_floor_contacts'] for s in records),0)
        self.assertEqual(np.max(abs(robot.data.xfrc_applied)),0)
        self.assertEqual(np.max(abs(robot.data.qfrc_applied)),0)

    def test_hidden_feet_pause_stop_and_calibration_never_reset_base(self):
        robot=self.robot()
        observation=pose()
        self.assertTrue(robot.calibrate(observation))
        for _ in range(1500):
            robot.step()
        before=robot.data.qpos[:7].copy()
        self.assertTrue(robot.calibrate(pose()))
        np.testing.assert_array_equal(robot.data.qpos[:7],before)
        for point in observation.landmarks[27:]:
            point.visibility=.1
        self.assertFalse(robot.calibration_ready(observation))
        robot.balance.update_intent(True,{'valid':True})
        targets=robot.targets_from_pose(observation)
        self.assertIsNotNone(targets)
        robot.set_targets(targets)
        self.assertFalse(robot.balance.intent_active)
        robot.hold()
        self.assertFalse(robot.footstep.active)
        for _ in range(1500):
            robot.step()
        self.assertFalse(robot.balance.fallen)
        self.assertLess(np.linalg.norm(robot.balance.command),.01)

    def test_large_references_turn_and_side_motion_remain_bounded(self):
        robot=self.robot()
        robot.calibrate(pose())
        records=[]
        for step in range(15000):
            t=step*.002
            if step%15==0:
                values={}
                if 3<t<24:
                    side='left' if int((t-3)/.6)%2==0 else 'right'
                    values={side+'_hip_pitch':-.65,side+'_knee_flex':1.,
                            'left_shoulder_pitch':-2.5,'right_shoulder_pitch':-2.5,
                            'left_elbow_flex':.5,'right_elbow_flex':.5,
                            'waist_roll':.35,'waist_pitch':.4,'waist_yaw':.5}
                observation=pose(turn=.25 if 10<t<24 else 0,
                                 lateral=.10 if 14<t<24 else 0,**values)
                observation.arrived=t
                targets=robot.targets_from_pose(observation)
                if targets is not None:
                    robot.set_targets(targets)
            robot.step()
            if step%50==0:
                records.append({'time':t,**robot.balance.status()})
            self.assertFalse(robot.balance.fallen,robot.balance.status())
        (ROOT/'artifacts/unitree_g1_walking_stress.json').write_text(json.dumps({
            'duration_s':robot.data.time,'samples':records,'final':robot.balance.status()},indent=2),encoding='utf-8')
        self.assertEqual(max(s['other_floor_contacts'] for s in records),0)
        self.assertLess(max(s['tilt_deg'] for s in records),25)

    def test_replay_of_real_failed_upper_reference_keeps_standing(self):
        robot=self.robot()
        target=np.radians(json.loads((ROOT/'fixtures/upper_reference_from_failed_run.json').read_text()))
        records=[]
        for step in range(15000):
            if step>1500:
                robot.set_targets(target)
            robot.step()
            if step%100==0:
                records.append(robot.balance.status())
            self.assertFalse(robot.balance.fallen,robot.balance.status())
        state=robot.balance.status()
        self.assertEqual(state['phase'],'STANDING')
        self.assertGreater(state['height_m'],.6)
        self.assertEqual(max(s['other_floor_contacts'] for s in records),0)
        (ROOT/'artifacts/unitree_g1_failed_reference_replay.json').write_text(json.dumps({
            'input':'motor reference captured at the real failing run',
            'duration_s':robot.data.time,'final':state,
            'max_tilt_deg':max(s['tilt_deg'] for s in records),
            'any_fall':any(s['fallen'] for s in records)},indent=2),encoding='utf-8')

    def test_zero_imitation_leaves_trained_policy_balance_actions_unchanged(self):
        baseline,following=self.robot(),self.robot()
        a,b=baseline.balance.locomotion,following.balance.locomotion
        b.config['imitation_strength']=0.
        target=np.radians(json.loads((ROOT/'fixtures/upper_reference_from_failed_run.json').read_text()))
        b.set_reference(target,following.entries)
        a.set_velocity([.3,.05,.05])
        b.set_velocity([.3,.05,.05])
        for _ in range(1500):
            a.step()
            b.step()
            np.testing.assert_allclose(b.target,a.target,atol=1e-10,rtol=0)
            np.testing.assert_allclose(following.data.qpos,baseline.data.qpos,atol=1e-10,rtol=0)

    def test_long_steps_and_repeated_stops_with_real_problematic_reference(self):
        robot=self.robot()
        robot.calibrate(pose())
        target=np.radians(json.loads((ROOT/'fixtures/upper_reference_from_hybrid_failed_run.json').read_text(encoding='utf-8-sig')))
        records=[]
        for step in range(90000):
            t=step*.002
            if step%15==0:
                if t<=12:
                    side='left' if int(max(0,t-3)/.6)%2==0 else 'right'
                    values={side+'_hip_pitch':-.65,side+'_knee_flex':1.,
                            'left_shoulder_pitch':-.5-.3*np.sin(t),
                            'right_shoulder_roll':-.3-.2*np.sin(t)} if t>3 else {}
                    observation=pose(**values)
                    observation.arrived=t
                    robot.set_targets(robot.targets_from_pose(observation))
                else:
                    robot.set_targets(target)
                active=t>3 and (t<24 or (t-24)%2.7<2.35)
                robot.balance.update_intent(active,{'valid':True,
                    'lateral_m':.15*np.sin(t/4),'yaw_rad':.35*np.sin(t/7)})
            robot.step()
            self.assertFalse(robot.balance.fallen,robot.balance.status())
            if step%100==0:
                records.append({'time':t,**robot.balance.status()})
        self.assertEqual(max(s['other_floor_contacts'] for s in records),0)
        self.assertLess(max(s['tilt_deg'] for s in records),20)
        self.assertEqual(max(s['external_force_max'] for s in records),0)
        self.assertIn('SETTLING',{s['phase'] for s in records})
        (ROOT/'artifacts/unitree_g1_long_switch_validation.json').write_text(json.dumps({
            'input':'captured problematic reference with synthetic turn/side/stop commands',
            'duration_s':robot.data.time,'any_fall':False,'final':robot.balance.status(),
            'max_tilt_deg':max(s['tilt_deg'] for s in records),'samples':records},indent=2),encoding='utf-8')


if __name__=='__main__':
    unittest.main()
