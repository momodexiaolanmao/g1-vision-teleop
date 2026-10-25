"""Stand with inverse dynamics, prepare support, walk, then settle to stand.

Switching changes only motor controllers. It never resets or edits base state.
"""
import math

import numpy as np

from standing_balance import StandingBalance
from g1_locomotion import G1Locomotion


class HybridBalance:
    def __init__(self, model, data, config=None, standing_config=None):
        self.model,self.data,self.config=model,data,config or {}
        policy_config={**self.config,'imitation_strength':min(.2,float(self.config.get('imitation_strength',.2)))}
        self.standing=StandingBalance(model,data,standing_config)
        self.locomotion=G1Locomotion(model,data,policy_config)
        self.pelvis=self.standing.pelvis
        self.qadr,self.vadr=self.standing.qadr,self.standing.vadr
        self.reset()

    def reset(self):
        self.standing.reset()
        self.locomotion.reset()
        self.phase='STANDING'
        self.intent_active=False
        self.root_intent={}
        self.reference=None
        self.entries=None
        self.phase_since=self.data.time
        self.stable_since=None
        self.preparation_block_reason=''
        self.preparation_remaining_s=0.
        self.fallen=False
        self.max_torque_ratio=0.
        self.calibrate_origin()

    def calibrate_origin(self):
        self.locomotion.calibrate_origin()
        self.origin=self.locomotion.origin.copy()
        self.origin_yaw=self.locomotion.origin_yaw

    def set_reference(self, values, entries):
        self.reference=np.asarray(values).copy()
        self.entries=entries

    def update_intent(self, active, root):
        self.intent_active=bool(active)
        self.root_intent=root.copy()

    def hold(self):
        self.intent_active=False
        self.locomotion.hold()

    def _neutral_reference(self):
        neutral=np.empty(29)
        neutral[self.locomotion.order]=self.locomotion.default
        return np.array([neutral[self.locomotion.index[e['rule']['robot_joint']]] for e in self.entries])

    def _change(self, phase):
        self.phase=phase
        self.phase_since=self.data.time
        self.stable_since=None
        self.preparation_block_reason=''
        self.preparation_remaining_s=0.

    def step(self):
        d=self.data
        rotation=d.xmat[self.pelvis].reshape(3,3)
        tilt=math.acos(float(np.clip(rotation[2,2],-1,1)))
        if self.reference is None:
            self.standing.step()
            self.fallen=self.standing.fallen
            return
        if self.phase=='STANDING' and self.intent_active and tilt<math.radians(8):
            self._change('PREPARING STEP')
        elif self.phase=='STANDING' and self.intent_active:
            self.preparation_block_reason='stance not stable'
        if self.phase=='PREPARING STEP':
            if not self.intent_active:
                self._change('STANDING')
            else:
                neutral=self._neutral_reference()
                # Core and shoulder posture determine a safe handoff. Elbows,
                # wrists and shoulder yaw can settle slowly under the standing
                # tasks and do not need to match every policy angle first.
                error=max((abs(neutral[i]-d.qpos[e['qpos']]) for i,e in enumerate(self.entries)
                           if e['rule']['robot_joint'].startswith('waist_')
                           or e['rule']['robot_joint'].endswith(('shoulder_pitch_joint','shoulder_roll_joint'))),
                          default=0.)
                core_velocities=d.qvel[self.vadr[:15]]
                supported=(min(self.locomotion.status()['foot_loads_n'])>40
                           and tilt<math.radians(6) and np.max(abs(core_velocities))<1.5
                           and np.linalg.norm(d.qvel[:3])<.2)
                self.preparation_remaining_s=max(0.,.65-(d.time-self.phase_since))
                self.preparation_block_reason=('stance not stable' if not supported else
                                               'shoulders preparing' if error>=.35 else
                                               'preparing support' if self.preparation_remaining_s>0 else '')
                if (d.time-self.phase_since>.65 and error<.35 and supported):
                    self.locomotion.reset()
                    self.locomotion.origin=self.origin.copy()
                    self.locomotion.origin_yaw=self.origin_yaw
                    self._change('STEPPING')
        if self.phase=='STEPPING' and not self.intent_active:
            self._change('SETTLING')
        if self.phase=='SETTLING':
            if self.intent_active:
                self._change('STEPPING')
            else:
                state=self.locomotion.status()
                stable=(min(state['foot_loads_n'])>40 and tilt<math.radians(6)
                        and np.max(abs(d.qvel[self.vadr]))<1. and np.linalg.norm(d.qvel[:3])<.15)
                if stable:
                    if self.stable_since is None:
                        self.stable_since=d.time
                    if d.time-self.stable_since>.25 and d.time-self.phase_since>.5:
                        self.standing.reset()
                        self._change('STANDING')
                else:
                    self.stable_since=None
        if self.phase in ('STANDING','PREPARING STEP'):
            values=self.reference if self.phase=='STANDING' else self._neutral_reference()
            self.standing.set_reference(values,self.entries)
            self.standing.step()
            self.fallen=self.fallen or self.standing.fallen
        else:
            active=self.phase=='STEPPING'
            self.locomotion.update_intent(active,self.root_intent)
            if active:
                self.locomotion.set_reference(self.reference,self.entries)
            else:
                self.locomotion.upper_enabled=False
            self.locomotion.step()
            self.fallen=self.fallen or self.locomotion.fallen
        self.max_torque_ratio=max(self.max_torque_ratio,float(np.max(abs(d.ctrl)/self.model.actuator_ctrlrange[:,1])))

    @property
    def command(self):
        return self.locomotion.command if self.phase in ('STEPPING','SETTLING') else np.zeros(3)

    @property
    def failures(self):
        return self.standing.failures

    def status(self):
        state=(self.standing if self.phase in ('STANDING','PREPARING STEP') else self.locomotion).status()
        feet=self.locomotion.status()
        state.update(state='FALLEN - press R to reset' if self.fallen else self.phase,
                     fallen=self.fallen,velocity_command=self.command.tolist(),
                     foot_clearance_m=feet['foot_clearance_m'],max_torque_limit_ratio=self.max_torque_ratio,
                     phase=self.phase,walking_upper_body_strength=self.locomotion.config['imitation_strength'],
                     preparation_block_reason=self.preparation_block_reason,
                     preparation_remaining_s=self.preparation_remaining_s,
                     solver_backend='Hybrid: inverse dynamics + official Unitree ONNX')
        return state
