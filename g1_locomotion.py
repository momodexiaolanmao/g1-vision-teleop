"""Official G1-29dof velocity policy, adapted for local torque-only MuJoCo.

Policy observations/actions retain the upstream order and five-frame history.
Upper-body imitation is an additional controller reference, not learned from
the webcam. No free-base pose/velocity or external force is set by this code.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).absolute().parent


class G1Locomotion:
    def __init__(self, model, data, config=None):
        import onnxruntime as ort
        self.model, self.data, self.config = model, data, config or {}
        strength=float(self.config.get('imitation_strength',.5))
        speed=float(self.config.get('upper_body_speed_deg_s',120))
        if not (np.isfinite(strength) and 0<=strength<=.5 and np.isfinite(speed) and 0<speed<=120):
            raise ValueError('Walking imitation strength must be 0..0.5, upper-body speed 0..120 deg/s')
        self.step_speed=float(self.config.get('step_speed_m_s',.35))
        self.step_half_span=float(self.config.get('step_half_span_m',.25))
        if not np.isfinite(self.step_speed) or not .25<=self.step_speed<=.45:
            raise ValueError('Step speed must be 0.25..0.45 meters/second')
        if not np.isfinite(self.step_half_span) or not .15<=self.step_half_span<=.40:
            raise ValueError('Step half span must be 0.15..0.40 meters')
        folder = ROOT/'policies/g1_velocity'
        source = json.loads((folder/'SOURCE.json').read_text())
        if hashlib.sha256((folder/'policy.onnx').read_bytes()).hexdigest() != source['policy_sha256']:
            raise ValueError('G1 policy checksum mismatch')
        self.policy_config = json.loads((folder/'deploy.json').read_text())
        self.order = np.array(self.policy_config['joint_ids_map'], dtype=int)
        self.default = np.array(self.policy_config['default_joint_pos'])
        self.kp = np.array(self.policy_config['stiffness'])
        self.kd = np.array(self.policy_config['damping'])
        self.period = float(self.policy_config['step_dt'])
        if (model.nv != 35 or model.nu != 29 or model.neq or model.nmocap
                or np.any(model.body_gravcomp) or not np.allclose(model.opt.gravity,[0,0,-9.81])):
            raise ValueError('G1 locomotion requires the unconstrained official 29-joint gravity model')
        self.joints = np.array([int(model.actuator_trnid[a,0]) for a in range(model.nu)])
        self.qadr = model.jnt_qposadr[self.joints]
        self.vadr = model.jnt_dofadr[self.joints]
        self.names = [model.joint(int(j)).name for j in self.joints]
        expected = [s+'_'+n+'_joint' for s in ['left','right']
                    for n in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
        expected += ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
        expected += [s+'_'+n+'_joint' for s in ['left','right']
                     for n in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]
        if self.names != expected or not np.allclose(model.actuator_gear[:,0],1):
            raise ValueError('G1 model joint order does not match the policy deployment map')
        self.index = {n:i for i,n in enumerate(self.names)}
        self.pelvis = model.body('pelvis').id
        self.feet = [model.body(s+'_ankle_roll_link').id for s in ['left','right']]
        self.foot_geoms = [i for i in range(model.ngeom)
            if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_bodyid[i] in self.feet]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(folder/'policy.onnx'), sess_options=options,
                                          providers=['CPUExecutionProvider'])
        if self.policy.get_inputs()[0].shape != [1,480] or self.policy.get_outputs()[0].shape != [1,29]:
            raise ValueError('Unexpected G1 velocity policy interface')
        self.reset()

    def reset(self):
        self.command = np.zeros(3)
        self.action = np.zeros(29)
        self.reference = np.empty(29)
        self.reference[self.order] = self.default
        self.target = self.reference.copy()
        self.upper_offset = np.zeros(29)
        self.upper_enabled = False
        self.histories = None
        self.last_update = -math.inf
        self.last_torque = np.zeros(29)
        self.max_torque_ratio = 0.
        self.fallen = False
        self.solves, self.failures = 0, 0
        self.limited_joints = []
        self.solve_time_ms = 0.
        self.intent_active=False
        self.use_intent=False
        self.root_intent={}
        # Preparing a controller resets policy history, but a brief stand or
        # settle must not replace a return step with another forward start.
        self.step_direction=getattr(self,'step_direction',1)
        self.intent_phase=getattr(self,'intent_phase',0.)
        self.calibrate_origin(reset_step_direction=False)

    def calibrate_origin(self,reset_step_direction=True):
        self.origin=self.data.qpos[:2].copy()
        r=self.data.xmat[self.pelvis].reshape(3,3)
        self.origin_yaw=math.atan2(r[1,0],r[0,0])
        if reset_step_direction:
            self.step_direction=1
            self.intent_phase=0.

    def update_intent(self, active, root):
        self.use_intent=True
        self.intent_active=bool(active)
        self.root_intent=root.copy()

    def _intent_velocity(self):
        if not self.intent_active:
            return np.zeros(3)
        self.intent_phase+=self.period
        root=self.root_intent
        yaw=self.origin_yaw
        r0=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        lateral=float(np.clip(root.get('lateral_m',0),-1.,1.)) if root.get('valid') else 0.
        local_position=r0.T@(self.data.qpos[:2]-self.origin)
        if local_position[0]>=self.step_half_span:
            self.step_direction=-1
        elif local_position[0]<=-self.step_half_span:
            self.step_direction=1
        r=self.data.xmat[self.pelvis].reshape(3,3)
        current_yaw=math.atan2(r[1,0],r[0,0])
        rb=np.array([[math.cos(current_yaw),-math.sin(current_yaw)],
                     [math.sin(current_yaw),math.cos(current_yaw)]])
        # Lateral tracking is bounded separately. An X restoring term must
        # not cancel the stepping speed when the robot has moved from origin.
        velocity=rb.T@(r0@np.array([0.,.8*(lateral-local_position[1])]))
        velocity[0]=self.step_direction*self.step_speed
        wanted_yaw=self.origin_yaw+float(root.get('yaw_rad',0))
        error=math.atan2(math.sin(wanted_yaw-current_yaw),math.cos(wanted_yaw-current_yaw))
        return np.array([velocity[0],np.clip(velocity[1],-.2,.2),np.clip(.5*error,-.15,.15)])

    def set_velocity(self, command):
        command = np.asarray(command,dtype=float)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError('Velocity command needs forward/left/yaw finite values')
        self.command = np.clip(command,[-.5,-.3,-.2],[1.,.3,.2])

    def set_reference(self, mapped_angles, entries):
        for value, entry in zip(mapped_angles, entries):
            index = self.index[entry['rule']['robot_joint']]
            if index >= 12:
                self.reference[index] = value
        self.upper_enabled = True

    def hold(self):
        self.intent_active=False
        self.set_velocity([0,0,0])

    def _observation(self):
        d,m = self.data,self.model
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.pelvis,velocity,1)
        rotation = d.xmat[self.pelvis].reshape(3,3)
        return [velocity[:3]*.2, rotation.T @ np.array([0.,0.,-1.]),self.command.copy(),
                d.qpos[self.qadr[self.order]]-self.default,
                d.qvel[self.vadr[self.order]]*.05,self.action.copy()]

    def _update(self):
        started = time.perf_counter()
        if self.use_intent:
            self.command=self._intent_velocity()
        observations = self._observation()
        if self.histories is None:
            self.histories = [np.tile(o,(5,1)) for o in observations]
        for h,o in zip(self.histories,observations):
            h[:-1] = h[1:]
            h[-1] = o
        vector = np.concatenate([h.ravel() for h in self.histories]).astype(np.float32)[None,:]
        action = self.policy.run(None,{'obs':vector})[0][0]
        if not np.all(np.isfinite(action)):
            raise RuntimeError('G1 policy produced invalid motor targets')
        self.action = action.copy()
        self.target[self.order] = self.default + .25 * self.action
        # Smooth only the added human offset. Rate limiting or clipping the
        # complete policy target also suppresses its trained balance response,
        # even with imitation strength zero. That can fail after many steps.
        tilt=math.acos(float(np.clip(self.data.xmat[self.pelvis].reshape(3,3)[2,2],-1,1)))
        support_scale=float(np.clip((math.radians(18)-tilt)/math.radians(10),0,1))
        strength=float(self.config.get('imitation_strength',.5))
        neutral=np.zeros(29)
        neutral[18]=neutral[25]=math.pi/2
        speed=math.radians(float(self.config.get('upper_body_speed_deg_s',120)))*max(.05,strength)
        for i in range(12,29):
            offset=float(np.clip(self.reference[i]-neutral[i],-1.5,1.5))
            if self.names[i].startswith('waist_'):
                axis=self.names[i].split('_')[1]
                limit=math.radians({'yaw':18,'roll':6,'pitch':8}[axis])
                offset=float(np.clip(offset,-limit,limit))
            goal=strength*support_scale*offset if self.upper_enabled else 0.
            self.upper_offset[i]+=float(np.clip(goal-self.upper_offset[i],-speed*self.period,speed*self.period))
            self.target[i]+=self.upper_offset[i]
        limits = self.model.jnt_range[self.joints]
        self.target = np.clip(self.target,limits[:,0]+.01,limits[:,1]-.01)
        self.solve_time_ms=1000*(time.perf_counter()-started)
        self.solves += 1

    def step(self):
        m,d=self.model,self.data
        if d.qpos[2]<.45 or d.xmat[self.pelvis].reshape(3,3)[2,2]<math.cos(math.radians(50)):
            self.fallen = True
        if not self.fallen and d.time-self.last_update>=self.period-1e-9:
            self._update()
            self.last_update = d.time
        torque=(-3*d.qvel[self.vadr] if self.fallen else
                self.kp*(self.target-d.qpos[self.qadr])-self.kd*d.qvel[self.vadr])
        self.last_torque = np.clip(torque,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        d.ctrl[:] = self.last_torque
        self.max_torque_ratio = max(self.max_torque_ratio,float(np.max(abs(d.ctrl)/m.actuator_ctrlrange[:,1])))
        mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)

    def status(self):
        from physics_benchmark import measure
        measured=measure(self.model,self.data)
        clearances=[min(float(self.data.geom_xpos[g,2]-self.model.geom_size[g,0])
                        for g in self.foot_geoms if self.model.geom_bodyid[g]==body) for body in self.feet]
        state='STEPPING' if np.linalg.norm(self.command)>.025 else 'BALANCING'
        return {'state':'FALLEN - press R to reset' if self.fallen else state,'fallen':self.fallen,
                'solver_status':'official policy','qp_solves':self.solves,'qp_failures':0,
                'solver_backend':'Unitree G1-29dof ONNX','solve_ms':self.solve_time_ms,
                'max_torque_limit_ratio':self.max_torque_ratio,'limited_joints':self.limited_joints,
                'feet_planted':False,'kinematic_preview':False,'velocity_command':self.command.tolist(),
                'intent_phase':self.intent_phase,'step_direction':self.step_direction,
                'foot_clearance_m':clearances,**measured}
