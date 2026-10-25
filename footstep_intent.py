"""Visible human foot lifts trigger locomotion; static noise must not."""
import math
import numpy as np

from pose_retarget import IDS


class FootstepIntent:
    def __init__(self, config=None):
        self.config=config or {}
        threshold=float(self.config.get('foot_lift_threshold_m',.035))
        if not np.isfinite(threshold) or not .02<=threshold<=.1:
            raise ValueError('Foot lift threshold must be 0.02..0.1 meters')
        bend=float(self.config.get('min_knee_bend_deg',12))
        if not np.isfinite(bend) or not 5<=bend<=45:
            raise ValueError('Step knee bend change must be 5..45 degrees')
        self.request_hold_s=float(self.config.get('step_request_hold_s',1.25))
        if not np.isfinite(self.request_hold_s) or not .25<=self.request_hold_s<=3.:
            raise ValueError('Step request hold must be 0.25..3 seconds')
        self.reset()

    def reset(self):
        self.reference_y=None
        self.reference_flex=None
        self.flex_change=np.zeros(2)
        self.scale=None
        self.lift=np.zeros(2)
        self.active=False
        self.last_time=None
        self.onset=None
        self.until=-math.inf
        self.state='press C with both feet visible'

    def _feet(self, observation):
        points=getattr(observation,'landmarks',None)
        if points is None or len(points)!=33:
            return None
        result=[]
        for side in ('left','right'):
            indices=[IDS[f'{side}_{n}'] for n in ('ankle','heel','foot_index')]
            if not all(min(points[i].visibility or 0,points[i].presence or 0)>=.65
                       and 0<=points[i].x<=1 and 0<=points[i].y<=1 for i in indices):
                return None
            result.append(np.mean([points[i].y for i in indices[1:]]))
        return np.array(result)

    def calibration_ready(self, observation):
        flex=self._knee_flex(observation)
        return self._feet(observation) is not None and flex is not None and np.max(flex)<45

    def _knee_flex(self, observation):
        world=getattr(observation,'world_landmarks',None)
        points=getattr(observation,'landmarks',None)
        if world is None or points is None or len(world)!=33 or len(points)!=33:
            return None
        values=[]
        for side in ('left','right'):
            indices=[IDS[f'{side}_{n}'] for n in ('hip','knee','ankle')]
            if not all(min(points[i].visibility or 0,points[i].presence or 0)>=.65 for i in indices):
                return None
            xyz=np.array([[world[i].x,world[i].y,world[i].z] for i in indices])
            thigh,shin=np.diff(xyz,axis=0)
            lengths=np.linalg.norm([thigh,shin],axis=1)
            if not np.all(np.isfinite(xyz)) or not np.all((lengths>.18)&(lengths<.65)):
                return None
            values.append(np.degrees(np.arccos(np.clip(np.dot(thigh,shin)/np.prod(lengths),-1,1))))
        return np.array(values)

    def calibrate(self, observation, torso_image_scale):
        feet=self._feet(observation)
        flex=self._knee_flex(observation)
        if feet is None or flex is None or np.max(flex)>=45 or torso_image_scale<.025:
            return False
        self.reset()
        self.reference_y=feet
        self.reference_flex=flex
        self.scale=.42/torso_image_scale
        self.state='READY - bend knee and lift a foot'
        return True

    def update(self, observation):
        feet=self._feet(observation)
        stamp=float(getattr(observation,'arrived',0))
        if feet is None or self.reference_y is None:
            self.active=False
            self.onset=None
            self.until=-math.inf
            self.state='FEET HIDDEN - walking stopped' if self.reference_y is not None else 'press C with both feet visible'
            return False
        if self.last_time is None or stamp<=self.last_time:
            self.last_time=stamp
            return self.active
        dt=min(.2,stamp-self.last_time)
        self.last_time=stamp
        alpha=1-math.exp(-dt/.06)
        relative=self.reference_y-feet
        # The grounded foot is the height reference. Both image feet moving
        # together (camera motion, body translation or scale drift) is not an
        # alternating foot lift. Jump detection is deliberately separate.
        relative-=np.min(relative)
        self.lift+=alpha*(relative*self.scale-self.lift)
        threshold=float(self.config.get('foot_lift_threshold_m',.035))
        flex=self._knee_flex(observation)
        if flex is None or self.reference_flex is None:
            self.hold()
            self.state='LEGS UNCLEAR - walking stopped'
            return False
        self.flex_change=flex-self.reference_flex
        lifted=bool(np.any((self.lift>threshold)&
                          (self.flex_change>float(self.config.get('min_knee_bend_deg',12)))))
        if lifted:
            if self.onset is None:
                self.onset=stamp
            if stamp-self.onset>=.08:
                # A confirmed step requests locomotion across the following
                # grounded interval. Preparation must outlive a short lift;
                # fresh step evidence renews this bounded request.
                self.until=stamp+self.request_hold_s
        else:
            self.onset=None
        self.active=stamp<self.until
        self.state='HUMAN STEPPING' if self.active else 'READY - bend knee and lift a foot'
        return self.active

    def hold(self):
        self.active=False
        self.onset=None
        self.until=-math.inf
        if self.reference_y is not None:
            self.state='INPUT HELD - walking stopped'

    def status(self):
        return {'state':self.state,'active':self.active,'human_foot_lift_m':self.lift.tolist(),
                'knee_bend_change_deg':self.flex_change.tolist(),
                'step_request_remaining_s':max(0.,self.until-(self.last_time or 0.))}
