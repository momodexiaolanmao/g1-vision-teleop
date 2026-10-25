"""Hands-free sampling that starts after calibration and a settling countdown."""
from copy import deepcopy
import json
import math
from pathlib import Path

from check_live_tracking import build_report


class SamplingSession:
    """Collect numeric live-status snapshots without restarting the control loop."""

    def __init__(self, duration=20, output=None, kind='static'):
        if kind not in ('static','motion'):
            raise ValueError('Sampling kind must be static or motion')
        self.kind=kind
        self.duration = float(duration)
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Sampling duration must be positive and finite")
        self.output = Path(output) if output is not None else (
            Path(__file__).resolve().parent / "artifacts" / "unitree_g1_handsfree_static_check.json")
        self.reset()

    def reset(self):
        """Allow a new calibration while preserving any completed report on disk."""
        self.state = "WAITING_CALIBRATION"
        self.samples = []
        self.report = None
        self.sampling_started_at = None
        self.deadline = None
        self._seen_timestamps = set()
        self.physics = None

    def start(self, now):
        """Start once after successful calibration; reset is needed for another run."""
        if self.state != "WAITING_CALIBRATION":
            return False
        self.sampling_started_at = float(now) + 3.0
        self.deadline = self.sampling_started_at + self.duration
        self.state = "SETTLING"
        return True

    def update(self, now, snapshot=None):
        """Return True only when sampling finishes and its report has been saved."""
        if self.state in ("WAITING_CALIBRATION", "FINISHED"):
            return False
        if now < self.sampling_started_at:
            return False
        if now >= self.deadline:
            self.report = build_report(self.samples, self.duration)
            if self.physics is not None:
                p=deepcopy(self.physics)
                self.report['continuous_physics']=p
                balance=self.report.setdefault('physical_balance',{})
                balance['status_snapshots']=balance.get('samples',0)
                balance.update({k:p[k] for k in ('samples','any_fall','min_pelvis_height_m',
                    'max_pelvis_tilt_deg','min_total_foot_load_n','max_total_foot_load_n',
                    'max_other_floor_contacts','max_external_force','max_applied_body_force',
                    'max_torque_limit_ratio')})
                balance['measurement_source']='each physics step within the sampling window'
                steps=self.report.setdefault('physical_steps',{})
                steps.update(foot_clearance_max_m=p['foot_clearance_max_m'],
                             foot_unloaded_samples=p['foot_unloaded_samples'],
                             physics_samples=p['samples'],
                             note='Foot clearance and contact force are accumulated each physics step; human-intent counts use separate status snapshots.')
            self.report["workflow_note"] = (
                "handsfree sampling after calibration; ground truth movement needs operator")
            self.report["sampling_completed"] = True
            self.report['sampling_kind']=self.kind
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
            self.state = "FINISHED"
            return True
        self.state = "SAMPLING"
        if snapshot is not None:
            timestamp = snapshot.get("updated_at_unix")
            if (isinstance(timestamp, (int, float)) and math.isfinite(timestamp)
                    and timestamp not in self._seen_timestamps):
                self.samples.append(deepcopy(snapshot))
                self._seen_timestamps.add(timestamp)
        return False

    def observe_physics(self, now, sim_time, measured):
        """Accumulate brief contacts/lifts without retaining a private trajectory."""
        if (self.state in ('WAITING_CALIBRATION','FINISHED') or
                now < self.sampling_started_at or now >= self.deadline):
            return
        if self.physics is not None and sim_time <= self.physics['last_simulation_s']:
            return
        self.state='SAMPLING'
        if self.physics is None:
            self.physics={'samples':0,'first_simulation_s':float(sim_time),
                'last_simulation_s':float(sim_time),'max_sample_gap_s':0.,
                'first_wall_clock_s':float(now),'last_wall_clock_s':float(now),
                'phase_samples':{},'any_fall':False,'min_pelvis_height_m':float('inf'),
                'max_pelvis_tilt_deg':0.,'min_total_foot_load_n':float('inf'),
                'max_total_foot_load_n':0.,'max_other_floor_contacts':0,
                'max_external_force':0.,'max_applied_body_force':0.,
                'max_torque_limit_ratio':0.,'foot_clearance_max_m':[float('-inf')]*2,
                'foot_unloaded_samples':[0,0]}
        p=self.physics
        p['max_sample_gap_s']=max(p['max_sample_gap_s'],float(sim_time)-p['last_simulation_s'])
        p['last_simulation_s']=float(sim_time)
        p['last_wall_clock_s']=float(now)
        p['samples']+=1
        phase=measured.get('phase','unknown')
        p['phase_samples'][phase]=p['phase_samples'].get(phase,0)+1
        p['any_fall']=p['any_fall'] or bool(measured['fallen'])
        p['min_pelvis_height_m']=min(p['min_pelvis_height_m'],float(measured['height_m']))
        p['max_pelvis_tilt_deg']=max(p['max_pelvis_tilt_deg'],float(measured['tilt_deg']))
        total=sum(measured['foot_loads_n'])
        p['min_total_foot_load_n']=min(p['min_total_foot_load_n'],total)
        p['max_total_foot_load_n']=max(p['max_total_foot_load_n'],total)
        for target,source in (('max_other_floor_contacts','other_floor_contacts'),
                              ('max_external_force','external_force_max'),
                              ('max_applied_body_force','body_force_max'),
                              ('max_torque_limit_ratio','max_torque_limit_ratio')):
            p[target]=max(p[target],measured[source])
        for side in (0,1):
            p['foot_clearance_max_m'][side]=max(p['foot_clearance_max_m'][side],
                                              float(measured['foot_clearance_m'][side]))
            p['foot_unloaded_samples'][side]+=int(measured['foot_loads_n'][side]<5.)

    def observe_robot(self, now, robot):
        """Measure the actual free-body simulation, including genuine contacts."""
        if (self.state in ('WAITING_CALIBRATION','FINISHED') or
                now < self.sampling_started_at or now >= self.deadline):
            return
        import numpy as np
        from physics_benchmark import measure
        balance=robot.balance
        controller=(balance.locomotion if hasattr(balance,'locomotion') else
                    balance.controller if hasattr(balance,'controller') else balance)
        # Standing-only profiles have no foot-clearance interface.
        if not hasattr(controller,'foot_geoms'):
            return
        model,data=robot.model,robot.data
        measured=measure(model,data)
        measured['foot_clearance_m']=[min(float(data.geom_xpos[g,2]-model.geom_size[g,0])
            for g in controller.foot_geoms if model.geom_bodyid[g]==body)
            for body in controller.feet]
        torque_limits=(np.maximum(np.abs(controller.torque_low),np.abs(controller.torque_high))
                       if hasattr(controller,'torque_high') else
                       np.max(np.abs(model.actuator_ctrlrange),axis=1))
        measured['max_torque_limit_ratio']=float(np.max(np.abs(data.ctrl)/torque_limits))
        measured['fallen']=balance.fallen
        measured['phase']=getattr(balance,'phase','TRACKING' if getattr(balance,'tracking_enabled',False) else 'HOLDING')
        self.observe_physics(now,data.time,measured)

    def display(self, now):
        """A persistent, plain-text cue suitable for the camera window overlay."""
        if self.state == "WAITING_CALIBRATION":
            return "WAITING: walk into camera, full body, arms relaxed"
        if self.state == "FINISHED":
            return "SAMPLING FINISHED - you may move"
        if self.state == "SETTLING":
            remaining = max(0, math.ceil(self.sampling_started_at - now))
            return f"CALIBRATED: stand still - sampling starts in {remaining}s"
        remaining = max(0, math.ceil(self.deadline - now))
        if self.kind=='motion':
            return f'MOVEMENT SAMPLING: shift weight, alternate knee lifts - {remaining}s remaining'
        return f"SAMPLING: stand still - {remaining}s remaining"
