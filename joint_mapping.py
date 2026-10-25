"""Configurable estimated human angles -> named MuJoCo hinge position actuators."""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from control import ROOT
from pose_retarget import BodyRetarget, CHANNEL_LABELS
from shadow_motion import ShadowMotion


LANDMARK_NAMES = (
    "nose left_eye_inner left_eye left_eye_outer right_eye_inner right_eye right_eye_outer "
    "left_ear right_ear mouth_left mouth_right left_shoulder right_shoulder left_elbow "
    "right_elbow left_wrist right_wrist left_pinky right_pinky left_index right_index "
    "left_thumb right_thumb left_hip right_hip left_knee right_knee left_ankle "
    "right_ankle left_heel right_heel left_foot_index right_foot_index"
).split()
LANDMARK_IDS = {name: index for index, name in enumerate(LANDMARK_NAMES)}


def source_angle(source, points, image_aspect=None):
    names = source["landmarks"]
    coordinates = [np.array([points[LANDMARK_IDS[name]].x,
                             points[LANDMARK_IDS[name]].y,
                             points[LANDMARK_IDS[name]].z]) for name in names]
    if image_aspect is not None:
        # Image coordinates have separate width/height normalizations; use a
        # common unit and a 2D projection instead of mixing in estimated z.
        coordinates = [np.array([point[0] * image_aspect, point[1], 0.0]) for point in coordinates]
    if not all(np.all(np.isfinite(point)) for point in coordinates):
        return None
    if source["type"] == "angle":
        first, center, last = coordinates
        a, b = first - center, last - center
        length = np.linalg.norm(a) * np.linalg.norm(b)
        if length < 1e-8:
            return None
        return math.degrees(math.acos(float(np.clip(np.dot(a, b) / length, -1, 1))))
    delta = coordinates[1] - coordinates[0]
    axes = ["xyz".index(axis) for axis in source.get("axes", ["x", "y"])]
    if np.linalg.norm(delta[axes]) < 1e-8:
        return None
    return math.degrees(math.atan2(delta[axes[1]], delta[axes[0]]))


class JointRobot:
    def __init__(self, model_path, mapping_path):
        self.model_path = str(Path(model_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.config = json.loads(Path(mapping_path).read_text(encoding="utf-8-sig"))
        self.confidence = float(self.config.get("confidence", 0.6))
        self.smoothing = float(self.config.get("smoothing_seconds", 0.12))
        self.mode = self.config.get("control_mode", "physics")
        if self.mode not in ("physics", "shadow", "standing", "walking", "tracking"):
            raise ValueError("control_mode must be physics, shadow, standing, walking or tracking")
        self.retarget = BodyRetarget(self.confidence, self.config.get("pose_quality", {}))
        self.shadow = None
        self.balance = None
        self.footstep = None
        self.motion_input = None
        if not 0 <= self.confidence <= 1 or self.smoothing < 0:
            raise ValueError("confidence must be 0..1 and smoothing_seconds nonnegative")
        self.rules = self.config.get("rules", [])
        if not self.rules:
            raise ValueError("Mapping must contain at least one rule")
        self.entries = []
        seen = set()
        for rule in self.rules:
            required = ("source", "robot_joint", "human_range_deg", "robot_range_deg")
            if any(key not in rule for key in required):
                raise ValueError(f"Each mapping rule needs: {', '.join(required)}")
            source = rule["source"]
            if "type" not in source:
                raise ValueError("source needs type")
            kind = source["type"]
            if kind not in ("angle", "vector_angle", "retarget"):
                raise ValueError("source.type must be angle, vector_angle or retarget")
            if kind == "retarget" and source.get("channel") not in CHANNEL_LABELS:
                raise ValueError(f"Unknown body channel: {source.get('channel')}")
            required_count = 3 if kind == "angle" else 2
            if kind != "retarget" and len(source.get("landmarks", [])) != required_count:
                raise ValueError(f"{kind} needs {required_count} landmark names")
            for name in source.get("landmarks", []):
                if name not in LANDMARK_IDS:
                    raise ValueError(f"Unknown human landmark: {name}")
            if source.get("space", "world") not in ("world", "image"):
                raise ValueError("source.space must be world or image")
            axes = source.get("axes", ["x", "y"])
            if kind == "vector_angle" and (len(axes) != 2 or len(set(axes)) != 2
                                             or any(a not in ("x", "y", "z") for a in axes)):
                raise ValueError("vector_angle.axes needs two distinct axes from x, y, z")
            if kind == "vector_angle" and source.get("space", "world") == "image" and set(axes) != {"x", "y"}:
                raise ValueError("Image vector angles must use axes x/y; use world for estimated 3D")
            joint_name = rule["robot_joint"]
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint < 0:
                raise ValueError(f"Robot joint not found: {joint_name}")
            if joint in seen:
                raise ValueError(f"Multiple rules target the same joint: {joint_name}")
            seen.add(joint)
            if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"{joint_name}: this angle mapping supports hinge joints only")
            actuator_name = rule.get("actuator")
            if actuator_name:
                actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            else:
                candidates = [i for i in range(self.model.nu)
                              if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                              and self.model.actuator_trnid[i, 0] == joint]
                if len(candidates) != 1:
                    raise ValueError(f"{joint_name}: specify one actuator by name")
                actuator = candidates[0]
            if actuator < 0:
                raise ValueError(f"Actuator not found: {actuator_name}")
            if (self.model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                    or self.model.actuator_trnid[actuator, 0] != joint):
                raise ValueError(f"Actuator {actuator_name} does not drive {joint_name}")
            gain = self.model.actuator_gainprm[actuator, 0]
            bias = self.model.actuator_biasprm[actuator]
            if self.mode in ("standing", "walking", "tracking"):
                if (self.model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                        or self.model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_NONE
                        or not np.isclose(gain, 1) or np.any(bias)):
                    raise ValueError(f"{joint_name}: standing mode requires direct torque motors")
            elif (self.model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                    or self.model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                    or gain <= 0 or not np.isclose(bias[1], -gain)
                    or bias[2] > 0):
                raise ValueError(f"{joint_name}: requires a position servo; torque motors need a controller")
            gear = float(self.model.actuator_gear[actuator, 0])
            if not np.isfinite(gear) or abs(gear) < 1e-9:
                raise ValueError(f"{joint_name}: invalid actuator gear")
            human_range = np.asarray(rule["human_range_deg"], dtype=float)
            robot_range = np.asarray(rule["robot_range_deg"], dtype=float)
            offset = float(rule.get("offset_deg", 0))
            speed = math.radians(float(rule.get("max_speed_deg_s", 60)))
            if (human_range.shape != (2,) or robot_range.shape != (2,)
                    or not np.all(np.isfinite(human_range)) or not np.all(np.isfinite(robot_range))
                    or human_range[0] == human_range[1] or not np.isfinite(offset)
                    or not np.isfinite(speed) or speed <= 0):
                raise ValueError(f"{joint_name}: invalid angle ranges, offset or speed")
            low, high = -np.inf, np.inf
            if self.model.jnt_limited[joint]:
                low, high = self.model.jnt_range[joint]
            if self.model.actuator_ctrllimited[actuator] and self.mode not in ("standing", "walking", "tracking"):
                actuator_limits = np.sort(self.model.actuator_ctrlrange[actuator] / gear)
                low, high = max(low, actuator_limits[0]), min(high, actuator_limits[1])
            if low > high:
                raise ValueError(f"{joint_name}: joint and actuator ranges do not overlap")
            self.entries.append({
                "rule": rule, "joint": joint, "qpos": int(self.model.jnt_qposadr[joint]),
                "actuator": actuator, "gear": gear, "limits": (low, high),
                "human_range": human_range, "robot_range": robot_range,
                "offset": offset, "speed": speed,
            })
        self.reset()
        if self.mode == "shadow":
            self.shadow = ShadowMotion(self.model, self.data, self.config.get("root_tracking", {}))
        elif self.mode == "standing":
            from standing_balance import StandingBalance
            self.balance = StandingBalance(self.model, self.data, self.config.get("balance", {}))
        elif self.mode == "walking":
            from hybrid_balance import HybridBalance
            from footstep_intent import FootstepIntent
            self.balance=HybridBalance(self.model,self.data,self.config.get("locomotion",{}),self.config.get('balance',{}))
            self.footstep=FootstepIntent(self.config.get("locomotion",{}))
        elif self.mode == 'tracking':
            from twist2_camera import TwistCameraBalance
            self.balance=TwistCameraBalance(self.model,self.data,self.config.get('tracking',{}))

    def reset(self):
        self.retarget.reset()
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.command = np.array([self.data.qpos[e["qpos"]] for e in self.entries])
        self.goal = self.command.copy()
        if self.shadow is not None:
            self.shadow.reset()
        if self.balance is not None:
            self.balance.reset()
        if self.footstep is not None:
            self.footstep.reset()
        self.motion_input=None
        self.hold()

    def calibration_ready(self, observation):
        if self.mode=='tracking':
            channels,_=self.retarget.evaluate(observation)
            knees=[channels[s+'_knee_flex'] for s in ('left','right')]
            if not all(c['valid'] and c.get('raw_angle_deg',c['angle'])<45 for c in knees):
                return False
        return ((self.mode not in ("shadow", "standing", "walking", "tracking") or self.retarget.calibration_ready(observation))
                and (self.footstep is None or self.footstep.calibration_ready(observation)))

    def calibrate(self, observation):
        if self.mode in ("shadow", "standing", "walking", "tracking"):
            if not self.calibration_ready(observation) or not self.retarget.calibrate(observation):
                return False
            if self.footstep is not None:
                self.footstep.calibrate(observation,self.retarget.reference_scale)
                self.balance.calibrate_origin()
            if self.mode=='tracking':
                mapped=self.targets_from_pose(observation)
                if mapped is None:
                    return False
                self.balance.calibrate(mapped,self.entries,[d['valid'] for d in self.pose_diagnostics])
            return True
        return True

    def targets_from_pose(self, observation):
        # Rules are independent. An unseen hip must not disable a visible elbow.
        # Invalid rules hold their current setpoint, never their previous goal.
        targets = self.command.copy()
        diagnostics = []
        channels, root = self.retarget.evaluate(observation) if any(
            r["source"]["type"] == "retarget" for r in self.rules) else ({}, {})
        self.root_observation = root
        self.motion_input=observation
        for index, entry in enumerate(self.entries):
            source = entry["rule"]["source"]
            diagnostic = {"joint": entry["rule"]["robot_joint"], "valid": False,
                          "reason": "no person detected", "missing": []}
            diagnostics.append(diagnostic)
            if source["type"] == "retarget":
                channel = channels[source["channel"]]
                diagnostic.update({key: value for key, value in channel.items() if key != "angle"})
                diagnostic["channel"] = source["channel"]
                if not channel["valid"]:
                    continue
                angle = channel["angle"]
                lo, hi = entry["human_range"]
                fraction = np.clip((angle - lo) / (hi - lo), 0, 1)
                first, last = entry["robot_range"]
                target = math.radians(first + fraction * (last - first) + entry["offset"])
                targets[index] = np.clip(target, *entry["limits"])
                diagnostic.update(human_angle_deg=float(angle), target_deg=float(np.degrees(targets[index])))
                continue
            if observation is None or not observation.landmarks:
                continue
            points = (observation.world_landmarks if source.get("space", "world") == "world"
                      else observation.landmarks)
            if not points:
                diagnostic["reason"] = "estimated 3D pose missing"
                continue
            for name in source["landmarks"]:
                point = observation.landmarks[LANDMARK_IDS[name]]
                qualities = [point.visibility or 0, point.presence or 0]
                if not all(np.isfinite(value) and value >= self.confidence for value in qualities):
                    diagnostic["missing"].append(name)
            if diagnostic["missing"]:
                diagnostic["reason"] = "show " + ", ".join(name.replace("_", " ")
                                                          for name in diagnostic["missing"])
                continue
            aspect = None
            if source.get("space", "world") == "image":
                aspect = observation.frame.shape[1] / observation.frame.shape[0]
            angle = source_angle(source, points, aspect)
            if angle is None:
                diagnostic["reason"] = "landmarks cannot define an angle"
                continue
            lo, hi = entry["human_range"]
            fraction = np.clip((angle - lo) / (hi - lo), 0, 1)
            first, last = entry["robot_range"]
            target = math.radians(first + fraction * (last - first) + entry["offset"])
            targets[index] = np.clip(target, *entry["limits"])
            diagnostic.update(valid=True, reason="tracking", human_angle_deg=float(angle),
                              target_deg=float(np.degrees(targets[index])))
        self.pose_diagnostics = diagnostics
        return targets if any(item["valid"] for item in diagnostics) else None

    def set_targets(self, targets):
        targets = np.asarray(targets, dtype=float)
        if targets.shape != self.goal.shape or not np.all(np.isfinite(targets)):
            raise ValueError("Joint targets must contain one finite value per mapping rule")
        bounded=np.array([np.clip(value,*entry["limits"]) for value,entry in zip(targets,self.entries)])
        deadband=math.radians(float(self.config.get('pose_quality',{}).get('deadband_deg',0)))
        self.goal=np.where(abs(bounded-self.goal)<deadband,self.goal,bounded)
        if self.mode=='tracking':
            self.balance.allow_tracking(self.goal,self.entries,[d['valid'] for d in self.pose_diagnostics])
        if self.shadow is not None:
            self.shadow.update(self.root_observation)
        if self.footstep is not None and self.motion_input is not None:
            active=self.footstep.update(self.motion_input)
            self.balance.update_intent(active,self.root_observation)

    def hold(self):
        self.goal = self.command.copy()
        if self.shadow is not None:
            self.shadow.hold()
        if self.footstep is not None:
            self.footstep.hold()
            self.balance.hold()
        elif self.mode=='tracking' and self.balance is not None:
            self.balance.hold()
        if self.balance is not None or self.mode in ("standing", "walking", "tracking"):
            return
        for value, entry in zip(self.command, self.entries):
            self.data.ctrl[entry["actuator"]] = value * entry["gear"]

    def step(self):
        dt = self.model.opt.timestep
        alpha = 1 if self.smoothing == 0 else 1 - math.exp(-dt / self.smoothing)
        delta = alpha * (self.goal - self.command)
        speeds = np.array([entry["speed"] for entry in self.entries])
        if self.balance is not None:
            speeds = np.minimum(speeds, math.radians(float(self.balance.config.get("upper_body_speed_deg_s", 90))))
        self.command += np.clip(delta, -speeds * dt, speeds * dt)
        if self.balance is not None:
            self.balance.set_reference(self.goal if self.mode=='tracking' else self.command, self.entries)
            self.balance.step()
            return
        for value, entry in zip(self.command, self.entries):
            self.data.ctrl[entry["actuator"]] = value * entry["gear"]
            if self.mode == "shadow":
                self.data.qpos[entry["qpos"]] = value
        if self.shadow is not None:
            self.shadow.step(dt)
            self.data.qvel[:] = 0
            self.data.time += dt
            mujoco.mj_forward(self.model, self.data)
            self.shadow.keep_feet_above_floor()
            return
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)


def describe_model(path, output=None):
    model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
    joints = []
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                     for i in range(model.nu)
                     if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                     and model.actuator_trnid[i, 0] == joint]
        joints.append({"joint": name, "type": mujoco.mjtJoint(int(model.jnt_type[joint])).name,
                       "actuators": actuators,
                       "range_deg": np.degrees(model.jnt_range[joint]).tolist()
                       if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
                       and model.jnt_limited[joint] else None})
    document = json.dumps(joints, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(document, encoding="utf-8")
    print(document)
