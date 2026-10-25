"""DJI USB webcam -> MediaPipe pose -> named MuJoCo robot joints or an arm demo."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time

import mujoco
import numpy as np

from control import Arm, WristMapper
from download_model import MODEL_PATH
from joint_mapping import JointRobot, describe_model
from tracking_session import TrackingSession, CalibrationGate
from sampling_session import SamplingSession
from workflow_display import render_workflow_banner
from subject_tracker import SubjectTracker


@dataclass
class Observation:
    arrived: float
    processed: float
    sample: object
    frame: object
    landmarks: object
    world_landmarks: object = None
    error: str = ""
    subject_state: str = ""
    subject_count: int = 0
    subject_revision: int = 0


class CameraPose:
    """Continuously capture into one latest-frame slot; infer without queuing."""
    def __init__(self, args):
        import cv2
        import mediapipe as mp
        self.cv2, self.mp = cv2, mp
        self.args = args
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.subject=SubjectTracker()
        self.subject_revision=0
        self.subject_request=None
        self.frame = None
        self.sequence = 0
        self.observation = None
        self.error = ""
        backend = {
            "auto": cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "v4l2": cv2.CAP_V4L2,
        }[args.backend]
        self.capture = cv2.VideoCapture(args.camera, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                f"Cannot open camera {args.camera}. Select Webcam on DJI, close other "
                "camera apps, or try --camera 1 / --backend msmf."
            )
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Camera requested:", args.width, "x", args.height, "@ 30 fps")
        print("Camera reported:", int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "x", int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              "@", self.capture.get(cv2.CAP_PROP_FPS), "fps")
        self.threads = [
            threading.Thread(target=self.capture_loop, daemon=True),
            threading.Thread(target=self.pose_loop, daemon=True),
        ]
        for worker in self.threads:
            worker.start()

    def capture_loop(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                arrived = time.monotonic()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame read failed; check USB connection")
                with self.lock:
                    self.frame = (arrived, frame)
                    self.sequence += 1
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def pose_loop(self):
        try:
            mp, cv2 = self.mp, self.cv2
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.args.model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=getattr(self.args,'max_people',1),
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            previous_sequence, previous_stamp = -1, -1
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
                while not self.stop_event.is_set():
                    with self.lock:
                        current, sequence = self.frame, self.sequence
                    if current is None or sequence == previous_sequence:
                        self.stop_event.wait(0.005)
                        continue
                    previous_sequence = sequence
                    arrived, frame = current
                    if time.monotonic() - arrived > 0.35:
                        continue
                    stamp = max(previous_stamp + 1, int(arrived * 1000))
                    previous_stamp = stamp
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    )
                    result = detector.detect_for_video(image, stamp)
                    selected, revision=self._select_subject(result.pose_landmarks,frame.shape[1]/frame.shape[0])
                    sample = None
                    landmarks = result.pose_landmarks[selected] if selected is not None else []
                    if landmarks:
                        shoulder_id, elbow_id, wrist_id = (
                            (12, 14, 16) if self.args.hand == "right" else (11, 13, 15)
                        )
                        needed = (11, 12, shoulder_id, elbow_id, wrist_id)
                        quality = all(
                            (landmarks[index].visibility or 0.0) >= 0.6 and
                            (landmarks[index].presence or 0.0) >= 0.6 for index in needed
                        )
                        shoulder_width = math.hypot(
                            landmarks[11].x - landmarks[12].x,
                            landmarks[11].y - landmarks[12].y,
                        )
                        if quality and shoulder_width > 0.04:
                            shoulder, wrist = landmarks[shoulder_id], landmarks[wrist_id]
                            depth = 0.0
                            if self.args.depth and result.pose_world_landmarks:
                                world = result.pose_world_landmarks[selected]
                                depth = world[wrist_id].z - world[shoulder_id].z
                            sample = np.array([
                                (wrist.x - shoulder.x) / shoulder_width,
                                (wrist.y - shoulder.y) / shoulder_width,
                                depth,
                            ])
                            if not np.all(np.isfinite(sample)):
                                sample = None
                    world_landmarks = result.pose_world_landmarks[selected] if selected is not None and result.pose_world_landmarks else []
                    observation = Observation(arrived, time.monotonic(), sample, frame, landmarks, world_landmarks)
                    observation.subject_state=self.subject.state
                    observation.subject_count=len(result.pose_landmarks)
                    observation.subject_revision=revision
                    self._publish_observation(observation)
        except Exception as exc:
            with self.lock:
                self.error = f"Pose processing failed: {exc}"

    def latest(self):
        with self.lock:
            return self.observation, self.error

    def reacquire_subject(self):
        with self.lock:
            self.subject_revision+=1
            self.subject_request=(self.subject_revision,None)
            self.observation=None

    def lock_subject(self, observation):
        if observation is None or len(observation.landmarks)!=33:
            return False
        candidate=self.subject._candidate(observation.landmarks,
                                          observation.frame.shape[1]/observation.frame.shape[0])
        if candidate is not None:
            with self.lock:
                if observation.subject_revision!=self.subject_revision:
                    return False
                self.subject_revision+=1
                self.subject_request=(self.subject_revision,candidate)
                # A provisional observation cannot control the robot while the
                # inference thread has yet to acknowledge the explicit lock.
                self.observation=None
            return True
        return False

    def _select_subject(self, poses, aspect):
        with self.lock:
            request=self.subject_request
            self.subject_request=None
            revision=self.subject_revision
        if request is not None:
            target=request[1]
            if target is None:
                self.subject.reset()
            else:
                self.subject.lock_to(*target)
        return self.subject.select(poses,aspect), revision

    def _publish_observation(self, observation):
        with self.lock:
            if observation.subject_revision==self.subject_revision:
                self.observation=observation

    def close(self):
        self.stop_event.set()
        self.capture.release()
        for worker in self.threads:
            worker.join(timeout=1)
        self.cv2.destroyAllWindows()


def demo_target(arm, seconds):
    return arm.home_target + np.array([
        0.015 * math.sin(0.7 * seconds),
        0.12 * math.sin(0.8 * seconds),
        0.09 * math.sin(0.6 * seconds),
    ])


def gait_workflow(arm):
    if not isinstance(arm,JointRobot) or arm.mode!='walking' or not arm.retarget.calibrated:
        return None
    state=arm.balance.status()
    if state['fallen']:
        return 'GAIT FALLEN: press R, then C'
    if state['phase']=='PREPARING STEP':
        return 'GAIT PREPARING: '+state.get('preparation_block_reason','preparing support')
    if state['phase']=='STEPPING':
        return 'GAIT RUNNING: stop lifting to settle'
    if state['phase']=='SETTLING':
        return 'GAIT SETTLING: both feet down'
    intent=arm.footstep.status()['state']
    if intent.startswith('FEET HIDDEN'):
        return 'GAIT FEET HIDDEN: show ankles, heels and toes'
    if intent.startswith('LEGS UNCLEAR'):
        return 'GAIT LEGS UNCLEAR: show hips, knees and ankles'
    if intent.startswith('INPUT HELD'):
        return 'GAIT INPUT HELD: return to camera'
    return 'GAIT READY: bend knee and lift a foot'


def draw_camera(source, observation, status, arm, workflow_text=None):
    cv2 = source.cv2
    if observation is None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
    else:
        frame = observation.frame.copy()
        height, width = frame.shape[:2]
        for left, right in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                            (24, 26), (26, 28), (27, 29), (29, 31), (27, 31),
                            (28, 30), (30, 32), (28, 32), (15, 17), (15, 19),
                            (17, 19), (16, 18), (16, 20), (18, 20)]:
            if observation.landmarks:
                a, b = observation.landmarks[left], observation.landmarks[right]
                if min(a.visibility or 0, b.visibility or 0) > 0.6:
                    cv2.line(frame, (int(a.x * width), int(a.y * height)),
                             (int(b.x * width), int(b.y * height)), (40, 220, 90), 3)
        # Mirror only the display; inference and anatomical hand selection are unmirrored.
        frame = cv2.flip(frame, 1)
    latency = "--" if observation is None else f"{1000 * (observation.processed - observation.arrived):.0f}"
    detail = (f"Tool error: {1000 * np.linalg.norm(arm.target - arm.tool_position):.1f} mm"
              if isinstance(arm, Arm) else arm.config.get("display_name", "Custom mapping")
              + f" | {len(arm.entries)} robot joints")
    texts = [status, "C: calibrate/reset | Space: pause/resume | Q: quit",
             f"Host frame-to-pose: {latency} ms (excludes camera/USB delay)",
             detail]
    if isinstance(arm, JointRobot) and arm.mode in ("shadow", "standing", "walking", "tracking"):
        diagnostics = getattr(arm, "pose_diagnostics", [])
        texts[3] += " | " + ("PHYSICS / BALANCE" if arm.mode in ("standing","walking","tracking") else "SHADOW PREVIEW")
        if observation is not None:
            texts.append(observation.subject_state+' | N: select foreground person again')
        groups = [("Arms / wrists", "shoulder", "elbow", "wrist"),
                  ("Waist", "waist"), ("Legs / ankles", "hip", "knee", "ankle")]
        for title, *tokens in groups:
            entries = [d for d in diagnostics if any(t in d["joint"] for t in tokens)]
            texts.append(f"{title}: {sum(d['valid'] for d in entries)}/{len(entries)} tracking")
        if arm.balance is not None:
            balance = arm.balance.status()
            texts[1] = "C: calibrate | Space: pause/resume | R: reset fallen robot | Q: quit"
            texts.append(f"{balance['state']} | tilt {balance['tilt_deg']:.1f} deg | "
                         f"foot loads {balance['foot_loads_n'][0]:.0f} / {balance['foot_loads_n'][1]:.0f} N")
            if arm.footstep is not None:
                intent=arm.footstep.status()
                texts.append(intent['state']+f" | foot lifts {100*intent['human_foot_lift_m'][0]:.1f} / {100*intent['human_foot_lift_m'][1]:.1f} cm")
                texts.append("Lift either foot to step; legs balance, upper body follows at limited amplitude")
            elif arm.mode=='tracking':
                texts.append('TWIST2 full-body tracking: feet follow pose, no foot-lift start gate')
            else:
                texts.append("Feet stay planted; legs maintain balance and follow crouching")
        else:
            root = arm.shadow.status()
            texts.append(f"Body: turn {root['yaw_deg']:+.0f} deg | "
                         f"position {root['position_m'][0]:+.2f}, {root['position_m'][1]:+.2f}, {root['position_m'][2]:.2f} m")
        if not arm.retarget.calibrated:
            texts.append("Stand upright, head and feet in view, then press C")
        elif any(not d["valid"] and any(t in d["joint"] for t in ("hip", "knee", "ankle")) for d in diagnostics):
            texts.append("Keep both feet in frame; hidden / ambiguous joints HOLD")
    elif isinstance(arm, JointRobot):
        for item in getattr(arm, "pose_diagnostics", []):
            if item["valid"]:
                message = f"{item['joint']}: ready | human {item['human_angle_deg']:.0f} deg -> target {item['target_deg']:.0f} deg"
            else:
                message = f"{item['joint']}: HOLD | {item['reason']}"
            texts.append(message)
    panel = frame.copy()
    offset=80 if workflow_text else 0
    text_width = max(cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0][0] for message in texts)
    cv2.rectangle(panel, (0, offset), (min(frame.shape[1], text_width + 24),
                                 min(frame.shape[0], offset+27 * len(texts) + 12)), (16, 20, 24), -1)
    frame = cv2.addWeighted(panel, 0.70, frame, 0.30, 0)
    for row, message in enumerate(texts):
        cv2.putText(frame, message, (12, offset+26 + 27 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2, cv2.LINE_AA)
    if workflow_text:
        frame=render_workflow_banner(frame,workflow_text)
    cv2.imshow("DJI pose control", frame)
    return cv2.waitKey(1) & 0xFF


def write_live_status(path, arm, session, observation, valid, source_error=""):
    """Local diagnostic snapshot contains state/numbers, no captured images."""
    tick = time.monotonic()
    report = {
        "updated_at_unix": time.time(), "enabled": session.enabled,
        "start_pending": session.start_pending, "input_valid": bool(valid),
        "state": session.status(valid, isinstance(arm, JointRobot)),
        "frame_age_ms": None if observation is None else 1000 * (tick - observation.arrived),
        "source_error": source_error,
        "rules": getattr(arm, "pose_diagnostics", []),
        "camera_subject":{'state':getattr(observation,'subject_state',''),
                          'detected_people':getattr(observation,'subject_count',0)},
    }
    if isinstance(arm, JointRobot):
        report["robot_model"] = arm.model_path
        report["command_deg"] = np.degrees(arm.command).tolist()
        if arm.mode=='tracking':
            reference=arm.balance.reference[6:]
            report['command_deg']=[float(np.degrees(reference[arm.balance.index[e['rule']['robot_joint']]])) for e in arm.entries]
        report["actual_deg"] = [float(np.degrees(arm.data.qpos[e["qpos"]])) for e in arm.entries]
        report["control_mode"] = arm.mode
        if arm.shadow is not None:
            report["root_tracking"] = arm.shadow.status()
            report["calibrated"] = arm.retarget.calibrated
        if arm.balance is not None:
            report["balance"] = arm.balance.status()
            report["calibrated"] = arm.retarget.calibrated
            report["motor_torque_nm"] = arm.data.ctrl.tolist()
            report["reference_note"] = "Human references; support joints are adjusted by balance constraints"
            if arm.mode=='tracking':
                report['reference_note']='Calibrated relative human motion around TWIST2 neutral; physical reference may be limited for balance.'
        if arm.footstep is not None:
            report['footstep_intent']=arm.footstep.status()
        report['pose_quality']={'enabled':bool(arm.config.get('pose_quality',{}).get('enabled')),
                                'rejected_bones':getattr(arm.retarget,'rejected_bones',[]),
                                'deadband_deg':arm.config.get('pose_quality',{}).get('deadband_deg',0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(path)
    except PermissionError:
        # Windows readers/antivirus can briefly lock the diagnostic file.
        # Preserve the previous complete snapshot and keep control running;
        # the next periodic update retries. Timed sampling uses this report.
        report['diagnostic_write_error']='file busy; retry at next status update'
    return report


def workflow_sound():
    if sys.platform=='win32':
        import winsound
        try:
            winsound.PlaySound('SystemAsterisk',winsound.SND_ALIAS|winsound.SND_ASYNC)
        except RuntimeError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="synthetic targets, no webcam or model")
    parser.add_argument("--headless", action="store_true", help="demo without GUI")
    parser.add_argument("--seconds", type=float, default=0, help="auto-exit after N seconds")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "v4l2"], default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument('--max-people',type=int,choices=(1,2,3),default=1,
                        help='default one-person tracking retains built-in temporal landmark smoothing')
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--depth", action="store_true", help="enable estimated monocular world-z")
    parser.add_argument("--scale", type=float, default=0.35,
                        help="xy meters per shoulder-width; depth meters per estimated meter")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--mapping", type=Path, help="JSON human angle -> robot joint rules")
    parser.add_argument("--robot", type=Path, default=Path(__file__).parent / "arm.xml",
                        help="MuJoCo XML model used with --mapping or --list-joints")
    parser.add_argument("--list-joints", action="store_true", help="show robot joints and actuators, then exit")
    parser.add_argument("--output", type=Path, help="save --list-joints output for the mapping editor")
    parser.add_argument("--record-control", type=Path, help="optional numeric G1 control trace; no images")
    parser.add_argument("--auto-calibrate", action='store_true', help="wait for relaxed upright G1 pose, then start")
    parser.add_argument('--calibration-delay',type=float,default=8.,help='seconds to leave the computer before G1 calibration')
    parser.add_argument('--sample-seconds',type=float,default=0.,help='timed hands-free static check after calibration')
    parser.add_argument('--sample-kind',choices=('static','motion'),default='static')
    parser.add_argument('--sample-output',type=Path,default=Path('artifacts/unitree_g1_handsfree_static_check.json'))
    parser.add_argument('--record-pose',type=Path,help='optional numeric pose trace during timed sampling; no images')
    args = parser.parse_args()
    if args.list_joints:
        describe_model(args.robot, args.output)
        return
    if args.mapping and args.demo:
        parser.error("--demo uses the bundled arm's Cartesian trajectory; mapping mode uses camera input")
    if not args.mapping and args.robot.resolve() != (Path(__file__).parent / "arm.xml").resolve():
        parser.error("A custom --robot requires --mapping")
    if args.headless and not args.demo:
        parser.error("--headless requires --demo; calibration uses the camera window")
    if args.seconds < 0 or args.scale <= 0 or args.calibration_delay<0 or args.sample_seconds<0:
        parser.error("--seconds must be nonnegative and --scale positive")
    if not args.demo and not args.model.is_file():
        parser.error(f"Missing pose model: {args.model}. Run download_model.py first.")
    if args.headless:
        arm = Arm()
        duration = args.seconds or 10.0
        errors = []
        for frame in range(int(duration * 50)):
            arm.set_target(demo_target(arm, frame / 50))
            for _ in range(10):
                arm.step()
            if frame > 50:
                errors.append(float(np.linalg.norm(arm.target - arm.tool_position)))
        print(f"Demo finished: {arm.data.time:.2f} simulated seconds; "
              f"mean error={1000 * np.mean(errors):.2f} mm" if errors else "Demo finished")
        return

    import mujoco.viewer
    arm = JointRobot(args.robot, args.mapping) if args.mapping else Arm()
    recorder=None
    if args.record_control:
        if not isinstance(arm,JointRobot) or arm.mode!='walking':
            parser.error('--record-control requires the walking G1 profile')
        from control_trace import ControlTrace
        recorder=ControlTrace(args.record_control,arm)
    if args.model == MODEL_PATH and isinstance(arm,JointRobot) and arm.config.get('pose_quality',{}).get('enabled'):
        full=Path(__file__).absolute().parent/'models/pose_landmarker_full.task'
        if full.is_file():
            args.model=full
    mapper = None if args.mapping else WristMapper(arm.home_target, args.scale, args.depth)
    source = None
    pose_recorder=None
    if args.record_pose:
        if not args.sample_seconds:
            parser.error('--record-pose requires --sample-seconds')
        from control_trace import PoseTrace
        pose_recorder=PoseTrace(args.record_pose)
    session = TrackingSession(enabled=args.demo, ready=args.demo)
    observation, previous_observation = None, None
    last_report = 0.0
    last_status = None
    calibration_gate=CalibrationGate(args.calibration_delay)
    sampler=SamplingSession(args.sample_seconds,args.sample_output,args.sample_kind) if args.sample_seconds else None
    report_path = Path(__file__).resolve().parent / "artifacts" / "live_status.json"
    # Camera-window keys and MuJoCo-window keys share the same pending key slot.
    key_lock = threading.Lock()
    pending_keys = []

    def on_key(keycode):
        with key_lock:
            pending_keys.append(keycode)

    try:
        if not args.demo:
            source = CameraPose(args)
        with mujoco.viewer.launch_passive(arm.model, arm.data, key_callback=on_key,
                                         show_left_ui=False, show_right_ui=False) as viewer:
            if args.mapping and arm.mode in ("shadow", "standing", "walking", "tracking"):
                # Official collision and visual meshes overlap; display only
                # the visual meshes and the separately grouped floor.
                viewer.opt.geomgroup[0] = 0
            viewer.cam.lookat[:] = arm.model.stat.center if args.mapping else [0.25, 0, 0.25]
            viewer.cam.distance = max(1.0, 1.6 * arm.model.stat.extent) if args.mapping else 1.45
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -25
            if args.mapping:
                camera_config = arm.config.get("viewer", {})
                for field in ("azimuth", "elevation", "distance"):
                    if field in camera_config:
                        setattr(viewer.cam, field, float(camera_config[field]))
                if "lookat" in camera_config:
                    viewer.cam.lookat[:] = camera_config["lookat"]
            start = previous_tick = time.monotonic()
            if args.auto_calibrate and isinstance(arm,JointRobot) and arm.mode in ('shadow','standing','walking','tracking'):
                session.request_start()
                calibration_gate.begin(start)
            simulation_budget = 0.0
            while viewer.is_running():
                tick = time.monotonic()
                elapsed = tick - start
                if args.seconds and elapsed >= args.seconds:
                    break
                key = 255
                valid = False
                stream_fresh = False
                if source is not None:
                    observation, error = source.latest()
                    if pose_recorder is not None and sampler.state=='SAMPLING':
                        pose_recorder.sample(observation)
                    if error:
                        arm.hold()
                        raise RuntimeError(error)
                    joint_targets = arm.targets_from_pose(observation) if args.mapping else None
                    has_input = joint_targets is not None if args.mapping else (
                        observation is not None and observation.sample is not None)
                    if args.mapping and arm.mode in ("shadow", "standing", "walking", "tracking") and not arm.retarget.calibrated:
                        has_input = has_input and arm.calibration_ready(observation)
                    if args.mapping and arm.balance is not None and arm.balance.fallen:
                        has_input = False
                    stream_fresh = observation is not None and tick - observation.arrived < 0.35
                    valid = stream_fresh and has_input
                    status = session.status(valid, bool(args.mapping), args.depth)
                    workflow=(calibration_gate.display(tick) if session.start_pending and isinstance(arm,JointRobot)
                              and arm.mode in ('shadow','standing','walking','tracking') else sampler.display(tick) if sampler else gait_workflow(arm))
                    key = draw_camera(source, observation, status, arm, workflow)
                elif session.enabled:
                    arm.set_target(demo_target(arm, elapsed))
                with key_lock:
                    keys = pending_keys[:]
                    pending_keys.clear()
                keys.append(key)
                quit_requested = False
                for keycode in keys:
                    if keycode in (81, 113, 27):
                        quit_requested = True
                    elif keycode in (67, 99) and source is not None:
                        session.request_start()
                        calibration_gate.begin(tick)
                        if sampler:
                            sampler.reset()
                    elif keycode in (78,110) and source is not None:
                        source.reacquire_subject()
                        arm.hold()
                        if isinstance(arm,JointRobot):
                            arm.retarget.reset()
                            if arm.footstep is not None:
                                arm.footstep.reset()
                        session=TrackingSession()
                        previous_observation=None
                        observation=None
                        valid=stream_fresh=False
                        joint_targets=None
                        if sampler:
                            sampler.reset()
                    elif keycode in (82, 114) and args.mapping and arm.mode in ("standing","walking","tracking"):
                        # Explicit user recovery resets the simulation. Ordinary
                        # calibration, pauses and input loss never move the base.
                        arm.reset()
                        session = TrackingSession()
                        previous_observation = None
                        if sampler:
                            sampler.reset()
                    elif keycode == 32:
                        session.toggle_pause(valid or args.demo)
                if quit_requested:
                    break
                if source is not None:
                    start_ready=(not args.mapping or arm.mode not in ('shadow','standing','walking','tracking')
                                 or calibration_gate.ready(observation,valid and arm.calibration_ready(observation),tick))
                    restart, can_track = session.advance(valid, tick, stream_fresh, start_ready)
                    if restart:
                        if not args.mapping or arm.mode not in ("standing","walking","tracking"):
                            arm.reset()
                        if args.mapping:
                            if arm.calibrate(observation) is False:
                                session.request_start()
                                can_track=False
                            else:
                                joint_targets = arm.targets_from_pose(observation)
                        else:
                            mapper = WristMapper(arm.home_target, args.scale, args.depth)
                            mapper.calibrate(observation.sample)
                        if can_track and source.lock_subject(observation):
                            can_track=False
                            workflow_sound()
                            if sampler:
                                sampler.start(tick)
                        previous_observation = None
                    if can_track and observation is not previous_observation:
                        if args.mapping:
                            arm.set_targets(joint_targets)
                        else:
                            target = mapper.update(observation.sample, observation.arrived)
                            if target is not None:
                                arm.set_target(target)
                        previous_observation = observation
                    elif not can_track:
                        arm.hold()
                    new_status = session.status(valid, bool(args.mapping), args.depth)
                    if new_status != last_status:
                        print("[teleop]", new_status, flush=True)
                        last_status = new_status
                    if tick - last_report >= 0.5:
                        snapshot=write_live_status(report_path, arm, session, observation, valid)
                        if sampler and sampler.update(tick,snapshot):
                            workflow_sound()
                            print('[teleop] SAMPLING FINISHED - you may move',flush=True)
                        last_report = tick
                elif not session.enabled:
                    arm.hold()
                # Fixed 2 ms physics steps, independent of pose inference and display.
                if recorder is not None:
                    recorder.sample(arm)
                simulation_budget = min(simulation_budget + tick - previous_tick, 0.05)
                previous_tick = tick
                while simulation_budget >= arm.model.opt.timestep:
                    arm.step()
                    if sampler and isinstance(arm,JointRobot) and arm.balance is not None:
                        sampler.observe_robot(tick,arm)
                    simulation_budget -= arm.model.opt.timestep
                if sampler and sampler.update(tick):
                    workflow_sound()
                    print('[teleop] SAMPLING FINISHED - you may move',flush=True)
                if args.mapping and arm.mode in ('walking','tracking'):
                    viewer.cam.lookat[:2]=arm.data.qpos[:2]
                viewer.sync()
                time.sleep(max(0, 1 / 60 - (time.monotonic() - tick)))
    finally:
        if pose_recorder is not None:
            pose_recorder.close()
        if recorder is not None:
            recorder.close()
        if source is not None:
            source.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
