"""Compare models and retarget filters on the same local camera frames.

Saves numeric landmarks only. The operator must stand still for a valid
stationary-jitter interpretation; this is not a ground-truth accuracy score.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np

from pose_retarget import BodyRetarget
from teleop import CameraPose, Observation

ROOT = Path(__file__).absolute().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=18)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    options = SimpleNamespace(camera=args.camera, backend='auto', width=1280, height=720,
                              hand='right', depth=False, model=ROOT/'models/pose_landmarker_lite.task')
    source = CameraPose(options)
    full_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/pose_landmarker_full.task')),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=.6, min_pose_presence_confidence=.6,
        min_tracking_confidence=.6)
    mappers = {name: BodyRetarget(quality_config={'enabled': name.endswith('filtered'),
              'neutral_correction': name.endswith('filtered')})
              for name in ('lite_raw', 'lite_filtered', 'full_raw', 'full_filtered')}
    history = {name: [] for name in mappers}
    traces, times, previous = [], [], None
    started = time.monotonic()
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(full_options) as full:
            while time.monotonic() - started < args.seconds:
                observation, error = source.latest()
                if error:
                    raise RuntimeError(error)
                if observation is None or observation is previous:
                    time.sleep(.005)
                    continue
                previous = observation
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(observation.frame, cv2.COLOR_BGR2RGB)))
                before = time.perf_counter()
                result = full.detect_for_video(image, int(observation.arrived * 1000))
                elapsed = 1000*(time.perf_counter()-before)
                full_obs = Observation(observation.arrived, time.monotonic(), None, observation.frame,
                                      result.pose_landmarks[0] if result.pose_landmarks else [],
                                      result.pose_world_landmarks[0] if result.pose_world_landmarks else [])
                for prefix, obs in [('lite', observation), ('full', full_obs)]:
                    for suffix in ('raw', 'filtered'):
                        mapper = mappers[prefix+'_'+suffix]
                        if not mapper.calibrated and mapper.calibration_ready(obs):
                            mapper.calibrate(obs)
                        channels, _ = mapper.evaluate(obs)
                        history[prefix+'_'+suffix].append({
                            'time': obs.arrived-started,
                            'angles': {n: c['angle'] for n,c in channels.items() if c['valid']}})
                    traces.append({'model': prefix, 'time': obs.arrived-started,
                        'image_size': [1280,720],
                        'image': [[p.x,p.y,p.z,p.visibility or 0,p.presence or 0] for p in obs.landmarks],
                        'world': [[p.x,p.y,p.z] for p in obs.world_landmarks]})
                times.append(elapsed)
                display = cv2.flip(observation.frame, 1)
                cv2.putText(display, 'STABILITY CHECK: stand still, arms relaxed, both feet visible',
                            (16,30), cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
                cv2.imshow('DJI pose stability check', display)
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
    finally:
        source.close()
    summary = {}
    for name, records in history.items():
        names = sorted({n for r in records for n in r['angles']})
        rows = {}
        for channel in names:
            values = [(r['time'],r['angles'][channel]) for r in records if r['time']>2 and channel in r['angles']]
            if len(values)<5:
                continue
            data = np.array(values)
            rows[channel]={'samples':len(values),'std_deg':float(np.std(data[:,1])),
                          'span_deg':float(np.ptp(data[:,1])),
                          'p95_frame_change_deg':float(np.percentile(abs(np.diff(data[:,1])),95))}
        summary[name]=rows
    report={'duration_s': time.monotonic()-started, 'frames':len(times),
            'full_inference_median_ms':float(np.median(times)) if times else None,
            'note':'Same-frame camera comparison. Stationary jitter only if operator stood still; not accuracy.',
            'summary':summary,'angle_traces':history}
    (ROOT/'artifacts/unitree_g1_pose_stability.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'artifacts/pose_landmark_trace.jsonl').write_text('\n'.join(json.dumps(t) for t in traces),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='angle_traces'},indent=2))


if __name__=='__main__':
    main()
