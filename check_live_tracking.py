"""Sample the local control snapshot to verify actual camera-to-joint movement."""
import argparse
import json
from pathlib import Path
import time


def build_report(samples, seconds):
    """Build numeric statistics from live-status snapshots; no sampling or writes."""
    joints = {}
    for value in samples:
        if not value.get("enabled") or not value.get("input_valid"):
            continue
        for index, rule in enumerate(value.get("rules", [])):
            if not rule["valid"]:
                continue
            rows = joints.setdefault(rule["joint"], [])
            rows.append({"human": rule["human_angle_deg"], "target": rule["target_deg"],
                         "command": value["command_deg"][index], "actual": value["actual_deg"][index]})
    summary = {}
    for name, rows in joints.items():
        span = lambda field: max(row[field] for row in rows) - min(row[field] for row in rows)
        summary[name] = {
            "active_samples": len(rows), "human_motion_deg": span("human"),
            "target_motion_deg": span("target"), "actual_motion_deg": span("actual"),
            "max_command_tracking_error_deg": max(abs(row["command"] - row["actual"]) for row in rows),
            "motion_observed": len(rows) >= 3 and span("human") > 3
                               and span("target") > 2 and span("actual") > 3,
        }
        for field in ('human','command','actual'):
            changes=sorted(abs(b[field]-a[field]) for a,b in zip(rows,rows[1:]))
            summary[name][f'p95_sample_change_{field}_deg']=changes[min(len(changes)-1,int(.95*len(changes)))] if changes else 0.
    report = {"observed_at_unix": time.time(), "seconds": seconds,
              "fresh_samples": len(samples), "joints": summary,
              "robot_model": samples[-1].get("robot_model") if samples else None,
              "last_state": samples[-1]["state"] if samples else "no fresh live status"}
    if samples:
        report['snapshot_window']={'first_updated_at_unix':samples[0].get('updated_at_unix'),
                                   'last_updated_at_unix':samples[-1].get('updated_at_unix'),
                                   'note':'Sparse status snapshots can miss brief foot clearance; use continuous_physics when available.'}
    roots = [v["root_tracking"] for v in samples if v.get("enabled") and v.get("input_valid")
             and v.get("root_tracking", {}).get("valid")]
    if roots:
        report["root_tracking"] = {
            "active_samples": len(roots),
            "position_span_m": [max(r["position_m"][i] for r in roots) - min(r["position_m"][i] for r in roots)
                                for i in range(3)],
            "turn_span_deg": max(r["yaw_deg"] for r in roots) - min(r["yaw_deg"] for r in roots),
            "depth_is_estimated": roots[-1]["depth_is_estimated"],
            "mode": "kinematic preview",
        }
    if samples:
        balance = [v["balance"] for v in samples if "balance" in v]
        if balance:
            report["physical_balance"] = {
                "samples": len(balance), "any_fall": any(b["fallen"] for b in balance),
                "min_pelvis_height_m": min(b["height_m"] for b in balance),
                "max_pelvis_tilt_deg": max(b["tilt_deg"] for b in balance),
                "min_total_foot_load_n": min(sum(b["foot_loads_n"]) for b in balance),
                "max_total_foot_load_n": max(sum(b["foot_loads_n"]) for b in balance),
                "max_other_floor_contacts": max(b["other_floor_contacts"] for b in balance),
                "max_external_force": max(b["external_force_max"] for b in balance),
                "max_applied_body_force": max(b["body_force_max"] for b in balance),
                "solver": balance[-1]["solver_backend"], "qp_failures": balance[-1]["qp_failures"],
                "max_torque_limit_ratio": balance[-1]["max_torque_limit_ratio"],
            }
            report["control_mode"] = samples[-1].get("control_mode")
            stepping=[v for v in samples if 'foot_clearance_m' in v.get('balance',{})]
            if stepping:
                report['physical_steps']={
                    'human_step_active_samples':sum(v.get('footstep_intent',{}).get('active',False) for v in stepping)
                        if any('footstep_intent' in v for v in stepping) else None,
                    'feet_hidden_samples':sum(v.get('footstep_intent',{}).get('state','').startswith('FEET HIDDEN') for v in stepping)
                        if any('footstep_intent' in v for v in stepping) else None,
                    'foot_clearance_max_m':[max(v['balance']['foot_clearance_m'][i] for v in stepping) for i in (0,1)],
                    'foot_unloaded_samples':[sum(v['balance']['foot_loads_n'][i]<5 for v in stepping) for i in (0,1)],
                    'note':'Physical foot clearance and force, not joint-angle motion or recognition accuracy.'}
            report["reference_error_note"] = "Legs prioritize support; actual posture differs from human angle references. Waist references are bounded; reference error is not motion-capture accuracy."
        report["configured_joints"] = len(samples[-1].get("rules", []))
        report['camera_subject']={'last':samples[-1].get('camera_subject',{}),
            'max_detected_people':max(v.get('camera_subject',{}).get('detected_people',0) for v in samples),
            'selected_person_lost_samples':sum('LOST' in v.get('camera_subject',{}).get('state','') for v in samples)}
        report["observed_joints"] = len(summary)
        report["moving_joints"] = sum(item["motion_observed"] for item in summary.values())
        report["max_simultaneously_valid_joints"] = max(sum(r["valid"] for r in value.get("rules", []))
                                                       for value in samples if value.get("enabled")) if any(
                                                       value.get("enabled") for value in samples) else 0
        report["frame_age_ms"] = {"min": min(v["frame_age_ms"] for v in samples if v.get("frame_age_ms") is not None),
                                  "max": max(v["frame_age_ms"] for v in samples if v.get("frame_age_ms") is not None)} if any(
                                  v.get("frame_age_ms") is not None for v in samples) else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output", type=Path, help="save this run to a separate JSON report")
    parser.add_argument('--quiet',action='store_true',help='print physical summary only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = root / "artifacts" / "live_status.json"
    samples = []
    deadline = time.monotonic() + args.seconds
    previous = None
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["updated_at_unix"] != previous and time.time() - value["updated_at_unix"] < 2:
                samples.append(value)
                previous = value["updated_at_unix"]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    report = build_report(samples, args.seconds)
    output = args.output or root / "artifacts" / "live_tracking_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='joints'} if args.quiet else report, indent=2))


if __name__ == "__main__":
    main()
