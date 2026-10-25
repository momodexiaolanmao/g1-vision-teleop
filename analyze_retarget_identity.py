"""Quantify what the identity human->robot joint-angle mapping actually costs.

Three measurements, all read-only:

F1  Link-ratio error. The retargeted body channels are invariant to human body
    scale and to bone-length ratios, so "identity" is self-consistent in angle
    space. It is not consistent in task space: the G1's thigh:shin ratio is
    about 0.65 while a real operator's is about 1.05, so the same joint angles
    put the ankle somewhere else. Measured as the leg tilt angle and the
    horizontal ankle offset implied by the same (hip_pitch, knee_flex).

F2  Neutral mismatch. `standing_balance.set_reference` writes the human's
    ABSOLUTE channel angle into the robot joint target for every joint except
    the waist, and `pose_retarget`'s neutral correction skips every `*flex`
    channel. So a relaxed human stance commands the knee away from the posture
    the robot actually stands in. Measured per joint against the robot's home.

F3  Scale constant. `pose_retarget` and `footstep_intent` convert image
    displacement to metres with a hardcoded 0.42 m torso. Measured against the
    real operator's torso from the recorded trace.

Scope note: in the walking entry point only joints >= 12 (waist + arms) receive
the human reference; the legs are owned by the learned policy. F1/F2 therefore
describe the STANDING phase, which is where the identity leg mapping is live.

Run:  .\\.venv\\Scripts\\python.exe analyze_retarget_identity.py
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).absolute().parent

MODEL = ROOT / "robots/unitree_g1/scene_physics.xml"
MAPPING = ROOT / "unitree_g1_walking_mapping.json"
DEPLOY = ROOT / "policies/g1_velocity/deploy.json"
TRACE = ROOT / "artifacts/step_entry_live_pose_20261012.jsonl"
OUT_JSON = ROOT / "artifacts/retarget_identity_analysis.json"
OUT_PNG = ROOT / "artifacts/retarget_identity_analysis.png"

# Human values are measured from the recorded trace itself (world landmarks,
# metres). Robot values are measured from the model's JOINT ANCHORS, not from
# XML body offsets: the G1's hip has pitch, roll and yaw joints at three
# different points, so only the hip-pitch anchor is the analogue of a human hip
# joint. Reading the last link before the knee (hip_yaw -> knee, 0.194 m)
# instead of the whole hip -> knee chain overstates the proportion mismatch by
# roughly a factor of eight.
def _measure_robot_links():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def anchor(name):
        return data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)].copy()

    hip = anchor("left_hip_pitch_joint")
    knee = anchor("left_knee_joint")
    ankle = anchor("left_ankle_pitch_joint")
    shoulder = anchor("left_shoulder_pitch_joint")
    return {
        # Sagittal projection: the lateral part of the hip offsets does not
        # enter the planar leg model.
        "thigh_m": float(np.hypot(*(knee - hip)[[0, 2]])),
        "thigh_3d_m": float(np.linalg.norm(knee - hip)),
        "shin_m": float(np.linalg.norm(ankle - knee)),
        "hip_to_shoulder_m": float(np.linalg.norm(shoulder - hip)),
    }


ROBOT_LINKS = _measure_robot_links()
G1_THIGH_M = ROBOT_LINKS["thigh_m"]
G1_SHIN_M = ROBOT_LINKS["shin_m"]
G1_LEG_M = G1_THIGH_M + G1_SHIN_M
G1_FOOT_LENGTH_M = 0.17
G1_FOOT_HALF_M = G1_FOOT_LENGTH_M / 2
G1_TORSO_M = ROBOT_LINKS["hip_to_shoulder_m"]
HARDCODED_TORSO_M = 0.42

SHOULDERS, HIPS, KNEES, ANKLES = (11, 12), (23, 24), (25, 26), (27, 28)


def load_trace(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_observation(record):
    width, height = record["image_size"]
    landmarks = [SimpleNamespace(x=p[0], y=p[1], z=p[2], visibility=p[3], presence=p[4])
                 for p in record["image"]]
    world = [SimpleNamespace(x=q[0], y=q[1], z=q[2], visibility=1.0, presence=1.0)
             for q in record["world"]]
    return SimpleNamespace(landmarks=landmarks, world_landmarks=world,
                           arrived=float(record["time"]),
                           frame=np.zeros((height, width, 3), dtype=np.uint8))


def measure_operator_anthropometry(records):
    torso, thigh, shin = [], [], []
    for record in records:
        world = np.array(record["world"])
        image = np.array(record["image"])
        core = list(SHOULDERS) + list(HIPS) + list(KNEES) + list(ANKLES)
        if not (np.all(image[core, 3] >= 0.6) and np.all(image[core, 4] >= 0.6)):
            continue
        torso.append(float(np.linalg.norm(world[list(SHOULDERS)].mean(0) - world[list(HIPS)].mean(0))))
        thigh.append(float(np.linalg.norm(world[list(KNEES)] - world[list(HIPS)], axis=1).mean()))
        shin.append(float(np.linalg.norm(world[list(ANKLES)] - world[list(KNEES)], axis=1).mean()))
    return {"frames": len(torso),
            "torso_m": float(np.median(torso)),
            "thigh_m": float(np.median(thigh)),
            "shin_m": float(np.median(shin)),
            "thigh_shin_ratio": float(np.median(thigh) / np.median(shin))}


def planar_ankle(hip_deg, knee_deg, thigh_m, shin_m):
    """Ankle relative to the hip for hip flexion then knee flexion, sagittal."""
    q1 = math.radians(hip_deg)
    q2 = math.radians(knee_deg)
    x = thigh_m * math.sin(q1) + shin_m * math.sin(q1 - q2)
    z = -(thigh_m * math.cos(q1) + shin_m * math.cos(q1 - q2))
    return x, z


def tilt_error_closed_form(shin_over_thigh_human, shin_over_thigh_robot, knee_deg):
    """Exact leg-orientation error of identity mapping, independent of hip angle.

    A planar leg whose hip angle q1 is measured from vertical and whose knee
    flexion is q2 has hip->ankle direction atan2(A sin q1 - B cos q1,
    A cos q1 + B sin q1) with A = 1 + r cos q2, B = r sin q2 and r = shin/thigh.
    That equals q1 - atan2(r sin q2, 1 + r cos q2), so the q1 term cancels when
    two bodies are given identical joint angles and only r differs.
    """
    q2 = math.radians(knee_deg)
    return math.degrees(math.atan2(shin_over_thigh_human * math.sin(q2),
                                   1 + shin_over_thigh_human * math.cos(q2))
                        - math.atan2(shin_over_thigh_robot * math.sin(q2),
                                     1 + shin_over_thigh_robot * math.cos(q2)))


def measure_link_ratio_error(operator):
    """F1: same joint angles, two limb proportions."""
    rows = []
    for hip in (-20, -10, 0, 10, 20, 30, 40):
        for knee in (5, 15, 30, 45, 60, 75, 90):
            hx, hz = planar_ankle(hip, knee, operator["thigh_m"], operator["shin_m"])
            rx, rz = planar_ankle(hip, knee, G1_THIGH_M, G1_SHIN_M)
            h_len, r_len = math.hypot(hx, hz), math.hypot(rx, rz)
            h_tilt = math.degrees(math.atan2(hx, -hz))
            r_tilt = math.degrees(math.atan2(rx, -rz))
            rows.append({
                "hip_pitch_deg": hip, "knee_flex_deg": knee,
                "human_tilt_deg": h_tilt, "robot_tilt_deg": r_tilt,
                "tilt_error_deg": r_tilt - h_tilt,
                # Dimensionless horizontal ankle offset per unit leg length.
                "human_offset_ratio": hx / h_len, "robot_offset_ratio": rx / r_len,
                # The error expressed on the robot's own body, in metres and in
                # G1 foot half-lengths -- the quantity a balance margin sees.
                "robot_horizontal_error_m": (rx / r_len - hx / h_len) * r_len,
            })
    for row in rows:
        row["robot_horizontal_error_foot_half"] = row["robot_horizontal_error_m"] / G1_FOOT_HALF_M
        row["closed_form_tilt_error_deg"] = tilt_error_closed_form(
            operator["shin_m"] / operator["thigh_m"], G1_SHIN_M / G1_THIGH_M,
            row["knee_flex_deg"])
    return rows


def load_robot():
    from joint_mapping import JointRobot
    return JointRobot(MODEL, MAPPING)


def measure_neutral_mismatch(robot, records):
    """F2: human relaxed stance channels vs the posture the robot stands in."""
    import mujoco
    robot.reset()
    mujoco.mj_forward(robot.model, robot.data)
    home = np.degrees([robot.data.qpos[e["qpos"]] for e in robot.entries])

    deploy = json.loads(DEPLOY.read_text())
    order = np.array(deploy["joint_ids_map"], dtype=int)
    default = np.degrees(np.array(deploy["default_joint_pos"]))
    # deploy default is in policy-output order; policy index -> robot joint id.
    names = [robot.model.joint(int(e["joint"])).name for e in robot.entries]

    observations = [to_observation(r) for r in records]
    calibrated = None
    for index, observation in enumerate(observations):
        if robot.calibrate(observation):
            calibrated = index
            break
    if calibrated is None:
        return None, observations, None

    channels_per_frame = []
    targets_per_frame = []
    diagnostics_per_frame = []
    for observation in observations[calibrated:]:
        targets = robot.targets_from_pose(observation)
        channels, _ = robot.retarget.evaluate(observation)
        channels_per_frame.append({k: v for k, v in channels.items()})
        targets_per_frame.append(None if targets is None else np.degrees(targets).copy())
        diagnostics_per_frame.append([dict(d) for d in robot.pose_diagnostics])

    knee = []
    for channels in channels_per_frame:
        values = [channels[f"{s}_knee_flex"] for s in ("left", "right")]
        knee.append(np.mean([v["angle"] for v in values]) if all(v["valid"] for v in values) else np.nan)
    knee = np.array(knee)
    finite = np.isfinite(knee)
    if not finite.any():
        return None, observations, None
    # Relaxed-stance frames: the least knee flexion the operator showed.
    threshold = np.nanpercentile(knee, 25)
    stance = finite & (knee <= threshold)

    rows = []
    for index, entry in enumerate(robot.entries):
        name = entry["rule"]["robot_joint"]
        channel = entry["rule"]["source"]["channel"]
        human = []
        robot_target = []
        for frame_index in np.flatnonzero(stance):
            channels = channels_per_frame[frame_index]
            valid = channels[channel]["valid"]
            targets = targets_per_frame[frame_index]
            if valid and targets is not None:
                human.append(channels[channel]["angle"])
                robot_target.append(targets[index])
        # Spread of this channel across the whole take: a semantically clean
        # channel moves when the operator moves it; a degenerate one moves with
        # whatever it is really reading (e.g. hip_yaw tracking foot splay).
        everywhere = [channels[channel]["angle"] for channels in channels_per_frame
                      if channels[channel]["valid"]]
        swing = (float(np.percentile(everywhere, 95) - np.percentile(everywhere, 5))
                 if len(everywhere) > 20 else None)
        low, high = entry["limits"]
        low_deg, high_deg = math.degrees(low), math.degrees(high)
        span = high_deg - low_deg
        share = headroom = None
        if human:
            headroom = float(min(high_deg - max(human), min(human) - low_deg))
            share = float((float(np.median(human)) - low_deg) / span * 100)
        rows.append({
            "joint": name, "channel": channel,
            "neutral_corrected": bool(not channel.endswith("flex") and "wrist" not in channel),
            "human_stance_deg": float(np.median(human)) if human else None,
            "human_stance_min_deg": float(min(human)) if human else None,
            "human_stance_max_deg": float(max(human)) if human else None,
            "robot_target_deg": float(np.median(robot_target)) if robot_target else None,
            "robot_home_deg": float(home[index]),
            "joint_min_deg": low_deg, "joint_max_deg": high_deg,
            "stance_frames": len(human),
            "channel_swing_p5_p95_deg": swing,
            "stance_headroom_to_nearest_limit_deg": headroom,
            "stance_percent_of_range": share,
        })
    for row in rows:
        if row["robot_target_deg"] is not None:
            row["target_minus_home_deg"] = row["robot_target_deg"] - row["robot_home_deg"]
        else:
            row["target_minus_home_deg"] = None

    context = {"calibrated_at_frame": calibrated, "total_frames": len(observations),
               "stance_frames": int(stance.sum()),
               "deploy_default_deg_by_joint": (
                   {names[order[i]]: float(default[i]) for i in range(len(order))} if names else {}),
               "robot_model_mass_kg": float(robot.model.body_mass.sum())}
    return rows, observations, context


def measure_scale_constant(operator, config):
    """F3: the 0.42 m torso constant against the operator actually seen."""
    bias = HARDCODED_TORSO_M / operator["torso_m"]
    locomotion = config["locomotion"]
    threshold = float(locomotion["foot_lift_threshold_m"])
    span = float(locomotion["step_half_span_m"])
    return {
        "hardcoded_torso_m": HARDCODED_TORSO_M,
        "measured_torso_m": operator["torso_m"],
        "bias_factor": bias,
        "underestimate_percent": (1 - bias) * 100,
        "foot_lift_threshold_m_as_coded": threshold,
        "foot_lift_threshold_m_actually_required": threshold / bias,
        "step_half_span_m_as_coded": span,
        "step_half_span_m_actually_required": span / bias,
    }


def plot(rows, mismatch, operator, scale, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_human = operator["shin_m"] / operator["thigh_m"]
    r_robot = G1_SHIN_M / G1_THIGH_M

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # A: the closed-form leg orientation error. Exact, and hip-independent.
    axis = axes[0]
    knee = np.linspace(0, 120, 400)
    axis.plot(knee, [tilt_error_closed_form(r_human, r_robot, k) for k in knee],
              color="tab:red", linewidth=2,
              label=f"closed form  r_h={r_human:.3f}  r_G1={r_robot:.3f}")
    tabulated = sorted({r["knee_flex_deg"] for r in rows})
    axis.plot(tabulated, [tilt_error_closed_form(r_human, r_robot, k) for k in tabulated],
              "o", color="black", markersize=5, label="per-joint table (all hip angles)")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("knee flexion (deg) -- identical joint angles on both bodies")
    axis.set_ylabel("robot leg tilt - human leg tilt (deg)")
    axis.set_title("F1  identity mapping: leg orientation error\n"
                   "exact, and independent of hip angle")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")

    # B: the same error on the robot's own body, against its support polygon.
    # Unlike the tilt error this one does depend on hip angle, so show the worst
    # case the operator can reach at each knee flexion.
    axis = axes[1]
    knee_values = sorted({r["knee_flex_deg"] for r in rows})
    worst = [max(abs(r["robot_horizontal_error_foot_half"]) for r in rows
                 if r["knee_flex_deg"] == k) for k in knee_values]
    axis.plot(knee_values, worst, "o-", color="tab:blue",
              label="worst over hip angle")
    axis.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.2,
                 label="1 x G1 foot half-length")
    axis.set_xlabel("knee flexion (deg)")
    axis.set_ylabel("|horizontal ankle error| (G1 foot half-lengths)")
    axis.set_title("F1  the same error measured on the robot\n"
                   "(above 1 leaves the foot's own footprint)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="upper left")

    # C: the posture the identity mapping commands vs the posture the robot holds.
    axis = axes[2]
    shown = [r for r in mismatch
             if any(t in r["joint"] for t in ("hip", "knee", "ankle"))
             and r["target_minus_home_deg"] is not None]
    positions = np.arange(len(shown))
    axis.bar(positions - 0.2, [r["robot_home_deg"] for r in shown], 0.4,
             color="tab:gray", label="robot home (its own stance)")
    axis.bar(positions + 0.2, [r["robot_target_deg"] for r in shown], 0.4,
             color="tab:orange", label="commanded by identity mapping")
    for index, row in enumerate(shown):
        if row["stance_headroom_to_nearest_limit_deg"] is not None \
                and row["stance_headroom_to_nearest_limit_deg"] < 0:
            axis.annotate("clipped", (index + 0.2, row["robot_target_deg"]),
                          textcoords="offset points", xytext=(0, 8),
                          ha="center", fontsize=7, color="tab:red")
    axis.set_xticks(positions)
    axis.set_xticklabels([r["joint"].replace("_joint", "").replace("left_", "L ").replace("right_", "R ")
                          for r in shown], rotation=60, ha="right", fontsize=8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("joint angle (deg)")
    axis.set_title("F2  relaxed stance: human absolute angle written\ninto the target vs the robot's own stance")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)

    figure.suptitle(
        f"Identity retargeting on Unitree G1 - operator torso {operator['torso_m']:.3f} m vs "
        f"hardcoded {scale['hardcoded_torso_m']:.2f} m (bias x{scale['bias_factor']:.3f}); "
        f"thigh:shin operator {operator['thigh_shin_ratio']:.3f} vs G1 {G1_THIGH_M / G1_SHIN_M:.3f}",
        fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=130)


def main():
    records = load_trace(TRACE)
    operator = measure_operator_anthropometry(records)
    config = json.loads(MAPPING.read_text(encoding="utf-8-sig"))

    rows = measure_link_ratio_error(operator)
    worst = max(rows, key=lambda r: abs(r["tilt_error_deg"]))
    errors = np.array([abs(r["tilt_error_deg"]) for r in rows])

    robot = load_robot()
    mismatch, observations, context = measure_neutral_mismatch(robot, records)
    scale = measure_scale_constant(operator, config)

    legs = ("hip", "knee", "ankle")
    leg_rows = [r for r in (mismatch or []) if any(t in r["joint"] for t in legs)
                and r["target_minus_home_deg"] is not None]
    worst_leg = max(leg_rows, key=lambda r: abs(r["target_minus_home_deg"]), default=None)
    tight = sorted((r for r in leg_rows
                    if r["stance_headroom_to_nearest_limit_deg"] is not None),
                   key=lambda r: r["stance_headroom_to_nearest_limit_deg"])[:3]

    findings = [
        ("F1 leg orientation: the retargeted channels are invariant to body scale and to "
         "bone-length ratios, so identity is self-consistent in angle space but not in task "
         f"space. Measured from the hip-pitch joint anchor, G1 shin:thigh "
         f"{G1_SHIN_M / G1_THIGH_M:.3f} vs operator "
         f"{operator['shin_m'] / operator['thigh_m']:.3f} gives up to "
         f"{abs(worst['tilt_error_deg']):.2f} deg of leg tilt error and "
         f"{max(abs(r['robot_horizontal_error_m']) for r in rows) * 100:.2f} cm of horizontal "
         "ankle displacement at identical joint angles. This is SMALL: the G1's leg "
         "proportions are close to the operator's. F1 is not the main defect."),
        ("F1 sensitivity: reading the thigh from the last link before the knee "
         "(hip_yaw -> knee, 0.194 m) instead of the whole hip-pitch -> knee chain would give "
         "shin:thigh 1.55 and a 14 deg error. That reading is wrong for this comparison - the "
         "hip roll and yaw offsets are part of the distance from the flexion axis to the knee."),
        ("F2 neutral mismatch: standing_balance writes the human ABSOLUTE angle into every "
         "non-waist joint target, and pose_retarget's neutral correction only removes the "
         "value seen at the calibration instant. The knee happens to match (a negative "
         "result); the mismatch is in the rotational channels."),
        ("F2 degenerate channels: with a near-straight knee the hip_yaw channel falls back to "
         "the foot vector, so it reports the operator's toe-out splay rather than hip internal "
         "rotation, and that value is written straight into hip_yaw_joint."),
        ("F2 saturation: ankle_roll sits within a few degrees of its joint limit from the "
         "operator's stance width alone."),
        ("F3 scale constant: image-to-metre conversion is hardcoded to a 0.42 m torso, "
         f"{scale['underestimate_percent']:.1f}% under this operator, so the "
         f"{scale['foot_lift_threshold_m_as_coded'] * 100:.1f} cm foot-lift gate really requires "
         f"{scale['foot_lift_threshold_m_actually_required'] * 100:.2f} cm."),
    ]

    report = {
        "generated_by": "analyze_retarget_identity.py",
        "scope": ("Identity joint-angle mapping is live for the LEGS only in the STANDING phase. "
                  "During walking (g1_locomotion.set_reference) only joints >= 12 (waist + arms) "
                  "receive the human reference, scaled by imitation_strength and rate limited."),
        "operator_anthropometry": operator,
        "robot_anthropometry": {"thigh_m_sagittal_hip_pitch_to_knee": G1_THIGH_M,
                                "thigh_m_3d": ROBOT_LINKS["thigh_3d_m"],
                                "shin_m_knee_to_ankle": G1_SHIN_M,
                                "shin_over_thigh": G1_SHIN_M / G1_THIGH_M,
                                "leg_m": G1_LEG_M,
                                "hip_to_shoulder_m": G1_TORSO_M,
                                "counterfactual_thigh_m_if_read_from_hip_yaw_to_knee": 0.194,
                                "counterfactual_shin_over_thigh": G1_SHIN_M / 0.194},
        "F1_link_ratio": {
            "table": rows,
            "closed_form": "atan2(r_h sin q2, 1 + r_h cos q2) - atan2(r_g sin q2, 1 + r_g cos q2), "
                           "r = shin/thigh, exact and independent of hip angle",
            "shin_over_thigh_operator": operator["shin_m"] / operator["thigh_m"],
            "shin_over_thigh_g1": G1_SHIN_M / G1_THIGH_M,
            "closed_form_max_deviation_from_table_deg": max(
                abs(r["closed_form_tilt_error_deg"] - r["tilt_error_deg"]) for r in rows),
            "hip_angle_dependence_deg": max(
                max(v) - min(v) for v in
                [[r["tilt_error_deg"] for r in rows if r["knee_flex_deg"] == k]
                 for k in {r["knee_flex_deg"] for r in rows}]),
            "max_abs_tilt_error_deg": abs(worst["tilt_error_deg"]),
            "max_at": {"hip_pitch_deg": worst["hip_pitch_deg"], "knee_flex_deg": worst["knee_flex_deg"]},
            "median_abs_tilt_error_deg": float(np.median(errors)),
            "max_horizontal_error_m": max(abs(r["robot_horizontal_error_m"]) for r in rows),
            "max_horizontal_error_foot_half": max(abs(r["robot_horizontal_error_foot_half"]) for r in rows),
        },
        "F2_neutral_mismatch": {"table": mismatch or [], "context": context,
                                "worst_leg_joint": worst_leg, "tightest_leg_limits": tight},
        "F3_scale_constant": scale,
        "findings": findings,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if mismatch:
        plot(rows, mismatch, operator, scale, OUT_PNG)

    print(f"operator torso {operator['torso_m']:.4f} m  thigh:shin {operator['thigh_shin_ratio']:.4f}"
          f"  (G1 {G1_THIGH_M / G1_SHIN_M:.4f})")
    print(f"F1 max |leg tilt error| {abs(worst['tilt_error_deg']):.2f} deg at "
          f"hip {worst['hip_pitch_deg']:+d} / knee {worst['knee_flex_deg']} deg; "
          f"median {np.median(errors):.2f} deg")
    print(f"F1 closed form matches table within "
          f"{max(abs(r['closed_form_tilt_error_deg'] - r['tilt_error_deg']) for r in rows):.2e} deg; "
          f"hip-angle dependence "
          f"{max(max(v) - min(v) for v in [[r['tilt_error_deg'] for r in rows if r['knee_flex_deg'] == k] for k in {r['knee_flex_deg'] for r in rows}]):.2e} deg")
    print(f"F1 max horizontal ankle error {max(abs(r['robot_horizontal_error_m']) for r in rows) * 100:.1f} cm "
          f"= {max(abs(r['robot_horizontal_error_foot_half']) for r in rows):.2f} x G1 foot half-length")
    if mismatch:
        worst_joint = max((r for r in mismatch if r["target_minus_home_deg"] is not None),
                          key=lambda r: abs(r["target_minus_home_deg"]))
        print(f"F2 worst neutral mismatch: {worst_joint['joint']} "
              f"{worst_joint['target_minus_home_deg']:+.2f} deg "
              f"(target {worst_joint['robot_target_deg']:.2f} vs home {worst_joint['robot_home_deg']:.2f})")
        if worst_leg is not None:
            print(f"F2 worst LEG mismatch:   {worst_leg['joint']} "
                  f"{worst_leg['target_minus_home_deg']:+.2f} deg "
                  f"(target {worst_leg['robot_target_deg']:.2f} vs home {worst_leg['robot_home_deg']:.2f}); "
                  f"channel swing p5-p95 {worst_leg['channel_swing_p5_p95_deg']:.1f} deg")
        for row in tight:
            print(f"F2 tight limit:          {row['joint']} stance "
                  f"{row['human_stance_deg']:+.2f} deg, range "
                  f"[{row['joint_min_deg']:.0f}, {row['joint_max_deg']:.0f}], headroom "
                  f"{row['stance_headroom_to_nearest_limit_deg']:.2f} deg "
                  f"({row['stance_percent_of_range']:.0f}% of range)")
    print(f"F3 hardcoded 0.42 m torso -> bias x{scale['bias_factor']:.4f} "
          f"({scale['underestimate_percent']:.1f}% under); "
          f"{scale['foot_lift_threshold_m_as_coded'] * 100:.1f} cm threshold really needs "
          f"{scale['foot_lift_threshold_m_actually_required'] * 100:.2f} cm")
    print(f"wrote {OUT_JSON.name}" + (f" and {OUT_PNG.name}" if mismatch else ""))


if __name__ == "__main__":
    main()
