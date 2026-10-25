"""The hip_yaw channel is two different quantities behind one name.

pose_retarget.py picks the hip yaw estimate by a knee-flexion threshold:

    abs(sin(flex)) > 0.35   ->  bend plane: which way the shin bends off the thigh
    else, if heel+toe seen  ->  the foot vector: which way the foot points

With pose_quality enabled the threshold is sin 0.35, i.e. flex > 20.49 deg.

Those are not the same physical quantity. A person standing with no hip internal
rotation but their toes turned out (habitual stance, tibial torsion) has
bend-plane yaw near zero and foot yaw near the splay angle. So the channel
switches value as the knee crosses the threshold, and the retargeting layer
treats the result as one scalar and writes it straight into hip_yaw_joint.

This measures the switch, its size, and what commanding the result does to the
robot while standing.

Run:  .\\.venv\\Scripts\\python.exe analyze_hip_yaw_channel.py
"""
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pose_retarget import BodyRetarget, IDS, rotate
from validate_unitree_g1 import A

ROOT = Path(__file__).absolute().parent
TRACE = ROOT / "artifacts/step_entry_live_pose_20261012.jsonl"
MAPPING = ROOT / "unitree_g1_physics_mapping.json"
OUT_JSON = ROOT / "artifacts/hip_yaw_channel_analysis.json"
OUT_PNG = ROOT / "artifacts/hip_yaw_channel_analysis.png"

BEND_THRESHOLD_SIN = 0.35          # pose_retarget.py, with pose_quality enabled
BEND_THRESHOLD_DEG = math.degrees(math.asin(BEND_THRESHOLD_SIN))
CLOCK = [1000.0]


def rig(knee_flex=0.0, hip_yaw=0.0, foot_splay=0.0, hip_pitch=0.0, hip_roll=0.0,
        ankle_pitch=0.0, ankle_roll=0.0, waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0,
        stamp=None):
    """validate_unitree_g1.pose() with an explicit foot-splay axis.

    Splay rotates the foot about the vertical relative to the shank, which the
    shipped rig cannot express -- without it a splayed foot is indistinguishable
    from a rotated thigh, and there is nothing to disagree about.
    """
    values = locals()
    v = lambda key: float(values.get(key, 0.0) or 0.0)
    waist = rotate("z", v("waist_yaw")) @ rotate("x", v("waist_roll")) @ rotate("y", v("waist_pitch"))
    positions = {}
    for sign, side in ((1, "left"), (-1, "right")):
        hip = np.array([0., sign * 0.12, 0.])
        shoulder = waist @ np.array([0., sign * 0.21, 0.52])
        elbow = shoulder + np.array([0., 0., -0.27])
        wrist = elbow + np.array([0., 0., -0.24])
        hand = np.eye(3)
        leg = (rotate("y", v("hip_pitch")) @ rotate("x", sign * v("hip_roll"))
               @ rotate("z", sign * v("hip_yaw")))
        knee = hip + leg @ np.array([0., 0., -0.37])
        shank = leg @ rotate("y", v("knee_flex"))
        ankle = knee + shank @ np.array([0., 0., -0.36])
        # Splay turns the LEFT foot towards +y (anatomical left) and mirrors.
        foot = (shank @ rotate("y", v("ankle_pitch")) @ rotate("x", sign * v("ankle_roll"))
                @ rotate("z", sign * v("foot_splay")))
        for name, point in (("hip", hip), ("shoulder", shoulder), ("elbow", elbow), ("wrist", wrist),
                            ("index", wrist + hand @ np.array([0.11, 0.025, 0.])),
                            ("pinky", wrist + hand @ np.array([0.11, -0.025, 0.])),
                            ("thumb", wrist + hand @ np.array([0.07, 0.04, 0.015])),
                            ("knee", knee), ("ankle", ankle),
                            ("heel", ankle + foot @ np.array([-0.04, 0., -0.06])),
                            ("foot_index", ankle + foot @ np.array([0.15, 0., -0.06]))):
            positions[f"{side}_{name}"] = point
    for name in IDS:
        if name not in positions:
            positions[name] = waist @ np.array([0.06, 0.02 if "left" in name else -0.02, 0.72])
    world, image = [], []
    for name in IDS:
        point = A @ positions[name]
        world.append(SimpleNamespace(x=point[0], y=point[1], z=point[2], visibility=1., presence=1.))
        image.append(SimpleNamespace(x=0.5 + point[0] / 2.2 / (1280 / 720),
                                     y=0.5 + point[1] / 2.2, z=point[2] / 2.2,
                                     visibility=1., presence=1.))
    if stamp is None:
        CLOCK[0] += 1.0 / 30.0
        stamp = CLOCK[0]
    return SimpleNamespace(landmarks=image, world_landmarks=world, arrived=stamp,
                           frame=np.zeros((720, 1280, 3), dtype=np.uint8))


def mapper():
    config = json.loads(MAPPING.read_text(encoding="utf-8-sig"))
    return BodyRetarget(float(config["confidence"]), config["pose_quality"])


def settle(brain, **target):
    """Displace clearly, then hold the target.

    A slow ramp does NOT work here: the quality filter only accepts a new value
    after the pose moves more than min_image_motion_px from the last accepted
    reference, and a gradual synthetic ramp stays under it, freezing the channel
    at whatever value it had when the ramp began.
    """
    for _ in range(5):
        brain.evaluate(rig(knee_flex=math.radians(60)))
    for _ in range(30):
        brain.evaluate(rig(**target))
    channels, _ = brain.evaluate(rig(**target))
    return channels


def sweep_splay():
    """A still-standing operator, no hip rotation, only their toes turned out."""
    rows = []
    for splay_deg in (0, 10, 20, 30, 40):
        for knee_deg in (0, 5, 10, 15, 18, 20, 22, 25, 30, 40):
            brain = mapper()
            brain.calibrate(rig())
            channels = settle(brain, knee_flex=math.radians(knee_deg),
                              foot_splay=math.radians(splay_deg))
            yaw = channels["left_hip_yaw"]
            knee = channels["left_knee_flex"]
            rows.append({
                "foot_splay_deg": splay_deg, "knee_flex_requested_deg": knee_deg,
                "knee_flex_reported_deg": knee["angle"] if knee["valid"] else None,
                "hip_yaw_deg": yaw["angle"] if yaw["valid"] else None,
                "hip_yaw_valid": yaw["valid"],
                "bend_plane_regime": (knee["valid"] and abs(math.sin(math.radians(knee["angle"])))
                                      > BEND_THRESHOLD_SIN),
            })
    return rows


def sweep_axial():
    """No splay: the only rotation is the thigh itself, both estimators agree."""
    rows = []
    for axial_deg in (0, 10, 20, 30):
        brain = mapper()
        brain.calibrate(rig())
        channels = settle(brain, knee_flex=math.radians(30),
                          hip_yaw=math.radians(axial_deg))
        yaw = channels["left_hip_yaw"]
        rows.append({"hip_yaw_requested_deg": axial_deg,
                     "hip_yaw_deg": yaw["angle"] if yaw["valid"] else None})
    return rows


def real_data():
    """The same switch in the recorded take, and the splay it was reading."""
    records = [json.loads(line) for line in open(TRACE, encoding="utf-8") if line.strip()]
    observations = []
    for record in records:
        width, height = record["image_size"]
        observations.append(SimpleNamespace(
            landmarks=[SimpleNamespace(x=p[0], y=p[1], z=p[2], visibility=p[3], presence=p[4])
                       for p in record["image"]],
            world_landmarks=[SimpleNamespace(x=q[0], y=q[1], z=q[2], visibility=1., presence=1.)
                             for q in record["world"]],
            arrived=float(record["time"]),
            frame=np.zeros((height, width, 3), dtype=np.uint8)))
    config = json.loads(MAPPING.read_text(encoding="utf-8-sig"))
    brain = BodyRetarget(float(config["confidence"]), config["pose_quality"])
    if not brain.calibrate(observations[0]):
        return None
    rows = []
    for index, observation in enumerate(observations):
        channels, _ = brain.evaluate(observation)
        yaw, knee = channels["left_hip_yaw"], channels["left_knee_flex"]
        if not (yaw["valid"] and knee["valid"]):
            continue
        # The foot direction in the pelvis frame, the quantity the fallback reads.
        brain._load(observation)
        foot_deg = None
        if brain._has("left_heel", "left_foot_index"):
            pelvis, _ = brain._frames()
            if pelvis is not None:
                foot = pelvis.T @ (brain.points["left_foot_index"] - brain.points["left_heel"])
                foot_deg = math.degrees(math.atan2(foot[1], foot[0]))
        rows.append({
            "frame": index, "time_s": observation.arrived - observations[0].arrived,
            "knee_flex_deg": knee["angle"], "hip_yaw_deg": yaw["angle"],
            "bend_plane_regime": abs(math.sin(math.radians(knee["angle"]))) > BEND_THRESHOLD_SIN,
            "foot_direction_in_pelvis_deg": foot_deg,
        })
    return rows


def simulate():
    """What commanding that phantom hip yaw does to the robot while standing.

    Foot placement is a hard equality constraint in the standing QP, so a hip
    yaw command cannot move the foot. Measure where it goes instead.
    """
    import mujoco
    from joint_mapping import JointRobot

    robot = JointRobot(ROOT / "robots/unitree_g1/scene_physics.xml", MAPPING)
    model, data = robot.model, robot.data
    home = np.array([data.qpos[e["qpos"]] for e in robot.entries])
    names = [e["rule"]["robot_joint"] for e in robot.entries]
    index = {name: i for i, name in enumerate(names)}

    def foot_yaw(side):
        body = model.body(f"{side}_ankle_roll_link").id
        rotation = data.xmat[body].reshape(3, 3)
        forward = rotation[:, 0]
        return math.degrees(math.atan2(forward[1], forward[0]))

    rows = []
    for yaw_deg in (0.0, 10.0, 20.0, 29.5, 40.0):
        reference = home.copy()
        for side in ("left", "right"):
            reference[index[f"{side}_hip_yaw_joint"]] = math.radians(yaw_deg)
        robot.reset()
        mujoco.mj_forward(model, data)
        before_foot = foot_yaw("left")
        for _ in range(int(4.0 / 0.002)):
            robot.balance.set_reference(reference, robot.entries)
            robot.balance.step()
        state = robot.balance.status()
        achieved = {name: math.degrees(data.qpos[e["qpos"]])
                    for name, e in ((e["rule"]["robot_joint"], e) for e in robot.entries)
                    if name.split("_", 1)[1].startswith(("hip_yaw", "hip_roll", "knee", "ankle_roll"))}
        rows.append({
            "hip_yaw_command_deg": yaw_deg,
            "achieved_left_hip_yaw_deg": achieved["left_hip_yaw_joint"],
            "left_foot_world_yaw_before_deg": before_foot,
            "left_foot_world_yaw_after_deg": foot_yaw("left"),
            "foot_turned_deg": foot_yaw("left") - before_foot,
            "compensation_left_hip_roll_deg": achieved["left_hip_roll_joint"],
            "compensation_left_knee_deg": achieved["left_knee_joint"],
            "compensation_left_ankle_roll_deg": achieved["left_ankle_roll_joint"],
            "max_torque_limit_ratio": state.get("max_torque_limit_ratio"),
            "fallen": state.get("fallen"),
        })
    return rows


def main():
    print(f"switch threshold: knee flexion {BEND_THRESHOLD_DEG:.2f} deg "
          f"(sin = {BEND_THRESHOLD_SIN})")

    splay_rows = sweep_splay()
    print()
    print(f"{'splay':>6s} {'knee':>6s} | {'kneeRep':>8s} {'hipYaw':>8s} {'regime':>10s}")
    for row in splay_rows:
        if row["knee_flex_requested_deg"] % 10 and row["knee_flex_requested_deg"] not in (18, 22):
            continue
        print(f"{row['foot_splay_deg']:6.0f} {row['knee_flex_requested_deg']:6.0f} | "
              f"{(row['knee_flex_reported_deg'] or float('nan')):8.2f} "
              f"{(row['hip_yaw_deg'] if row['hip_yaw_deg'] is not None else float('nan')):8.2f} "
              f"{'bend plane' if row['bend_plane_regime'] else 'foot vector':>10s}")

    # Size of the discontinuity, per splay, across the switch.
    jumps = []
    for splay in sorted({r["foot_splay_deg"] for r in splay_rows}):
        below = [r for r in splay_rows if r["foot_splay_deg"] == splay
                 and not r["bend_plane_regime"] and r["hip_yaw_deg"] is not None]
        above = [r for r in splay_rows if r["foot_splay_deg"] == splay
                 and r["bend_plane_regime"] and r["hip_yaw_deg"] is not None]
        if below and above:
            low = float(np.median([r["hip_yaw_deg"] for r in below]))
            high = float(np.median([r["hip_yaw_deg"] for r in above]))
            jumps.append({"foot_splay_deg": splay, "foot_vector_regime_deg": low,
                          "bend_plane_regime_deg": high, "jump_deg": high - low})
    for jump in jumps:
        print(f"splay {jump['foot_splay_deg']:3.0f} deg: foot-vector regime "
              f"{jump['foot_vector_regime_deg']:+7.2f} deg, bend-plane "
              f"{jump['bend_plane_regime_deg']:+7.2f} deg -> jump {jump['jump_deg']:+7.2f} deg")

    axial_rows = sweep_axial()
    print()
    print("no splay, thigh rotated instead (both estimators should agree):")
    for row in axial_rows:
        print(f"  requested {row['hip_yaw_requested_deg']:5.0f} deg -> "
              f"channel {row['hip_yaw_deg']:7.2f} deg")

    real = real_data()
    if real:
        low = [r for r in real if not r["bend_plane_regime"]]
        high = [r for r in real if r["bend_plane_regime"]]
        print()
        print(f"recorded take: {len(real)} frames, {len(low)} in the foot-vector regime, "
              f"{len(high)} in the bend-plane regime")
        for label, group in (("foot vector", low), ("bend plane", high)):
            if group:
                yaws = np.array([r["hip_yaw_deg"] for r in group])
                feet = np.array([r["foot_direction_in_pelvis_deg"] for r in group
                                 if r["foot_direction_in_pelvis_deg"] is not None])
                print(f"  {label:12s} hip_yaw median {np.median(yaws):+7.2f} deg "
                      f"(p5 {np.percentile(yaws, 5):+7.2f}, p95 {np.percentile(yaws, 95):+7.2f})"
                      + (f" | foot direction median {np.median(feet):+7.2f} deg" if len(feet) else ""))

    sim = simulate()
    print()
    print("what the robot does with that hip_yaw command while standing:")
    print(f"  {'cmd':>6s} {'achvYaw':>8s} {'footTurn':>9s} {'hipRoll':>8s} {'knee':>7s} {'ankRoll':>8s} {'torq':>6s}")
    for row in sim:
        print(f"  {row['hip_yaw_command_deg']:6.1f} {row['achieved_left_hip_yaw_deg']:8.2f} "
              f"{row['foot_turned_deg']:9.2f} {row['compensation_left_hip_roll_deg']:8.2f} "
              f"{row['compensation_left_knee_deg']:7.2f} "
              f"{row['compensation_left_ankle_roll_deg']:8.2f} "
              f"{row['max_torque_limit_ratio']:6.3f}")

    report = {
        "generated_by": "analyze_hip_yaw_channel.py",
        "simulated_standing": sim,
        "switch_threshold_knee_flex_deg": BEND_THRESHOLD_DEG,
        "switch_threshold_sin": BEND_THRESHOLD_SIN,
        "mechanism": "pose_retarget.py:359-370 -- bend-plane yaw when |sin(flex)| > 0.35, "
                     "otherwise the foot vector in the pelvis frame",
        "splay_sweep": splay_rows,
        "discontinuity": jumps,
        "axial_sweep": axial_rows,
        "recorded_take": {
            "frames": len(real) if real else 0,
            "summary": [{"regime": label, "frames": len(group),
                         "hip_yaw_median_deg": float(np.median([r["hip_yaw_deg"] for r in group])),
                         "hip_yaw_p5_deg": float(np.percentile([r["hip_yaw_deg"] for r in group], 5)),
                         "hip_yaw_p95_deg": float(np.percentile([r["hip_yaw_deg"] for r in group], 95))}
                        for label, group in (("foot_vector", low), ("bend_plane", high))]
            if real else [],
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        plot(splay_rows, axial_rows, real, OUT_PNG)
        print(f"\nwrote {OUT_JSON.name} and {OUT_PNG.name}")
    except Exception as error:  # noqa: BLE001
        print(f"\nwrote {OUT_JSON.name} (plot skipped: {error})")


def plot(splay_rows, axial_rows, real, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    axis = axes[0]
    for splay in (0, 10, 20, 30, 40):
        rows = [r for r in splay_rows if r["foot_splay_deg"] == splay and r["hip_yaw_deg"] is not None]
        axis.plot([r["knee_flex_reported_deg"] for r in rows],
                  [r["hip_yaw_deg"] for r in rows], "o-", label=f"toe-out {splay} deg")
    axis.axvline(BEND_THRESHOLD_DEG, color="black", linestyle="--", linewidth=1.3,
                 label=f"switch at {BEND_THRESHOLD_DEG:.1f} deg knee flexion")
    axis.set_xlabel("reported knee flexion (deg)")
    axis.set_ylabel("hip_yaw channel (deg)")
    axis.set_title("Standing still, no hip rotation,\nonly the toes turned out")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[1]
    axis.plot([r["hip_yaw_requested_deg"] for r in axial_rows],
              [r["hip_yaw_deg"] for r in axial_rows], "s-", color="tab:green")
    axis.plot([0, 30], [0, 30], "k--", linewidth=1, label="identity")
    axis.set_xlabel("thigh axial rotation actually applied (deg)")
    axis.set_ylabel("hip_yaw channel (deg)")
    axis.set_title("No splay, knee at 30 deg:\nthe channel tracks the thigh")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[2]
    if real:
        for label, group, color in (("foot vector", [r for r in real if not r["bend_plane_regime"]], "tab:orange"),
                                    ("bend plane", [r for r in real if r["bend_plane_regime"]], "tab:blue")):
            if group:
                axis.scatter([r["knee_flex_deg"] for r in group],
                             [r["hip_yaw_deg"] for r in group],
                             s=14, alpha=0.6, color=color, label=label)
        axis.axvline(BEND_THRESHOLD_DEG, color="black", linestyle="--", linewidth=1.3)
    axis.set_xlabel("knee flexion (deg)")
    axis.set_ylabel("hip_yaw channel (deg)")
    axis.set_title("Recorded take\n(same switch, on real camera poses)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
