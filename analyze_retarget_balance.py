"""Does the identity mapping's geometric error reach the balance controller?

analyze_retarget_identity.py measured the error in geometry. This measures what
survives into the real physics simulation, and what a plausible fix would buy.

Three mappings are driven from the SAME operator leg angles, through the SAME
StandingBalance controller, from the SAME initial state:

  I  identity                robot knee = operator knee, exactly what ships.
  T  orientation-preserving  hip unchanged (so the thigh direction is exact),
                             knee re-solved from the closed form so the
                             hip->ankle direction matches the operator's.
  H  height-preserving       hip unchanged, knee re-solved so the operator's
                             normalised crouch depth is reproduced.

Only the knee differs between them, so everything measured is attributable to
the knee angle alone.

The quantity that matters is not the joint angle but the posture the operator's
configuration implies, normalised by leg length. A leg with shin/thigh ratio r
at hip angle q1 and knee flexion q2 holds its hip at
    (cos q1 + r cos(q1 - q2)) / (1 + r)
of a leg length above the ankle. That expression depends on r except on the
locus q2 = 2 q1, so identical joint angles generally mean different postures.

Run:  .\\.venv\\Scripts\\python.exe analyze_retarget_balance.py
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from analyze_retarget_identity import (G1_LEG_M, G1_SHIN_M, G1_THIGH_M,
                                       tilt_error_closed_form)
from joint_mapping import JointRobot
from physics_benchmark import measure

ROOT = Path(__file__).absolute().parent
MODEL = ROOT / "robots/unitree_g1/scene_physics.xml"
MAPPING = ROOT / "unitree_g1_physics_mapping.json"
TRACE = ROOT / "artifacts/step_entry_live_pose_20261012.jsonl"
OUT_JSON = ROOT / "artifacts/retarget_balance_analysis.json"
OUT_PNG = ROOT / "artifacts/retarget_balance_analysis.png"

SIM_DT = 0.002


def operator_ratio():
    with open(TRACE, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    thigh, shin = [], []
    for record in records:
        world = np.array(record["world"])
        image = np.array(record["image"])
        core = [23, 24, 25, 26, 27, 28]
        if not (np.all(image[core, 3] >= 0.6) and np.all(image[core, 4] >= 0.6)):
            continue
        thigh.append(float(np.linalg.norm(world[[25, 26]] - world[[23, 24]], axis=1).mean()))
        shin.append(float(np.linalg.norm(world[[27, 28]] - world[[25, 26]], axis=1).mean()))
    return float(np.median(shin) / np.median(thigh)), float(np.median(thigh)), float(np.median(shin))


def normalised_hip_height(r, hip_deg, knee_deg):
    q1, q2 = math.radians(hip_deg), math.radians(knee_deg)
    return (math.cos(q1) + r * math.cos(q1 - q2)) / (1 + r)


def knee_for_same_tilt(r_operator, r_robot, knee_deg):
    """Robot knee whose hip->ankle DIRECTION matches the operator's."""
    q2 = math.radians(knee_deg)
    target = math.atan2(r_operator * math.sin(q2), 1 + r_operator * math.cos(q2))
    sin_term = math.sin(target) / r_robot
    return knee_deg if abs(sin_term) > 1 else math.degrees(target + math.asin(sin_term))


def knee_for_same_height(r_operator, r_robot, hip_deg, knee_deg):
    """Robot knee whose normalised hip HEIGHT matches the operator's."""
    want = normalised_hip_height(r_operator, hip_deg, knee_deg) * (1 + r_robot) - math.cos(math.radians(hip_deg))
    cosine = want / r_robot
    if abs(cosine) > 1:
        return knee_deg
    return hip_deg - math.degrees(math.acos(cosine))


def reference_for(neutral, index, hip_deg, knee_deg):
    reference = neutral.copy()
    for side in ("left", "right"):
        reference[index[f"{side}_hip_pitch_joint"]] = math.radians(hip_deg)
        reference[index[f"{side}_knee_joint"]] = math.radians(knee_deg)
    return reference


def hold(robot, reference, seconds):
    robot.reset()
    for _ in range(int(seconds / SIM_DT)):
        robot.balance.set_reference(reference, robot.entries)
        robot.balance.step()
    state = robot.balance.status()
    measured = measure(robot.model, robot.data)
    model, data = robot.model, robot.data

    def anchor(name):
        return data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)].copy()

    hip = 0.5 * (anchor("left_hip_pitch_joint") + anchor("right_hip_pitch_joint"))
    knee = 0.5 * (anchor("left_knee_joint") + anchor("right_knee_joint"))
    ankle = 0.5 * (anchor("left_ankle_pitch_joint") + anchor("right_ankle_pitch_joint"))
    leg = ankle - hip
    extension = float(np.linalg.norm(leg))
    tilt = math.degrees(math.atan2(leg[0], -leg[2]))
    return {
        "com_z_m": measured["com"][2],
        "com_m": measured["com"],
        "hip_above_ankle_m": float(hip[2] - ankle[2]),
        "hip_above_ankle_over_leg": float(hip[2] - ankle[2]) / G1_LEG_M,
        "leg_extension_m": extension,
        "leg_extension_over_leg": extension / G1_LEG_M,
        "achieved_leg_tilt_deg": tilt,
        "achieved_knee_flex_deg": float(np.degrees(data.qpos[
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")]])),
        "tilt_deg": measured["tilt_deg"],
        "foot_loads_n": measured["foot_loads_n"],
        "other_floor_contacts": measured["other_floor_contacts"],
        "max_torque_limit_ratio": state.get("max_torque_limit_ratio"),
        "frozen_joints": state.get("limited_joints"),
        "fallen": state.get("fallen"),
    }


def grid(r_operator, r_robot):
    """Normalised crouch error over the (hip, knee) plane, for three mappings."""
    out = []
    for hip in range(0, 41, 5):
        for knee in range(10, 101, 5):
            want = normalised_hip_height(r_operator, hip, knee)
            identity = normalised_hip_height(r_robot, hip, knee)
            tilt = knee_for_same_tilt(r_operator, r_robot, knee)
            height = knee_for_same_height(r_operator, r_robot, hip, knee)
            out.append({
                "hip_deg": hip, "knee_deg": knee,
                "identity_error": identity - want,
                "orientation_preserving_error":
                    normalised_hip_height(r_robot, hip, tilt) - want,
                "height_preserving_error": normalised_hip_height(r_robot, hip, height) - want,
                "tilt_error_deg": tilt_error_closed_form(r_operator, r_robot, knee),
                "identity_error_m": (identity - want) * G1_LEG_M,
            })
    return out


def main():
    r_operator, thigh_m, shin_m = operator_ratio()
    r_robot = G1_SHIN_M / G1_THIGH_M
    print(f"operator r={r_operator:.4f} (thigh {thigh_m:.4f}, shin {shin_m:.4f} m)   "
          f"G1 r={r_robot:.4f} (leg {G1_LEG_M:.4f} m)")

    plane = grid(r_operator, r_robot)
    identity_worst = max(plane, key=lambda r: abs(r["identity_error"]))
    orientation_worst = max(plane, key=lambda r: abs(r["orientation_preserving_error"]))
    height_worst = max(plane, key=lambda r: abs(r["height_preserving_error"]))
    print(f"grid: worst identity crouch error {identity_worst['identity_error'] * 100:+.2f}% of leg "
          f"({identity_worst['identity_error_m'] * 100:+.2f} cm) at hip {identity_worst['hip_deg']} / "
          f"knee {identity_worst['knee_deg']}")
    print(f"grid: worst orientation-preserving crouch error "
          f"{orientation_worst['orientation_preserving_error'] * 100:+.2f}% of leg")
    print(f"grid: worst height-preserving crouch error "
          f"{height_worst['height_preserving_error'] * 100:+.2f}% of leg")

    robot = JointRobot(MODEL, MAPPING)
    home = np.array([robot.data.qpos[e["qpos"]] for e in robot.entries])
    names = [e["rule"]["robot_joint"] for e in robot.entries]
    index = {name: i for i, name in enumerate(names)}
    neutral = home.copy()
    home_knee = math.degrees(home[index["left_knee_joint"]])
    print(f"robot home knee {home_knee:.2f} deg")

    plan = []
    for hip in (10.0, 20.0, 30.0):
        for knee in (30.0, 50.0, 70.0, 90.0):
            plan.append((hip, knee))

    sims = []
    for hip, knee in plan:
        want = normalised_hip_height(r_operator, hip, knee)
        variants = {
            "identity": knee,
            "orientation_preserving": knee_for_same_tilt(r_operator, r_robot, knee),
            "height_preserving": knee_for_same_height(r_operator, r_robot, hip, knee),
        }
        entry = {"hip_deg": hip, "knee_operator_deg": knee,
                 "operator_norm_height": want,
                 "operator_norm_height_m_on_robot": want * G1_LEG_M, "runs": {}}
        for label, knee_robot in variants.items():
            result = hold(robot, reference_for(neutral, index, hip, knee_robot), 3.5)
            result["knee_reference_deg"] = knee_robot
            result["norm_height_error_vs_operator"] = (
                result["hip_above_ankle_over_leg"] - want)
            result["norm_height_error_m"] = (
                result["hip_above_ankle_over_leg"] - want) * G1_LEG_M
            entry["runs"][label] = result
        sims.append(entry)
        print(f"  hip {hip:4.0f} knee {knee:4.0f} | operator norm h {want:.4f} | "
              + " | ".join(f"{k[:3]} {v['norm_height_error_m'] * 100:+.2f}cm"
                           for k, v in entry["runs"].items()))

    def worst(label, key):
        return max((abs(e["runs"][label][key]) for e in sims), default=0.0)

    # The comparison that matters: how far apart do the MAPPINGS land, holding
    # the controller fixed? A large gap means the choice of mapping matters.
    mapping_gap = [abs(e["runs"]["orientation_preserving"]["com_z_m"]
                       - e["runs"]["identity"]["com_z_m"]) for e in sims]
    plateau = [e for e in sims if e["runs"]["identity"]["achieved_knee_flex_deg"] < e["knee_operator_deg"] - 5]
    config = json.loads(MAPPING.read_text(encoding="utf-8-sig"))["balance"]

    report = {
        "generated_by": "analyze_retarget_balance.py",
        "mappings": {
            "identity": "robot knee = operator knee (the shipped rule)",
            "orientation_preserving": "hip unchanged, knee re-solved so the hip->ankle direction matches",
            "height_preserving": "hip unchanged, knee re-solved so the normalised crouch depth matches",
        },
        "operator_r_shin_over_thigh": r_operator,
        "robot_r_shin_over_thigh": r_robot,
        "locus_where_ratio_cancels": "q2 = 2 q1 makes the normalised hip height exactly cos(q1), "
                                     "independent of r -- identity is accidentally exact there",
        "robot_home_knee_deg": home_knee,
        "plane_grid": plane,
        "plane_worst": {
            "identity": identity_worst,
            "orientation_preserving": orientation_worst,
            "height_preserving": height_worst,
        },
        "simulated": sims,
        "confounds": {
            "max_crouch_m": float(config["max_crouch_m"]),
            "note": "the CoM goal drop is capped at max_crouch, so deep knees cannot be "
                    "reached; where the achieved knee plateaus the identity vs "
                    "orientation-preserving difference goes to zero and the measured error "
                    "against the operator belongs to the controller's cap, not the mapping",
            "rows_where_achieved_knee_plateaued": [
                {"hip_deg": e["hip_deg"], "knee_operator_deg": e["knee_operator_deg"],
                 "achieved_knee_deg": e["runs"]["identity"]["achieved_knee_flex_deg"]}
                for e in plateau],
            "height_preserving_ill_posed": "for deep crouches the operator's normalised "
                                           "height needs a negative knee angle, outside the "
                                           "joint range, so that variant clips and its large "
                                           "errors are artefacts, not a mapping property",
        },
        "summary": {
            "plane_max_identity_crouch_error_percent_of_leg":
                abs(identity_worst["identity_error"]) * 100,
            "plane_max_identity_crouch_error_m": abs(identity_worst["identity_error_m"]),
            "sim_max_identity_height_error_m": worst("identity", "norm_height_error_m"),
            "sim_max_orientation_height_error_m":
                worst("orientation_preserving", "norm_height_error_m"),
            "sim_max_height_preserving_height_error_m":
                worst("height_preserving", "norm_height_error_m"),
            "sim_max_gap_between_mappings_m": max(mapping_gap),
            "sim_max_gap_between_mappings_clamped_m":
                max((abs(e["runs"]["orientation_preserving"]["com_z_m"]
                         - e["runs"]["identity"]["com_z_m"]) for e in sims
                     if e["runs"]["identity"]["achieved_knee_flex_deg"]
                     >= e["knee_operator_deg"] - 5), default=None),
            "sim_max_torque_limit_ratio": max(
                e["runs"][k]["max_torque_limit_ratio"] for e in sims for k in e["runs"]),
            "sim_any_fall": any(e["runs"][k]["fallen"] for e in sims for k in e["runs"]),
            "sim_any_joint_frozen": any(e["runs"][k]["frozen_joints"] for e in sims for k in e["runs"]),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        plot(plane, sims, r_operator, r_robot, OUT_PNG)
        print(f"wrote {OUT_JSON.name} and {OUT_PNG.name}")
    except Exception as error:  # noqa: BLE001 - reporting must survive a plot failure
        print(f"wrote {OUT_JSON.name} (plot skipped: {error})")

    print()
    print(f"max identity crouch error over the plane: "
          f"{abs(identity_worst['identity_error']) * 100:.2f}% of leg length "
          f"({abs(identity_worst['identity_error_m']) * 100:.2f} cm)")
    print(f"sim: identity {worst('identity', 'norm_height_error_m') * 100:.2f} cm vs operator, "
          f"orientation-preserving {worst('orientation_preserving', 'norm_height_error_m') * 100:.2f} cm")
    print(f"sim: the two MAPPINGS differ by at most {max(mapping_gap) * 100:.2f} cm of CoM "
          f"(unclamped rows: "
          f"{report['summary']['sim_max_gap_between_mappings_clamped_m'] * 100:.2f} cm)")
    print(f"confound: crouch goal capped at {report['confounds']['max_crouch_m']} m; "
          f"{len(plateau)} of {len(sims)} rows plateaued before the commanded knee")


def plot(plane, sims, r_operator, r_robot, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    for axis, key, title in (
            (axes[0], "identity_error",
             "I  identity (shipped)\nrobot knee = operator knee"),
            (axes[1], "orientation_preserving_error",
             "T  orientation-preserving\nleg direction matched, knee re-solved")):
        hips = sorted({r["hip_deg"] for r in plane})
        knees = sorted({r["knee_deg"] for r in plane})
        data = np.array([[next(r[key] for r in plane if r["hip_deg"] == h and r["knee_deg"] == k) * 100
                          for k in knees] for h in hips])
        limit = max(1.0, float(np.max(np.abs(data))))
        image = axis.imshow(data, origin="lower", aspect="auto", cmap="RdBu_r",
                            vmin=-limit, vmax=limit,
                            extent=[knees[0], knees[-1], hips[0], hips[-1]])
        line = [k for k in np.linspace(knees[0], knees[-1], 50)]
        axis.plot(line, [k / 2 for k in line], "k--", linewidth=1.2,
                  label="q2 = 2 q1 (error vanishes)")
        figure.colorbar(image, ax=axis, label="normalised crouch error (% of leg length)")
        axis.set_xlabel("knee flexion (deg)")
        axis.set_ylabel("hip flexion (deg)")
        axis.set_title(title)
        axis.legend(fontsize=8, loc="upper left")

    axis = axes[2]
    labels = {"identity": ("tab:red", "s"), "orientation_preserving": ("tab:blue", "^"),
              "height_preserving": ("tab:green", "o")}
    for label, (color, marker) in labels.items():
        xs = [e["knee_operator_deg"] for e in sims if e["hip_deg"] == 20.0]
        ys = [e["runs"][label]["norm_height_error_m"] * 100
              for e in sims if e["hip_deg"] == 20.0]
        axis.plot(xs, ys, marker + "-", color=color, label=label.replace("_", " "))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("operator knee flexion (deg), hip fixed at 20 deg")
    axis.set_ylabel("achieved crouch error (cm, on the robot's leg)")
    axis.set_title("Measured in MuJoCo, StandingBalance\n"
                   "(0 = the operator's own normalised posture)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    figure.suptitle(f"Identity retargeting transfers joint angles, not postures - "
                    f"operator shin:thigh {r_operator:.3f} vs G1 {r_robot:.3f}", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
