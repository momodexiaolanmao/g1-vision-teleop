"""Reproduce a portable 29-DOF shadow scene from the official Unitree MJCF."""
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from pose_retarget import CHANNEL_LABELS

ROOT = Path(__file__).absolute().parent
COMMIT = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
REPOSITORY = "https://github.com/unitreerobotics/unitree_mujoco"


def main():
    vendor = ROOT / "vendor" / "unitree_mujoco"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Expected {COMMIT}, found {revision}")
    source = vendor / "unitree_robots" / "g1"
    target = ROOT / "robots" / "unitree_g1"
    (target / "meshes").mkdir(parents=True, exist_ok=True)
    (target / "upstream").mkdir(exist_ok=True)
    shutil.copy2(source / "g1_29dof.xml", target / "upstream" / "g1_29dof.xml")
    shutil.copy2(vendor / "LICENSE", target / "LICENSE")
    scene = ET.parse(source / "g1_29dof.xml").getroot()
    scene.set("model", "Unitree G1 full body shadow")
    ET.SubElement(scene, "option", timestep="0.002", integrator="implicitfast")
    ET.SubElement(scene, "statistic", center="0 0 0.7", extent="1.5")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="900")
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.6 0.6 0.6")
    asset, world = scene.find("asset"), scene.find("worldbody")
    meshes = asset.findall("mesh")
    for mesh in meshes:
        shutil.copy2(source / "meshes" / mesh.get("file"), target / "meshes" / mesh.get("file"))
    ET.SubElement(asset, "texture", name="floor_checker", type="2d", builtin="checker",
                  rgb1="0.16 0.20 0.24", rgb2="0.11 0.14 0.18", width="512", height="512")
    ET.SubElement(asset, "material", name="floor_material", texture="floor_checker",
                  texrepeat="8 8", texuniform="true", reflectance="0.08")
    ET.SubElement(world, "light", pos="3 -2 4", dir="-0.4 0.2 -1", directional="true")
    ET.SubElement(world, "light", pos="-2 3 3", dir="0.3 -0.2 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 6 0.1", material="floor_material", group="2")
    joints = {j.get("name"): j for j in world.iter("joint") if j.get("type") != "free"}
    actuators = scene.find("actuator")
    for motor in actuators:
        joint = joints[motor.get("joint")]
        force_range = motor.get("ctrlrange")
        motor.tag = "position"
        motor.attrib.clear()
        motor.attrib.update(name=joint.get("name").replace("_joint", "_servo"),
                            joint=joint.get("name"), kp="100", kv="10",
                            ctrlrange=joint.get("range"), ctrllimited="true",
                            forcerange=force_range, forcelimited="true")
    home = {name: 0.0 for name in joints}
    for side in ("left", "right"):
        # At elbow zero the model's forearm points forward. +90 degrees
        # aligns it with a relaxed straight human arm.
        home[f"{side}_elbow_joint"] = np.pi / 2
        home[f"{side}_shoulder_roll_joint"] = 0.07 if side == "left" else -0.07
    keyframes = ET.SubElement(scene, "keyframe")
    key = ET.SubElement(keyframes, "key", name="shadow_home",
                        qpos="0 0 0.793 1 0 0 0 " + " ".join(str(home[n]) for n in joints),
                        ctrl=" ".join(str(home[a.get("joint")]) for a in actuators))
    output = target / "scene.xml"
    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    feet = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
            and "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or "")]
    bottom = min(data.geom_xpos[i, 2] - model.geom_size[i, 0] for i in feet)
    qpos = data.qpos.copy()
    qpos[2] += 0.001 - bottom
    key.set("qpos", " ".join(str(value) for value in qpos))
    ET.ElementTree(scene).write(output, encoding="utf-8", xml_declaration=True)
    manifest, rules = [], []
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        actuator = next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == i)
        servo = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        limits = np.degrees(model.jnt_range[i]).tolist()
        manifest.append({"joint": name, "type": "mjJNT_HINGE", "actuators": [servo], "range_deg": limits})
        channel = name.removesuffix("_joint")
        if channel.endswith("elbow") or channel.endswith("knee"):
            channel += "_flex"
        if channel.endswith("elbow_flex"):
            human, robot = [0, 150], [90, -60]
        elif channel.endswith("knee_flex"):
            human, robot = [0, 165], [0, 165]
        else:
            human, robot = [-180, 180], [-180, 180]
        rules.append({"label": CHANNEL_LABELS[channel], "source": {"type": "retarget", "channel": channel},
                      "robot_joint": name, "actuator": servo, "human_range_deg": human,
                      "robot_range_deg": robot, "offset_deg": 0,
                      "max_speed_deg_s": 240 if "shoulder" in channel else 180})
    mapping = {"description": "Unitree G1 29-DOF full-body kinematic shadow preview",
               "display_name": "Unitree G1 | 29 joints", "control_mode": "shadow",
               "confidence": 0.6, "smoothing_seconds": 0.07,
               "root_tracking": {"estimate_depth": True, "camera_distance_m": 2.5,
                                 "max_lateral_m": 1.8, "max_depth_m": 1.2, "max_speed_m_s": 1.8,
                                 "max_turn_deg_s": 240, "smoothing_seconds": 0.12, "floor_guard": True},
               "viewer": {"azimuth": 150, "elevation": -10, "distance": 3.6, "lookat": [0, 0, 0.75]},
               "rules": rules}
    (ROOT / "unitree_g1_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "unitree_g1_joints.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {"upstream_repository": REPOSITORY, "upstream_commit": revision,
                  "upstream_model": "unitree_robots/g1/g1_29dof.xml", "license": "BSD-3-Clause (see LICENSE)",
                  "mesh_files": len(meshes), "hinge_joints": 29, "free_root": True,
                  "simulation_changes": ["position actuator definitions", "neutral home keyframe", "floor and lights"],
                  "execution_mode": "Kinematic qpos preview; dynamics and balance are not integrated",
                  "model_variant": "Official G1 29dof; not a separately verified G1 Pro SKU or finger-hand model"}
    (target / "SOURCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
