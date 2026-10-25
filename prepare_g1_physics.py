"""Build a separate free-base, torque-actuated G1 standing scene/profile."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).absolute().parent


def main():
    destination = ROOT / "robots/unitree_g1/scene_physics.xml"
    tree = ET.parse(ROOT / "robots/unitree_g1/scene.xml")
    scene = tree.getroot()
    scene.set("model", "Unitree G1 physical standing follow")
    option = scene.find("option")
    option.set("gravity", "0 0 -9.81")
    option.set("cone", "elliptic")
    option.set("iterations", "80")
    for actuator in scene.find("actuator"):
        name, joint, limits = (actuator.get(key) for key in ("name", "joint", "forcerange"))
        actuator.tag = "motor"
        actuator.attrib.clear()
        actuator.attrib.update(name=name, joint=joint, gear="1", ctrllimited="true",
                              ctrlrange=limits, forcelimited="true", forcerange=limits)
    # Keep the official foot collision positions and realistic finite friction.
    for geom in scene.find("worldbody").iter("geom"):
        if geom.get("type") == "sphere" or geom.get("name") == "floor":
            geom.set("friction", "0.8 0.02 0.001")
            geom.set("condim", "3")
            geom.set("solref", "0.01 1")
    key = scene.find("keyframe/key")
    key.set("name", "standing_home")
    qpos = np.fromstring(key.get("qpos"), sep=" ")
    joints = [j for j in scene.find("worldbody").iter("joint") if j.get("type") != "free"]
    for index, joint in enumerate(joints):
        name = joint.get("name")
        if name.endswith("hip_pitch_joint"):
            qpos[7 + index] = -0.16
        elif name.endswith("knee_joint"):
            qpos[7 + index] = 0.32
        elif name.endswith("ankle_pitch_joint"):
            qpos[7 + index] = -0.16
    key.set("qpos", " ".join(map(str, qpos)))
    key.set("ctrl", " ".join(["0"] * 29))
    ET.indent(scene, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(destination))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    feet = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
    bottom = min(data.geom_xpos[i, 2] - model.geom_size[i, 0] for i in feet)
    qpos[2] -= bottom  # Initialization only; the controller never writes the root.
    key.set("qpos", " ".join(map(str, qpos)))
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    mapping = json.loads((ROOT / "unitree_g1_mapping.json").read_text(encoding="utf-8-sig"))
    mapping.update(description="Free-base G1 with torque and double-support balance constraints",
                   display_name="Unitree G1 | Physical standing follow", control_mode="standing")
    mapping.pop("root_tracking", None)
    mapping["pose_quality"] = {"enabled": True, "neutral_correction": True,
                               "max_angular_speed_deg_s": 650, "min_cutoff_hz": 3,
                               "deadband_deg": 1.5}
    mapping["balance"] = {"control_period_s": 0.006, "friction": 0.65,
                          "max_crouch_m": 0.12, "max_com_shift_m": 0.02,
                          "upper_body_speed_deg_s": 90, "waist_yaw_limit_deg": 35,
                          "waist_roll_limit_deg": 12, "waist_pitch_limit_deg": 18}
    (ROOT / "unitree_g1_physics_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "robots/unitree_g1/PHYSICS_SOURCE.json").write_text(json.dumps({
        "source": "SOURCE.json", "scene": destination.name,
        "changes": ["torque motors using upstream force limits", "gravity and finite foot friction",
                    "slightly bent-knee initial stance"],
        "root": "unactuated free joint", "external_root_forces": False,
        "controller": "Constrained inverse dynamics in double support; no learned policy",
    }, indent=2), encoding="utf-8")
    print("Prepared free-base standing scene and independent mapping profile")


if __name__ == "__main__":
    main()
