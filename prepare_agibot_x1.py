"""Prepare an attributed, fixed-pelvis teleoperation scene from the official X1 MJCF.

Run after a sparse checkout of AgibotTech/agibot_x1_infer into vendor/.
The upstream model is preserved; this script creates a separate simulation scene.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).absolute().parent
UPSTREAM_COMMIT = "9e0b818804d644fb9c9663e932dd33b03b24dfa4"
UPSTREAM_REPO = "https://github.com/AgibotTech/agibot_x1_infer"


def main():
    vendor = ROOT / "vendor" / "agibot_x1_infer"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}; found {revision}")
    source = vendor / "src" / "module" / "sim_module" / "model"
    robot_source = source / "mjcf" / "robot" / "xyber_x1" / "xyber_x1_serial.xml"
    target = ROOT / "robots" / "agibot_x1"
    (target / "meshes").mkdir(parents=True, exist_ok=True)
    (target / "upstream").mkdir(exist_ok=True)
    shutil.copy2(robot_source, target / "upstream" / "xyber_x1_serial.xml")
    shutil.copy2(vendor / "LICENSE.txt", target / "LICENSE.txt")
    scene = ET.parse(robot_source).getroot()
    scene.set("model", "AgiBot X1 camera control")
    compiler = scene.find("compiler")
    compiler.set("meshdir", "meshes")
    scene.find("option").set("timestep", "0.002")
    scene.find("option").set("integrator", "implicitfast")
    scene.find("visual/global").set("offwidth", "1280")
    scene.find("visual/global").set("offheight", "900")
    # Retain the official geometry and inertia. Only the root support and
    # simulation actuators are adapted; no real-robot interface is used.
    world = scene.find("worldbody")
    pelvis = world.find("body[@name='x1-body']")
    pelvis.remove(pelvis.find("freejoint"))
    for body in world.iter("body"):
        body.set("gravcomp", "1")
    # Ground-contact diagnostic markers from the upstream walking model do
    # not belong to the camera-control visualisation.
    for body in world.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("type") == "sphere" and geom.get("class") != "collision":
                body.remove(geom)
    meshes = scene.find("asset").findall("mesh")
    for mesh in meshes:
        shutil.copy2(source / "meshes" / mesh.get("file"), target / "meshes" / mesh.get("file"))
    ET.SubElement(scene, "statistic", center="0 0 0.75", extent="1.4")
    asset = scene.find("asset")
    ET.SubElement(asset, "texture", name="floor_checker", type="2d", builtin="checker",
                  rgb1="0.19 0.23 0.27", rgb2="0.13 0.16 0.19", width="512", height="512")
    ET.SubElement(asset, "material", name="floor_material", texture="floor_checker",
                  texrepeat="8 8", texuniform="true", reflectance="0.08")
    ET.SubElement(world, "light", name="key", pos="2 -2 4", dir="-0.4 0.3 -1", directional="true")
    ET.SubElement(world, "light", name="fill", pos="-2 2 3", dir="0.4 -0.2 -1", directional="true")
    floor = ET.SubElement(world, "geom", name="floor", type="plane", size="4 4 0.1",
                          pos="0 0 -0.1", material="floor_material", contype="0", conaffinity="7")
    joints = {j.get("name"): j for j in world.iter("joint")}
    actuators = scene.find("actuator")
    for motor in list(actuators):
        joint = joints[motor.get("joint")]
        force_range = motor.get("ctrlrange", "-50 50")
        motor.tag = "position"
        motor.attrib.clear()
        motor.attrib.update(name=joint.get("name").replace("_joint", "_servo"),
                            joint=joint.get("name"), kp="160", kv="14",
                            ctrlrange=joint.get("range"), ctrllimited="true",
                            forcerange=force_range, forcelimited="true")
    # Every unused position servo holds a defined neutral pose as well.
    home = {name: 0.0 for name in joints}
    for side in ("left", "right"):
        home[f"{side}_shoulder_roll_joint"] = -0.10
        home[f"{side}_elbow_pitch_joint"] = 0.15
    keyframes = scene.find("keyframe")
    keyframes.clear()
    ordered_joints = list(joints)
    qpos = " ".join(str(home[name]) for name in ordered_joints)
    ctrl = " ".join(str(home[a.get("joint")]) for a in actuators)
    ET.SubElement(keyframes, "key", name="camera_home", qpos=qpos, ctrl=ctrl)
    output = target / "scene.xml"
    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    # Put the floor immediately below the neutral foot contact points without
    # applying a supporting contact force to the fixed-pelvis scene.
    feet = [i for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
    floor_height = min(float(data.geom_xpos[i, 2] - model.geom_size[i, 0]) for i in feet) - 0.001
    floor.set("pos", f"0 0 {floor_height:.6f}")
    ET.ElementTree(scene).write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.nq == 29 and model.nu == 29
    manifest = []
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        actuator = next(i for i in range(model.nu) if model.actuator_trnid[i, 0] == index)
        manifest.append({"joint": name, "type": "mjJNT_HINGE",
                         "actuators": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)],
                         "range_deg": np.degrees(model.jnt_range[index]).tolist()})
    (ROOT / "agibot_x1_joints.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping = json.loads((ROOT / "agibot_x1_mapping.json").read_text(encoding="utf-8"))
    editor = (ROOT / "mapping_editor.html").read_text(encoding="utf-8")
    editor = editor.replace("<title>人体动作 → 机器人关节</title>", "<title>智元灵犀 X1 关节映射</title>")
    editor = editor.replace("让你选择的动作，驱动你选择的关节。", "用你的动作控制智元灵犀 X1")
    editor = editor.replace("添加对应关系，设置人体和机器人的角度范围。",
                            "已载入 X1 的 29 个关节，默认控制双肩与双肘。添加对应关系，设置人体和机器人的角度范围。")
    editor = re.sub(r"let joints=\[.*?\];", "let joints=" + json.dumps(manifest, ensure_ascii=False) + ";", editor, count=1)
    editor = re.sub(r"let rules=\[.*?\];", "let rules=" + json.dumps(mapping["rules"], ensure_ascii=False) + ";", editor, count=1)
    editor = editor.replace("joint_mapping.json", "agibot_x1_mapping.json")
    metadata = ",".join(key + ":" + json.dumps(mapping[key], ensure_ascii=False)
                        for key in ("description", "display_name", "viewer")) + ","
    editor = editor.replace("description:'Custom human-to-robot joint mapping',", metadata)
    (ROOT / "agibot_x1_mapping_editor.html").write_text(editor, encoding="utf-8")
    attribution = {
        "robot": "AgiBot / 智元灵犀 X1", "upstream_repository": UPSTREAM_REPO,
        "upstream_commit": revision,
        "upstream_model": "src/module/sim_module/model/mjcf/robot/xyber_x1/xyber_x1_serial.xml",
        "license": "Mulan PSL v2 (see LICENSE.txt)", "mesh_files": len(meshes),
        "simulation_changes": ["fixed pelvis", "gravity compensation", "position servos with original motor force limits",
                               "neutral keyframe", "floor and lighting", "removed foot diagnostic markers"],
        "joint_count": model.njnt, "actuator_count": model.nu,
        "default_mapped_joints": [r["robot_joint"] for r in mapping["rules"]],
    }
    (target / "SOURCE.json").write_text(json.dumps(attribution, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(attribution, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
