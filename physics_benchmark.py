"""Fast, camera-free measurement of genuine unsupported G1 dynamics."""
import json
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).absolute().parent


def measure(model, data):
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rotation = data.xmat[pelvis].reshape(3, 3)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_ankle_roll_link")
            for side in ("left", "right")]
    loads = [0., 0.]
    other_contacts = 0
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(data.ncon):
        contact = data.contact[i]
        if floor not in (contact.geom1, contact.geom2):
            continue
        geom = contact.geom2 if contact.geom1 == floor else contact.geom1
        body = int(model.geom_bodyid[geom])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if body in feet:
            loads[feet.index(body)] += max(0., float(force[0]))
        elif force[0] > 1:
            other_contacts += 1
    return {"height_m": float(data.qpos[2]),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
            "foot_loads_n": loads, "other_floor_contacts": other_contacts,
            "com": data.subtree_com[pelvis].tolist(),
            "max_joint_speed": float(np.max(np.abs(data.qvel[6:]))),
            "external_force_max": float(np.max(np.abs(data.qfrc_applied))),
            "body_force_max": float(np.max(np.abs(data.xfrc_applied)))}


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robots/unitree_g1/scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    records = []
    started = time.perf_counter()
    for step in range(5000):
        mujoco.mj_step(model, data)
        if step % 250 == 0:
            mujoco.mj_forward(model, data)
            records.append({"time": data.time, **measure(model, data)})
    report = {"mode": "unbalanced_position_servos", "gravity": model.opt.gravity.tolist(),
              "simulated_seconds": data.time, "wall_seconds": time.perf_counter() - started,
              "samples": records}
    destination = ROOT / "artifacts/unitree_g1_physics_baseline.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}))
    print(json.dumps(records[-1]))


if __name__ == "__main__":
    main()
