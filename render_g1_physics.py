"""Render an actual forward-dynamics standing-follow demonstration."""
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from joint_mapping import JointRobot
from validate_unitree_g1 import pose

ROOT = Path(__file__).absolute().parent


def main():
    robot = JointRobot(ROOT / "robots/unitree_g1/scene_physics.xml",
                       ROOT / "unitree_g1_physics_mapping.json")
    robot.calibrate(pose())
    output = ROOT / "artifacts"
    writer = cv2.VideoWriter(str(output / "unitree_g1_physics_demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20, (960, 720))
    if not writer.isOpened():
        raise RuntimeError("Cannot create physics demonstration video")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.05, 0, 0.68]
    camera.distance, camera.azimuth, camera.elevation = 2.65, 145, -13
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0
    samples = []
    try:
        with mujoco.Renderer(robot.model, height=720, width=960) as renderer:
            for frame in range(280):
                seconds = frame / 20
                robot.set_targets(robot.targets_from_pose(pose(max(0, seconds - 3), dancing=seconds >= 3)))
                for _ in range(25):
                    robot.step()
                status = robot.balance.status()
                samples.append(status)
                renderer.update_scene(robot.data, camera=camera, scene_option=options)
                image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                text = ["UNITREE G1 | physical standing follow | synthetic input",
                        "Gravity + foot contact + bounded joint torques",
                        f"Foot loads: {status['foot_loads_n'][0]:.0f} / {status['foot_loads_n'][1]:.0f} N"
                        f"  |  Body tilt: {status['tilt_deg']:.1f} deg"]
                for row, message in enumerate(text):
                    cv2.putText(image, message, (20, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX,
                                .57, (130, 240, 190) if row else (240, 247, 250), 1, cv2.LINE_AA)
                writer.write(image)
                if frame in (40, 125, 215):
                    cv2.imwrite(str(output / f"unitree_g1_physics_pose_{frame}.png"), image)
    finally:
        writer.release()
    report = {"input": "synthetic anatomical landmarks, not a recording of the user",
              "mode": "standing: mj_step with an unactuated free root",
              "duration_s": robot.data.time, "gravity": robot.model.opt.gravity.tolist(),
              "any_fall": any(s["fallen"] for s in samples),
              "max_tilt_deg": max(s["tilt_deg"] for s in samples),
              "qp_failures": robot.balance.failures, "motor_force_limit_ratio": robot.balance.max_torque_ratio,
              "video": "unitree_g1_physics_demo.mp4"}
    (output / "unitree_g1_physics_demo_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
