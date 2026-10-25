"""Check the official X1 scene, servo dynamics, visible-joint mapping and render it."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np

from joint_mapping import JointRobot, LANDMARK_IDS


ROOT = Path(__file__).absolute().parent


def actual(robot):
    return np.array([robot.data.qpos[e["qpos"]] for e in robot.entries])


def settle(robot, seconds=3):
    for _ in range(int(seconds / robot.model.opt.timestep)):
        robot.step()
    assert np.all(np.isfinite(robot.data.qpos))
    assert np.all(np.isfinite(robot.data.qvel))


def main():
    robot = JointRobot(ROOT / "robots" / "agibot_x1" / "scene.xml", ROOT / "agibot_x1_mapping.json")
    assert robot.model.nq == robot.model.nv == robot.model.njnt == robot.model.nu == 29
    assert np.all(robot.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    neutral = robot.data.qpos.copy()
    settle(robot, 2)
    neutral_drift = float(np.max(np.abs(np.degrees(robot.data.qpos - neutral))))
    assert neutral_drift < 0.25, neutral_drift
    static_errors = []
    wrist_positions = []
    for degrees in ([-45, -60, 70, 40], [-5, -15, 5, 85], [-80, -80, 50, 50]):
        targets = np.radians(degrees)
        robot.set_targets(targets)
        settle(robot, 4)
        error = float(np.max(np.abs(np.degrees(actual(robot) - targets))))
        assert error < 0.5, (degrees, error)
        static_errors.append(error)
        wrist_positions.append(robot.data.body("link_right_wrist_roll").xpos.copy())
    wrist_motion = float(np.linalg.norm(wrist_positions[0] - wrist_positions[1]))
    assert wrist_motion > 0.05, wrist_motion
    unused = [i for i in range(robot.model.nq) if i not in {e["qpos"] for e in robot.entries}]
    unused_error = float(np.max(np.abs(np.degrees(robot.data.qpos[unused] - neutral[unused]))))
    assert unused_error < 0.5, unused_error
    points = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.95, presence=0.95) for _ in range(33)]
    coordinates = {
        "left_hip": (0, 0, 0), "left_shoulder": (0, 0, 1),
        "left_elbow": (0.3, 0, 1), "left_wrist": (0.3, 0, 1.3),
        "right_hip": (0, 1, 0), "right_shoulder": (0, 1, 1),
        "right_elbow": (0.3, 1, 1), "right_wrist": (0.6, 1, 1),
    }
    for name, coordinates_xyz in coordinates.items():
        point = points[LANDMARK_IDS[name]]
        point.x, point.y, point.z = coordinates_xyz
    pose = SimpleNamespace(landmarks=points, world_landmarks=points)
    targets = robot.targets_from_pose(pose)
    expected = np.array([-78.6363636364, -78.6363636364, 61.0714285714, 0])
    assert np.allclose(np.degrees(targets), expected, atol=1e-8)
    points[LANDMARK_IDS["left_hip"]].visibility = 0.1
    points[LANDMARK_IDS["right_hip"]].visibility = 0.1
    partial = robot.targets_from_pose(pose)
    assert partial is not None
    assert [r["valid"] for r in robot.pose_diagnostics] == [False, False, True, True]
    assert np.allclose(partial[:2], robot.command[:2])
    robot.set_targets(targets)
    settle(robot, 4)
    pose_error = float(np.max(np.abs(np.degrees(actual(robot)) - expected)))
    assert pose_error < 0.5, pose_error
    artifact = ROOT / "artifacts"
    artifact.mkdir(exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.72]
    camera.distance = 2.45
    camera.azimuth = 145
    camera.elevation = -12
    with mujoco.Renderer(robot.model, height=768, width=1024) as renderer:
        renderer.update_scene(robot.data, camera=camera)
        pixels = renderer.render()
        assert cv2.imwrite(str(artifact / "agibot_x1_preview.png"), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    report = {
        "robot": "AgiBot X1", "joints": 29, "actuators": 29, "pelvis": "fixed",
        "neutral_hold_max_drift_deg": neutral_drift,
        "static_joint_max_error_deg": static_errors,
        "unused_joint_max_error_deg": unused_error,
        "right_wrist_motion_m": wrist_motion,
        "human_fixture_target_deg": expected.tolist(),
        "human_fixture_follow_max_error_deg": pose_error,
        "hidden_hips_visible_elbows": "passed", "finite_dynamics": "passed",
        "renderer": "passed", "real_camera": "requires separate live verification",
    }
    (artifact / "agibot_x1_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
