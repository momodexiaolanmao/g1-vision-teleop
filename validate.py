"""Meaningful offline checks: reachability, dynamics, input limits, pose API."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from control import Arm, WristMapper, clamp_workspace
from download_model import MODEL_PATH
from joint_mapping import JointRobot, LANDMARK_IDS


def main():
    arm = Arm()
    waypoints = [[0.43, 0.0, 0.22], [0.35, 0.15, 0.38],
                 [0.35, -0.18, 0.32], [0.50, 0.0, 0.16]]
    errors = []
    for target in waypoints:
        arm.set_target(target)
        for _ in range(1500):
            arm.step()
            assert np.all(np.isfinite(arm.data.qpos)), "Non-finite simulation state"
            assert np.all(arm.data.qpos >= arm.limits[:, 0] - 0.01), "Joint below limit"
            assert np.all(arm.data.qpos <= arm.limits[:, 1] + 0.01), "Joint above limit"
        error = float(np.linalg.norm(arm.tool_position - target))
        assert error < 0.008, f"Arm failed to reach {target}: {error:.4f} m"
        errors.append(error)
    arm.hold()
    command = arm.command.copy()
    for _ in range(100):
        arm.step()
    assert np.allclose(arm.command, command), "Lost-input hold changed the command"
    try:
        arm.set_target([np.nan, 0, 0])
        raise AssertionError("Invalid target accepted")
    except ValueError:
        pass
    bounded = clamp_workspace([50, -50, 50])
    assert np.linalg.norm(bounded - [0, 0, 0.18]) <= 0.565001
    mapper = WristMapper(arm.home_target)
    assert mapper.update([0, 0, 0], 0) is None
    mapper.calibrate([0, 0, 0])
    mapped = mapper.update([0.3, -0.3, 8], 0.1)
    assert abs(mapped[0] - arm.home_target[0]) < 1e-9, "Planar mode used estimated depth"
    assert mapped[1] > arm.home_target[1] and mapped[2] > arm.home_target[2]
    assert np.linalg.norm(mapped - arm.home_target) <= 0.4 / 30 + 1e-9
    depth_mapper = WristMapper(arm.home_target, depth=True)
    depth_mapper.calibrate([0, 0, 0])
    assert depth_mapper.update([0, 0, -0.1], 0.1)[0] > arm.home_target[0]

    joint_robot = JointRobot(Path(__file__).parent / "arm.xml",
                             Path(__file__).parent / "joint_mapping.json")
    points = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0)
              for _ in range(33)]
    def point(name, xyz):
        points[LANDMARK_IDS[name]].x, points[LANDMARK_IDS[name]].y, points[LANDMARK_IDS[name]].z = xyz
    # Left shoulder 60 degrees, right shoulder 90 degrees, right elbow 90 degrees.
    point("left_shoulder", [-0.2, 0, 0])
    point("left_hip", [-0.2, 0.5, 0])
    point("left_elbow", [-0.2 - np.sqrt(3) / 4, 0.25, 0])
    point("right_shoulder", [0.2, 0, 0])
    point("right_hip", [0.2, 0.5, 0])
    point("right_elbow", [0.5, 0, 0])
    point("right_wrist", [0.5, 0.3, 0])
    observation = SimpleNamespace(landmarks=points, world_landmarks=points,
                                  frame=np.zeros((480, 640, 3)))
    mapped = joint_robot.targets_from_pose(observation)
    expected_deg = [0, -70, 72]
    assert np.allclose(np.degrees(mapped), expected_deg), "Custom/reversed mapping mismatch"
    joint_robot.set_targets(mapped)
    for _ in range(2000):
        old = joint_robot.command.copy()
        joint_robot.step()
        assert np.max(np.abs(joint_robot.command - old)) <= np.radians(60) * 0.002 + 1e-9
    actual = np.array([joint_robot.data.qpos[e["qpos"]] for e in joint_robot.entries])
    assert np.max(np.abs(np.degrees(actual - mapped))) < 0.6, "Joint servos failed to track mapping"
    points[LANDMARK_IDS["right_wrist"]].visibility = 0.1
    partially_mapped = joint_robot.targets_from_pose(observation)
    assert partially_mapped is not None, "Occluded wrist blocked unrelated shoulder mappings"
    assert partially_mapped[2] == joint_robot.command[2], "Occluded elbow did not hold its setpoint"
    assert not joint_robot.pose_diagnostics[2]["valid"], "Occluded elbow was marked valid"
    joint_robot.hold()
    joint_command = joint_robot.command.copy()
    for _ in range(100):
        joint_robot.step()
    assert np.array_equal(joint_robot.command, joint_command)

    pose_smoke = "not run: model missing"
    if MODEL_PATH.exists():
        import mediapipe as mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
            result = pose.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.zeros((480, 640, 3), dtype=np.uint8)), 0
            )
            assert not result.pose_landmarks, "Empty image unexpectedly contained a pose"
        pose_smoke = "passed: model loaded and empty image processed"

    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with mujoco.Renderer(arm.model, height=720, width=960) as renderer:
        renderer.update_scene(arm.data, camera="overview")
        pixels = renderer.render()
        import cv2
        assert cv2.imwrite(str(artifacts / "arm_preview.png"), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    report = {
        "mujoco_version": mujoco.__version__, "arm_dof": arm.model.nv,
        "waypoints_m": waypoints, "settled_error_mm": [1000 * e for e in errors],
        "joint_limits_and_dynamics": "passed", "lost_input_hold": "passed",
        "invalid_target_rejection": "passed", "mapping_and_speed_limits": "passed",
        "custom_joint_mapping": "passed: 60/90/90-degree human angles -> 0/-70/72-degree robot targets",
        "mapped_joint_occlusion": "passed: low-confidence wrist holds elbow, visible shoulder mappings remain active",
        "joint_mapping_settled_error_deg": np.abs(np.degrees(actual - mapped)).tolist(),
        "pose_model_smoke": pose_smoke,
        "camera_person_tracking": "not tested: no camera/person input",
        "full_end_to_end_latency": "not measured",
    }
    (artifacts / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
