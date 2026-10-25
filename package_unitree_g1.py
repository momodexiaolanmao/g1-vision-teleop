"""Create and verify a portable G1 source/model package, excluding live state."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).absolute().parent


def main():
    names = ("teleop.py joint_mapping.py pose_retarget.py shadow_motion.py control.py tracking_session.py "
             "download_model.py setup.ps1 requirements.txt requirements-lock.txt arm.xml joint_mapping.json "
             "mapping_editor.html robot_joints.json start_unitree_g1.cmd unitree_g1_mapping.json "
             "unitree_g1_joints.json unitree_g1_mapping_editor.html UNITREE_G1_zh.md README_zh.md "
             "build_unitree_editor.py prepare_unitree_g1.py validate_unitree_g1.py "
             "validate_unitree_editor.cjs test_tracking_input.py check_live_tracking.py package_unitree_g1.py "
             "start_demo.cmd start_joint_control.cmd start_hand_follow.cmd validate.py validate_editor.cjs").split()
    names += ("standing_balance.py dense_qp.py physics_benchmark.py prepare_g1_physics.py "
              "test_g1_physics.py render_g1_physics.py start_unitree_g1_physics.cmd start_unitree_g1_shadow.cmd "
              "unitree_g1_physics_mapping.json unitree_g1_physics_mapping_editor.html "
              "G1_PHYSICS_zh.md G1_SHADOW_zh.md requirements-physics-extra.txt").split()
    names += ("g1_locomotion.py hybrid_balance.py footstep_intent.py subject_tracker.py test_subject_tracking.py "
              "control_trace.py replay_control_trace.py sampling_session.py workflow_display.py "
              "test_sampling_session.py test_calibration_workflow.py test_live_status.py check_unitree_g1_static.cmd check_unitree_g1_motion.cmd "
              "test_pose_quality.py test_footsteps.py "
              "check_pose_stability.py check_g1_locomotion.py render_g1_stepping.py fetch_assets.py "
              "start_unitree_g1_walking.cmd unitree_g1_walking_mapping.json "
              "unitree_g1_walking_mapping_editor.html requirements-walking.txt G1_STEPPING_zh.md").split()
    names += ("twist2_tracking.py twist2_camera.py fetch_twist2_assets.py "
              "start_unitree_g1_tracking.cmd unitree_g1_tracking_mapping.json "
              "unitree_g1_tracking_joints.json unitree_g1_tracking_mapping_editor.html "
              "OTHER_PROJECTS_zh.md G1_TRACKING_zh.md "
              "test/check_twist2_tracking.py test/test_twist2_camera.py").split()
    paths = [ROOT / name for name in names]
    paths += list((ROOT / "robots/unitree_g1").rglob("*"))
    paths += list((ROOT / "robots/unitree_g1_twist2").rglob("*"))
    paths += [ROOT / "models/pose_landmarker_lite.task", ROOT / "templates/unitree_g1_editor.html"]
    paths += [ROOT/'models/pose_landmarker_full.task']
    paths += [ROOT/'policies/g1_velocity'/n for n in ('policy.onnx','deploy.yaml','deploy.json','LICENSE','SOURCE.json')]
    paths += [ROOT/'policies/twist2_tracking'/n for n in ('policy.onnx','LICENSE','SOURCE.json')]
    paths += [ROOT/'fixtures'/n for n in ('stationary_feet.json','upper_reference_from_failed_run.json',
                                        'upper_reference_from_hybrid_failed_run.json','twist2_walk.pkl')]
    # Curated numeric reports include the experimental camera failure explicitly.
    # Personal landmark traces, camera photos and runtime state remain local.
    artifact_names=(
        'unitree_g1_footstep_validation.json','unitree_g1_long_switch_validation.json',
        'unitree_g1_failed_reference_replay.json','unitree_g1_walking_stress.json',
        'unitree_g1_stepping_demo.mp4','unitree_g1_stepping_demo_check.json',
        'unitree_g1_stepping_pose.png',
        'unitree_g1_handsfree_static_check.json',
        'unitree_g1_regression_check.json',
        'unitree_g1_step_entry_replay_20261012.json',
        'unitree_g1_step_entry_live_20261012.json',
        'unitree_g1_step_entry_continuous_20261012.json',
        'unitree_g1_step_velocity_real_replay.json',
        'unitree_g1_step_velocity_live_20261012.json',
        'unitree_g1_twist2_validation.json',
        'unitree_g1_twist2_camera_validation.json',
        'unitree_g1_twist2_camera_mapping_validation.json',
        'unitree_g1_twist2_camera_live_failed.json')
    paths += [ROOT/'artifacts'/name for name in artifact_names]
    output = ROOT.parent / "mujoco_unitree_g1_stepping_20261012.zip"
    paths = sorted(set(p for p in paths if p.is_file()))
    for name in names:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            name = "mujoco_unitree_g1/" + path.relative_to(ROOT).as_posix()
            if path.name == "README_zh.md":
                # This package contains G1 and the lightweight arm examples;
                # X1 remains available in the original workspace/package.
                text = path.read_text(encoding="utf-8")
                text = "\n\n".join(p for p in text.split("\n\n") if "start_agibot_x1.cmd" not in p)
                archive.writestr(name, text)
            else:
                archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP checksum failure: {bad}")
        entries = archive.namelist()
        meshes = [name for name in entries if "/robots/unitree_g1/meshes/" in name]
        assert len(meshes) == 36
        assert any(name.endswith("models/pose_landmarker_lite.task") for name in entries)
        assert any(name.endswith('models/pose_landmarker_full.task') for name in entries)
        assert any(name.endswith('policies/g1_velocity/policy.onnx') for name in entries)
        twist_meshes = [name for name in entries if "/robots/unitree_g1_twist2/meshes/" in name]
        assert len(twist_meshes) == 35
        assert any(name.endswith('policies/twist2_tracking/policy.onnx') for name in entries)
        assert not any(name.endswith('.jsonl') for name in entries)
        assert not any(name.endswith(('/live_status.json','/live_status.tmp'))
                       or "/vendor/" in name or "/.venv/" in name for name in entries)
    report = {"archive": str(output), "bytes": output.stat().st_size, "files": len(entries),
              "mesh_files": len(meshes), "zip_integrity": "passed",
              "experimental_tracking_mesh_files": len(twist_meshes),
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "dependencies": "run setup.ps1 on a new computer; .venv is not included"}
    (ROOT / "artifacts/unitree_g1_stepping_package_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
