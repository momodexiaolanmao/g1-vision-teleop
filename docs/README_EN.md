# G1 Vision Teleoperation

A camera-based teleoperation prototype for the Unitree G1 in MuJoCo, built for an AI control course. It combines MediaPipe pose estimation, geometric joint mapping, a pretrained Unitree locomotion policy, standing inverse-dynamics QP and torque-based physical simulation.

The default mode follows upper-body references and interprets stepping intent. The policy coordinates the legs; this is not arbitrary dance reproduction or exact left/right foot phase tracking. Only simulation has been validated, using the official 29-DOF G1 model rather than a verified G1 Pro hardware configuration.

## Run on Windows

Tested with 64-bit Python 3.14. Connect an OpenCV-compatible camera; DJI Osmo Action 3 USB Webcam mode was used during development.

```powershell
git clone https://github.com/momodexiaolanmao/g1-vision-teleop.git
cd g1-vision-teleop
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\start_unitree_g1.cmd
```

The application allows 8 seconds to enter the camera view, then calibrates automatically once the full body is reliably visible and steady. Keep both feet grounded and arms relaxed. C recalibrates, N reselects the person, Space pauses, R resets and Q/Esc exits.

## Architecture and evidence

MediaPipe Full supplies 33 landmarks. Geometry and configuration produce joint references. A pretrained ONNX policy consumes 96-dimensional observations with five histories and outputs 29 actions. Standing QP and stepping PD write actuator torques to `data.ctrl`; MuJoCo advances the physical state with `mj_step`.

[Architecture](ARCHITECTURE.md) · [Validation and limitations](VALIDATION.md) · [Course materials](course/README.md)

The repository includes third-party model and policy provenance. It does not claim to have trained the MediaPipe or Unitree networks. Demonstration logging is implemented, while a curated imitation-learning dataset and an autonomous learned policy remain future work.

Private camera video, raw human landmark traces, field JSONL, environments and credentials are excluded. Two private-input field regressions are skipped explicitly; public synthetic and physical regressions remain runnable. The public presentation has no embedded personal video or camera poster.

Original code and documentation are MIT licensed. Third-party assets retain their own licenses; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
