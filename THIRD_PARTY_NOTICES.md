# Third-party software, models and data

The root MIT license applies to original project code and documentation. It does not replace the following third-party licenses. Names identify upstream sources and do not imply endorsement.

| Component | Source and pinned revision | Included files | License |
|---|---|---|---|
| G1 model | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` | `robots/unitree_g1/` | BSD-3-Clause; included LICENSE |
| G1 velocity policy | [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab), `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` | `policies/g1_velocity/` | Apache-2.0; included LICENSE |
| TWIST2 | [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2), `d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a` | `robots/unitree_g1_twist2/`, `policies/twist2_tracking/`, `fixtures/twist2_walk.pkl` | MIT; included LICENSE files |
| AgiBot X1 | [AgibotTech/agibot_x1_infer](https://github.com/AgibotTech/agibot_x1_infer), `9e0b818804d644fb9c9663e932dd33b03b24dfa4` | `robots/agibot_x1/` | Mulan PSL v2; included LICENSE.txt |
| MediaPipe Pose Landmarker | Google MediaPipe official model distribution | Downloaded locally by `download_model.py`, not tracked | Upstream distribution terms apply |
| Runtime libraries | MuJoCo, MediaPipe, OpenCV, NumPy, Pillow, ONNX Runtime and optional SciPy/OSQP | Installed by pip; no environment bundled | Their respective upstream licenses |

`SOURCE.json` manifests record model/policy sources, local changes and hashes where available. G1 model adaptations add scene elements, position/torque actuator configurations and neutral poses; the original MJCF and upstream copyright notices are retained. The default physical path differs from the shadow-preview provenance description: inspect the relevant scene and mode documentation.

The TWIST2 interface adapts the upstream observation convention, reference handling and joint ordering. Its numerical reference is loaded through an allow-list unpickler. The velocity controller uses existing pretrained weights; local upper-body blending and mode transitions are simulation adaptations, not a new training result.

Presentation visuals reuse project-generated robot renders and experimental plots. The presentation design was prepared with Hugo He's [ppt-master](https://github.com/hugohe3/ppt-master) (MIT); the skill itself and its runtime dependencies are not bundled in this repository. AI assistance was used during development, diagnosis and course-material preparation.

The presentation includes shapes derived from [Tabler Icons](https://github.com/tabler/tabler-icons). Its MIT notice is retained at [docs/licenses/tabler-icons-LICENSE.txt](docs/licenses/tabler-icons-LICENSE.txt).
