# 安装与操作

## 环境

已验证环境为 Windows、64 位 Python 3.14。建议先安装 Python 与 Git，并确认 `python --version` 可用。项目主入口需要能显示 MuJoCo 和相机窗口的图形桌面。首次安装会联网下载 Python 依赖及官方 MediaPipe 模型。

```powershell
git clone https://github.com/momodexiaolanmao/g1-vision-teleop.git
Set-Location -LiteralPath '.\g1-vision-teleop'
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

若安装源较慢，可使用 `setup.ps1 -UseTsinghuaMirror`。只在这次脚本进程使用指定执行策略。项目依赖安装在仓库的 `.venv`，不要求修改系统 Python 环境。

默认站立 QP 使用 NumPy。`requirements-physics-extra.txt` 中的 SciPy、OSQP 是可选后端；已有部署与数值环境记录见 `requirements-lock.txt`，它不是所有平台通用的依赖保证。

## 连接相机

DJI Osmo Action 3 使用数据线接电脑，在相机上选择 Webcam。关闭占用相机的会议软件。相机固定在能同时拍到头、双手和双脚的位置，尽量保持只有操作者一人入镜。

```powershell
.\start_unitree_g1.cmd --camera 0
```

相机编号不一定是 0，可尝试 `--camera 1` 或 `--camera 2`。Windows 默认后端为 DirectShow，打不开时可尝试 `--backend msmf`。1280×720、30 FPS 是请求配置，不是端到端延迟测量。

## 无需跑回电脑的标定

程序给出 8 秒就位时间。进入镜头后双脚落地、直立、手臂自然放松，完整且稳定入镜约 1.5 秒后自动标定。按画面提示缓慢摆臂或交替提膝。C 重新标定，N 重选人物后再按 C，空格暂停，R 重置，Q／Esc 退出。

迈步需要同时满足画面足部抬高、膝弯曲和持续时间条件。低可信、人物丢失和暂停会取消迈步请求，控制器继续处理平衡。机器人同侧脚相位由策略决定。

## 选择运行模式

| 入口 | 用途 |
|---|---|
| `start_unitree_g1.cmd` | 默认物理迈步跟随 |
| `start_unitree_g1_physics.cmd` | 固定双脚的上身跟随 |
| `start_unitree_g1_shadow.cmd` | 全身姿态映射预览 |
| `start_unitree_g1_tracking.cmd` | TWIST2 实验路径，曾在真人输入中跌倒 |
| `start_demo.cmd` | 简化机械臂合成输入演示 |
| `start_joint_control.cmd` | 简化机械臂自定义关节映射 |
| `start_agibot_x1.cmd` | X1 双臂实验 |

## 修改对应关系

默认 G1 迈步模式使用 `unitree_g1_walking_mapping.json`，编辑页面是 `unitree_g1_walking_mapping_editor.html`。双击 HTML 在本机浏览器打开，修改通道、关节、方向、角度范围、偏置或最大速度后导出配置，替换相应 JSON 并重启程序。编辑器不向服务器上传相机数据。

迈步时腿部由策略接管，编辑人体到腿部的参考不能强制指定机器人脚相位。用影子模式检查全身对应关系，用物理模式验证可执行性。

## 自动采样

`check_unitree_g1_static.cmd`：自动标定、准备 3 秒、静止采样 20 秒。

`check_unitree_g1_motion.cmd`：自动标定、准备 3 秒、动作采样 25 秒。

直到看到“采样结束，可以活动了”并听到结束提示，再返回电脑。结果在 `artifacts/`，结束提示只表示数据保存完成，不能自动证明测试合格。相机采集和日志默认保留在本机。

更多参数和历史操作说明见根目录 [README_zh.md](../README_zh.md)、[G1_STEPPING_zh.md](../G1_STEPPING_zh.md)。
