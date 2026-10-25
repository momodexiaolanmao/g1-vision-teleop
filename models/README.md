# MediaPipe 模型（本机下载）

运行 `setup.ps1` 自动下载官方 Lite 与 Full Pose Landmarker bundle；也可运行：

```powershell
.\.venv\Scripts\python.exe download_model.py --variant lite
.\.venv\Scripts\python.exe download_model.py --variant full
```

权重保存在此目录，由 `.gitignore` 排除。默认 G1 使用 Full；模型来源 URL 在 `download_model.py`，下载后会打印 SHA256。仓库中的 ONNX 步态策略在 `policies/`，与人体姿态估计权重不同。
