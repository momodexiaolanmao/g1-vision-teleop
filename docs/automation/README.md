# 可选 GitHub Actions 模板

`regression.yml.example` 是 Windows / Python 3.14 的快速回归工作流。已有本地快速与完整测试入口，不依赖 GitHub Actions 即可验证。

需要启用时，将模板复制为仓库的 `.github/workflows/regression.yml`，由有工作流写入权限的维护者提交。首次公开不为自动化额外扩大账号授权。

工作流只申请仓库内容读取权限，执行 `scripts/run_checks.py` 并上传数值测试结果，不采集相机。
