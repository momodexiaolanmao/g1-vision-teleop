# 回归输入

两个 `upper_reference_*.json` 是机器人电机关节目标向量，不含相机画面或人体关键点轨迹。
`twist2_walk.pkl` 是 TWIST2 上游提供的数值动作示例，来源、提交和 SHA256 见 `policies/twist2_tracking/SOURCE.json`，许可见同目录 LICENSE。仅使用项目的允许列表读取器加载该已知文件，不加载不可信 pickle。

原工作区的 `stationary_feet.json` 和现场 JSONL 含个人姿态记录，未公开。依赖这些输入的现场回归用例会明确跳过；其他合成输入、控制器和接口测试仍可运行。公开版不能完整重现私有现场采样。
