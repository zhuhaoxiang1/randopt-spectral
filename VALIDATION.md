# 本地验收记录

更新日期：2026-09-07。环境：macOS arm64、8GB 内存、Python 3.12.8、PyTorch 2.10.0、Transformers 5.14.1、NumPy 1.26.4。未安装新依赖，未下载预训练模型或数据集。

## 已执行

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest tests -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m spectral_randopt run --config configs/qwen15b-12h.json --dry-run
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m spectral_randopt smoke --output-dir runs/smoke-publish
git diff --check
git diff --exit-code HEAD -- randopt.py core data_handlers utils scripts
```

- 全部 **57 项测试通过**，耗时 23.48 秒；覆盖矩阵性质、GQA、低精度恢复、异常退出、评分、投票、数据隔离、离线 smoke 和断点续跑。
- 新增公平性覆盖：FP32/BF16 逐参数对照官方 WorkerExtension 的结果一致；采样 seed/sigma 流与官方入口一致；eager/SDPA 的不同长度 batch 生成验证；EOS 后 padding 不计生成 tokens；MLP 权重变化会阻止旧结果续跑；search-only 不生成测试预测；预算决策不受准确率变化影响。
- 1.5B 的 12h 配置验证通过：3 种方法、每种 N=100、K=1/10/20/50，选择 200／探测 32／测试 256 题；未加载真实模型。测试解析了全部 7 份真实实验配置。
- CLI smoke 使用 **171,680 参数**的随机微型 Qwen2，完成 **15 种设置、60 个候选**，选择／探测阶段实际生成 960 个 tokens。
- 所有模型参数在 smoke 后逐项精确恢复。
- 保谱方法在抽样诊断头上的最大 FP32 构造相对谱误差为 **7.4479e-7**；该统计不代表已验证所有真实 1.5B 头的数值误差。
- 全流程测试及 CLI smoke 禁止 socket 连接。
- 官方入口及 core、data_handlers、utils、scripts 没有修改。新克隆不存在指向旧仓库的 Git alternates 依赖。
- 服务器任务书中的 8 段 Bash、3 个内嵌 Python 命令通过语法检查；预算估算只有合成计时的单元验证，尚无真实 GPU 吞吐数据。
- 本次 smoke 的 manifest 代码 SHA256 与交付代码一致；已产出同 K 配对比较表。

执行中修复了一个评分边界问题：上游宽松答案提取器可能把独立句号返回为答案。扩展增加了数值校验，避免无效回答影响多数投票；回归测试已通过。

## 留存产物

- `runs/smoke-publish/smoke-verification.json`：发布前复核验收状态。
- `runs/smoke-publish/run/report.md`：发布前离线 smoke 分析报告。
- 同目录下的 `manifest.json`、`summary.csv`、`fair_comparison.csv`、`candidates.jsonl`、`hybrid_pairs.json` 和逐候选结果。

先前 `runs/smoke` 的 48 测试／14 方法验收产物及 `runs/smoke-fair-12h` 的 15 方法验收产物保留为历史记录。发布前仅清除了 `prepare.py` 文件末尾多余空行，并更新仓库交接文档；已重跑 CLI smoke，指纹与交付代码一致。新 schema 或代码不应对旧运行强行续跑。

运行产物已被 Git 忽略，源代码、配置、测试和文档保留在工作树中。

## 尚未执行

真实 Qwen2.5-1.5B-Instruct 的权重加载、GPU 前向与 GSM8K 性能实验，以及网络数据准备命令。smoke 的随机模型准确率不能用于判断谱扰动是否优于高斯扰动。48GB 单卡、最多 12 小时的 GPU 操作见 `SERVER_EXPERIMENT_PLAN.md`。
