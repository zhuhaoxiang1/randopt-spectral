# 给服务器 AI：48GB 单卡、最多 12 小时的公平比较

## 1. 本轮任务与固定决策

在用户租用的一张标称 48GB 的 4090 上，比较 **Qwen/Qwen2.5-1.5B-Instruct + GSM8K**。总租用时间最多 **12 小时**，包括环境、下载、短测、实验和整理。只在服务器下载缺失内容，优先复用缓存。不要在用户 Mac 上安装依赖或下载模型、数据。

默认 **每方法 N=100，主终点 K=50**；同时报告 K=1、10、20。三个方法、一个搜索种子 42。先完成三组公平比较；本轮不要求运行原有 full、bands、angles、complement 配置，不启动多种子大扫参。

| 比较组 | 扰动定义 | 本轮作用 |
|---|---|---|
| `original_randopt` | 全部文本参数加原生精度高斯噪声，保留官方每参数重置同一 seed 的规则 | 与原始算法比较 |
| `gaussian_matched` | 所有注意力 Q/K/V/O 的真实头块，加高斯噪声并匹配目标相对距离 | 控制参数范围、扰动距离 |
| `spectral_uv` | 相同头块，SVD 后小角度旋转 U/V，固定奇异值；随机正负角度 | 本轮预先选定的新方法 |

不能在看过测试结果后，从 U-only、V-only、UV、不同谱带中挑一个最好者叫“我们的方法”。其他变体已经实现，留作后续机制实验。

## 2. 为什么选择这个 N 和 K

根据 [Neural Thickets 原文](https://arxiv.org/html/2603.12228v1) 的具体位置区分设置：

- **附录 E、E.3 / Table 3**：主体实现写的是 N=5000、K=50，使用前 200 条训练数据、完整测试集；sigma 档位为 0.001、0.002、0.003，BF16，GH200 硬件。Fig. 8 的模型规模实验另用 N=3000、K=50，不能混为一个统一 N。
- **Fig. 11 / 附录 D**：Qwen2.5-1.5B-Instruct 的 GSM8K 结果中，RandOpt K=1 为 67.9%，K=50 为 76.4%。所以只用原先的 K=1/5 会漏掉重要的集成效果。
- **Sec. 6 / Fig. 7**：N 与 K/N 的热图来自 **Qwen2.5-3B-Instruct、Countdown**；研究了 N=10 到 100000、比例 1% 到 100%。大 N 下较低选择比例更有利。它不证明 1.5B GSM8K 的最优设置，也不支持把大 N 的最优比例直接搬到 N=100。

因此，**K=50 有论文依据；N=100 是 12 小时下的工程取舍，不是声称论文最优**。N=100 时 K/N=50%，保留 K=1/10/20 作描述性比较。K 曲线复用已选 Top-50 的测试预测，避免分别重跑。

三组各 100 个候选，合计 300 个。N 已包含三个强度档的抽样，不是每档再跑 100 个。候选逐个加载、恢复，只保存种子和输出；N/K 不要求同时在显存里保存 100/50 个模型。48GB 主要用于模型、一个基础快照、KV cache 和批量生成；不能用“显存放得下”代替耗时测算。

## 3. 公平性约束与解释边界

1. **共同评估协议**：同一模型 snapshot、BF16、Transformers 后端、SDPA、batch=16、chat template、题目顺序、答案提取、多数投票和生成上限。均为贪心生成，每题每专家一条回答。原始基线和新方法比较时始终使用同一个 K。
2. **共同搜索预算**：官方 NumPy 规则抽取 N 个不重复 seed，再独立抽强度档；三组共享候选 ID、seed、档位索引。不启用只对新方法有利的 antithetic 配对，不额外给新方法筛种子或重试失败候选。
3. **原始范围确实保留**：`original_randopt` 改动 embeddings、MLP、norm、bias 和注意力等全部文本参数，不能用 attention-only Gaussian 冒充它。参考代码：[官方 worker](https://github.com/sunrainyg/RandOpt/blob/536df0a308f3990b6270c991fbb96bd0b779a58e/utils/worker_extn.py)、[官方采样入口](https://github.com/sunrainyg/RandOpt/blob/536df0a308f3990b6270c991fbb96bd0b779a58e/randopt.py)。
4. **准确命名复现程度**：本工程把原始扰动规则适配到共同 Transformers 参数布局，且全部方法都从精确快照恢复，消除重复 BF16 加减的漂移。官方 vLLM 有融合参数布局差异；当前比较不是其逐比特复现，也不是 N=5000 的论文数值复现。原始入口保留未改，但本轮不另跑不同后端来混算结果。
5. **sigma 与 rho 不等同**：原始 sigma={0.001,0.002,0.003} 是逐元素绝对标准差；另外两组 rho={0.001,0.002,0.003} 是每头相对 Frobenius 距离。后者是预设的小扰动探索范围，未证明最优。三组各有三档、相同总评估次数。算法级比较允许作用范围不同；只有 `gaussian_matched` vs `spectral_uv` 用于相同范围／半径的机制判断。
6. **数据隔离**：官方 train 固定打乱后取 200 条选择集，另取 32 条探测集；与官方 test 不交叉。Top-K 只按选择集排序，平分按候选 ID。测试默认固定 256 条，绝不根据测试准确率选方法、强度、N/K、精度或生成长度。这里使用固定打乱的 200 条，和论文“前 200 条”有区别，应披露。
7. **计算开销真实报告**：相同 N/K/题目数/生成上限控制评估预算，不等于相同实际 FLOPs 或时间。记录生成 tokens、SVD、参数变换、诊断、搜索、测试、总墙钟时间。当前多了探测和机制诊断，不能把它们隐去后声称端到端更省时。

配置最多生成 **1024 个新 tokens**，并单独限制 prompt≤1024；这不等同于论文的 max sequence length=1024。三组保持一致，明确报告截断率。主结果只对应这套预算、数据子集与单种子，不能直接与论文的 76.4% 比高低。

## 4. 收到工程与设置总截止时间

本扩展位于公开仓库 [zhuhaoxiang1/randopt-spectral](https://github.com/zhuhaoxiang1/randopt-spectral)，默认分支 `main`。基础提交为 `536df0a308f3990b6270c991fbb96bd0b779a58e`，本地开发分支 `spectral-perturbations` 推送到远端 `main`。**仅 clone 官方 sunrainyg/RandOpt 仓库拿不到本扩展。**

服务器可直接匿名克隆，无需 GitHub 登录或访问令牌：

```bash
git clone https://github.com/zhuhaoxiang1/randopt-spectral.git
cd randopt-spectral
```

以下均从仓库根目录、在服务器 Bash 中执行。先检查现有文件和环境，不重置仓库、不覆盖旧结果。

```bash
test -f spectral_randopt/original.py
test -f spectral_randopt/budget.py
test -f configs/qwen15b-12h.json
git status --short
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python3 --version
```

以**开始计费时刻**设置截止时间；若已消耗时间，扣除后填入实际剩余秒数。下面的 43200 仅适用于刚开始计费。不要在换终端或续跑时重新赠送 12 小时。

```bash
export RANDOPT_DEADLINE_UNIX=$(python3 -c 'import time; print(int(time.time()) + 43200)')
mkdir -p runs
printf '%s\n' "$RANDOPT_DEADLINE_UNIX" > runs/rental_deadline_unix.txt

# 后续同一 Bash 会话复用此函数。GNU timeout 通常由服务器 coreutils 提供。
run_before_deadline() {
  local remaining
  remaining=$(python3 -c 'import os,time; print(int(os.environ["RANDOPT_DEADLINE_UNIX"])-int(time.time())-600)')
  if [ "$remaining" -le 0 ]; then
    printf '%s\n' '已进入最后 10 分钟，停止启动任务。' >&2
    return 124
  fi
  timeout --signal=INT --kill-after=60s "${remaining}s" "$@"
}
```

最后 10 分钟留作整理、备份和按租用平台流程释放实例。`timeout` 只停止本实验进程，不会自动终止云实例计费；服务器 AI 必须遵守已授权的租用截止安排，不自动续费。若平台支持定时关机，设置在既定截止时刻。

## 5. 准备与短测：预计预算 0–1.5 小时，实际超出就扣减主实验

优先复用已有环境；缺依赖时只在服务器实验环境安装。记录 torch/transformers 版本和 CUDA/BF16 支持，不打印凭据或整个环境变量列表。新入口不需要 vLLM/Ray。

```bash
# 只在确实缺依赖时运行。
run_before_deadline python3 -m pip install -r requirements-spectral.txt

run_before_deadline env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest tests -q
python3 -m spectral_randopt run --config configs/qwen15b-12h.json --dry-run
```

普通 `run` 只加载本地权重，不会自动下载。以下三条是**替代选项，只运行必要的一条**：

```bash
# 模型、数据均缺失。
run_before_deadline python3 -m spectral_randopt prepare --output-dir data/spectral
# 只有数据缺失。
run_before_deadline python3 -m spectral_randopt prepare --data-only --output-dir data/spectral
# 只有模型缺失。
run_before_deadline python3 -m spectral_randopt prepare --model-only
```

已有本地模型目录时，统一修改两份 12h JSON 的 `model`。固定 revision 时两份也保持一致；不要只给短测传 `--model` 导致它与主实验配置不同。数据准备目录有部分旧文件时先核实来源，准备命令不自动覆盖 split。未拿到数据／模型就明确报告，别换任务或随机模型来当真实实验。

短测仍加载完整 1.5B、修改全部既定目标头；每方法 3 个候选、选择 32 题、探测 8 题，使用与主实验相同的 batch 和生成长度。**只运行 search-only，完全不生成测试集预测。** 它既验证真实 GPU 路径，也提供预算估计所需耗时。

```bash
run_before_deadline python3 -m spectral_randopt run --config configs/qwen15b-12h-calibration.json --search-only
```

检查 `runs/server-12h-calibration/search_complete.json`、`manifest.json`、`timing.json` 与候选记录：

- `smoke_only=false`；实际模型为预训练 1.5B，三组各 3 个候选完成，无 CUDA/SVD/NaN/全零扰动错误。
- `original_baseline` 作用于全部文本参数；另外两组 targets、rho 相同。查看两组 BF16 **applied** 的实际距离分布，不只看 FP32 construction。
- 抽样谱旋转 FP32 构造相对谱误差应在数值误差范围，超过 `1e-5` 先排查。BF16 后误差单独报告。该阈值是排错门槛，不是研究结论。
- 检查选择／探测回答、答案提取和截断率。若大量截断（排查门槛 10%），记录原因；需要加长时，三组和两份配置统一修改并重新计时。不得缩短原始基线回答来追求更高吞吐。
- 若匹配半径的实际距离偏离目标超过 20%，或 Gaussian / UV 的实际距离明显失配，先排查舍入；必要时统一做 FP32 短测。精度改动需三组共用并重新预算，不能把数值失配解释成谱方法优势。

batch=16 是起点。OOM 先统一降至 8；仍不够再降至 4，两份配置一起改，用新的短测输出目录计时。不要默默缩少层／头。吞吐充裕时可在正式实验前统一测 batch=32，只根据内存和速度决定，不看测试集。预算不足时保留明确限制，而不是承诺一定跑完。

## 6. 只用耗时确定共同规模

默认主配置：200 选择题、32 探测题、256 测试题、N=100、K=50。三组选择／探测需要 69600 条生成，测试需 38400 条，加基础模型 488 条，共 **108488 条**，还不含参考 KL 前向。即便 1.5B 能放进显存，也必须计时。

规划器使用短测中最慢的每题生成时间、参数变换／恢复和参考诊断耗时，加冷启动开销，再乘 **1.5 安全系数**。它只读取耗时和样本数作决策，不用准确率。预先固定的降档顺序：

| 优先级 | N | K 主终点 | 测试题数 | 说明 |
|---|---:|---:|---:|---|
| 1 | 100 | 50 | 256 | 默认 |
| 2 | 100 | 50 | 128 | 保留搜索预算，降低测试成本 |
| 3 | 50 | 50 | 128 | 三组同步降档；K=50 相当于集成所有候选，必须标注 |

如果第三档仍超预算，规划器拒绝生成可运行配置。不要擅自把选择集缩到个位数或只跑新方法；提交短测与不可行的耗时证据。K=1/10/20 在三档中均保留。

```bash
# 从实际总截止时间扣除 1 小时收尾／意外余量，主实验最多再分配 9.5 小时。
RANDOPT_MAIN_HOURS=$(python3 -c 'import os,time; print(min(9.5,(int(os.environ["RANDOPT_DEADLINE_UNIX"])-time.time()-3600)/3600))')

python3 -m spectral_randopt plan-budget --calibration-dir runs/server-12h-calibration --config configs/qwen15b-12h.json --output-config configs/server/qwen15b-12h-measured.json --hours "$RANDOPT_MAIN_HOURS" --safety-factor 1.5
```

保留生成的 `.budget.json` 和配置。估计不构成完工保证，正式运行仍受总截止时间约束。候选 seed 流依赖 N，改变 N 需新配置、新目录、新完整运行；**不要把 N=100 的前 50 个候选称为官方 N=50 采样复现，也不要修改 manifest 伪造续跑**。

## 7. 主实验：先冻结搜索，再统一测试

```bash
run_before_deadline python3 -m spectral_randopt run --config configs/server/qwen15b-12h-measured.json --search-only

# 仅当搜索完整、三组规模相同，且剩余预算仍足够完成三组测试时执行。
run_before_deadline python3 -m spectral_randopt run --config configs/server/qwen15b-12h-measured.json --resume

python3 -m spectral_randopt analyze --run-dir runs/server-12h-s42
```

候选按 ID 在三组之间交错执行，避免先把预算花完在单个方法上。每个候选从精确快照开始；结果原子落盘。中断后只在代码／配置／模型／数据／软件身份均不变时加 `--resume`。代码修复或精度／batch／长度变化要求新配置和新目录，不混合成一个完成实验。

进入测试前根据已产生的 timing 和剩余时间再估算所有三组的 Top-50 成本。如果实际吞吐比短测差很多，保留完整搜索结果并标记测试未完成；不要仅测完最好看的方法。`complete.json` 不存在时，不能声称完成公平测试比较。

本轮没有预授权的额外租用时间。即使提前完成，也优先核验、备份和写报告；N=300/1000/5000、多种子以及全量 1319 题实验留到下一轮，不根据本次测试结果决定是否追加有利实验。

## 8. 必须交付的结果

新建 `results/server_experiments/SERVER_RESULTS.md`，用中文说明实际做完了什么，保留 `manifest.json`、代码 diff、实测配置、预算报告、环境与计时日志、全部候选记录和选中专家 ID。不要只交一张最好看的图。

主表至少包含：基础模型与三组的 K=1/10/20/50 准确率；K=50 的 UV−原始、UV−matched Gaussian 两个预定差值；配对题目 bootstrap 95% 区间；独立探测命中率；实际扰动距离与作用域；参考 KL/NLL；截断率；实际生成 tokens、搜索与测试耗时。

现成产物：

- `summary.csv`：准确率、命中率、距离作用域、参考损失、搜索／测试 tokens 和耗时。
- `fair_comparison.csv`：相同 K、相同测试题上的配对差值、区间和双方独有正确题数。
- `report.md`：基础结果与解释边界；`ensembles/*.json` 保存 K 曲线、纠错／退化及正确性分歧。
- `candidates/*/*.json`：构造前后谱诊断、各阶段计时及逐题输出。原始基线的距离分母是全参数，另两组是目标头；不要画成统一尺度却不标注。

报告明确：这是一轮 **单种子、小预算、测试子集** 的探索性受控比较。题目 bootstrap 条件于本次已选专家，不涵盖搜索种子不确定性，也不校正查看多条 K 曲线后的选择偏差。命中率区间按强度档分层、固定本次各档计数，不涵盖档位比例的抽样波动。32 题探测集步长为 3.125 个百分点；全零命中的退化区间不证明真实命中概率为零。

判断方式：UV 胜过原始但未胜过同范围高斯，不能归因于谱结构；UV 胜过同范围高斯但未胜过原始，说明有机制线索却尚无整体优势；仅集成提升不能证明单专家更强。即使两个比较均为正，也需后续固定配置、多种子、完整测试集和无谱引导正交对照才能支持更强结论。

本地只完成代码与禁止联网的随机小模型 smoke。服务器 AI 应自行报告真实 CUDA 验证结果；不填造 1.5B 准确率，不隐去失败候选，不把未完成阶段写成完成。
