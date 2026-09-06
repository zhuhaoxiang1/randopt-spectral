# Spectral RandOpt：机制验证与 Qwen 1.5B 实验

本扩展实现 Neural Thickets × Diffract 的首轮实验：比较高斯候选、奇异方向／幅值交叉重构、小角度正交旋转和谱带消融。**算法是否改善真实任务性能，需要在预训练模型上实测；离线 smoke 不提供这种证据。**

**当前租卡预算为 48GB 单卡、最多 12 小时。优先执行 [SERVER_EXPERIMENT_PLAN.md](SERVER_EXPERIMENT_PLAN.md)，使用 `qwen15b-12h*.json`。** 下文 pilot/full/bands 等是研究扩展配置，不是本轮要求全部运行的任务清单。

## 本地：不下载模型、数据或依赖

从仓库根目录运行，使用已经安装的 Python 包：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m pytest tests -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m spectral_randopt smoke --output-dir runs/smoke
python3 -m spectral_randopt run --config configs/qwen15b-pilot.json --dry-run
```

Smoke 在内存中构建随机初始化的 2 层 Qwen2：hidden size 96、6 个 Q heads、2 个 KV heads。覆盖全部 15 种方法设置、60 个候选，包括原始 RandOpt 噪声规则、批量生成、评分、选择、投票、参考文本 KL、谱诊断与落盘；结束时逐参数检查精确恢复。整个过程禁止 socket 连接。随机模型答不对算术题属于预期，不能把 smoke 的命中率用于研究结论。

`runs/smoke/smoke-verification.json` 保存验收结果；`runs/smoke/run/report.md` 保存分析报告。重复运行请选择新输出目录，或加 `--resume`。测试另覆盖异常中断、种子重放和断点续跑。

`--dry-run` 只解析配置与候选预算，既不加载模型，也不要求本地存在实验数据。它不代表真实 1.5B 已通过前向测试。

## GPU 主机：准备真实模型和数据

以下命令只应在将要执行真实实验的机器上运行。若缺依赖，可在那里使用 `requirements-spectral.txt` 建环境；本扩展不需要安装官方入口依赖的 vLLM、Ray、datasets 或 pyarrow。

```bash
# 显式下载 Qwen2.5-1.5B-Instruct 的 safetensors、配置、tokenizer，以及官方 GSM8K JSONL。
python3 -m spectral_randopt prepare --output-dir data/spectral

# 已经有模型时，只准备数据；或用 --model-only 仅准备模型。
python3 -m spectral_randopt prepare --data-only --output-dir data/spectral
```

上面两条准备命令是替代关系，不应对同一个已有数据目录连续执行。准备步骤记录原始数据 SHA256；参考集为四段固定普通英文，只用于局部 KL/NLL 诊断，不是通用能力基准。模型下载使用 safetensors，不重复下载 PyTorch `.bin` 权重。

普通 `run` 始终以 `local_files_only=True` 加载。已有模型可用 `--model /path/to/model` 指定；未准备好模型或数据时明确失败，不会偷偷下载。`prepare --revision <commit>` 可以固定模型版本；配置中的 revision 也应同步设置。实际解析到的模型 commit 会写入运行清单。

## 初步实验与扩展实验

```bash
python3 -m spectral_randopt run --config configs/qwen15b-pilot.json
python3 -m spectral_randopt analyze --run-dir runs/qwen15b-pilot

# 相同代码、配置、模型和数据下中断续跑。
python3 -m spectral_randopt run --config configs/qwen15b-pilot.json --resume

# 扩展实验分别运行 3 个种子；先看初步实验的耗时、截断率和指标，再运行这些命令。
python3 -m spectral_randopt run --config configs/qwen15b-full.json --seed 42 --output-dir runs/qwen15b-full-s42
python3 -m spectral_randopt run --config configs/qwen15b-full.json --seed 43 --output-dir runs/qwen15b-full-s43
python3 -m spectral_randopt run --config configs/qwen15b-full.json --seed 44 --output-dir runs/qwen15b-full-s44
```

| 配置 | 选择集 | 独立探测集 | 测试集 | 每方法候选数 | K | 最多生成 tokens |
|---|---:|---:|---:|---:|---|---:|
| `qwen15b-pilot.json` | 64 | 32 | 128 | 32 | 1、5 | 512 |
| `qwen15b-full.json` | 256 | 64 | 完整 test | 256 | 1、5、10 | 1024 |

两档均默认 BF16、单 CUDA 设备、逐样本确定性生成、相对半径 0.001／0.003、随机正负角度、每侧 4 个旋转平面。全部注意力 Q/K/V/O 参与，其他参数冻结。SVD/旋转使用 FP32；诊断记录转换回模型精度前后的谱误差。原版训练集与测试集的来源保持分开，选择集和探测集从官方 train 中按固定种子划成不相交子集。

所有候选只在选择集和探测集上评估，Top-K **只根据选择集宽松答案准确率**确定，分数相同按候选 ID 排序。随后仅对 Top-K 做完整测试集生成。独立探测集用于估计候选命中率和配对干预效果，不参与排序。这样无需为每个候选都遍历完整 GSM8K test。

### 单独运行消融

```bash
python3 -m spectral_randopt run --config configs/qwen15b-bands.json
python3 -m spectral_randopt run --config configs/qwen15b-angles.json
python3 -m spectral_randopt run --config configs/qwen15b-complement.json
```

- `bands`：无谱引导正交对照、全谱、前 25%／中间 50%／后 25% 的带内旋转、跨带旋转。
- `angles`：随机正负、有界均匀、高斯角度系数，保持相同距离预算和参数范围。角度系数归一化后统一校准距离，所以比较的是方向分布，不能解释成高斯径向尾部实验。
- `complement`：在长方形矩阵的长边上将奇异向量旋向正交补；与仅在已有奇异子空间内旋转对比。

修改配置的 `layers` 可限制层，`projections` 可限制 Q/K/V/O；结构方法使用相同目标掩码，`original_randopt` 始终保持全部文本参数范围。`head_indices` 指每个投影中的本地块编号，K/V 只包含实际 KV heads，不把共享 KV 头复制成多个独立参数。它不表示一个可独立修改所有关联 KV 的逻辑 query-head 组。

配置中数据和输出路径相对于配置文件所在目录解析；命令行 `--output-dir` 相对于当前目录解析。模型 ID 不按文件路径转换。

## 方法与公平比较

`original_randopt` 是全部文本参数上的原始噪声规则，包含 embeddings、norm、bias；每个参数重新用同一候选 seed 生成原生精度噪声，与上游 `utils/worker_extn.py` 有直接一致性测试。所有组共用 Transformers 推理后端和精确快照恢复。vLLM 的融合参数布局不同，因此该基线应称为 **原始 RandOpt 算法的共同后端适配**，不能声称是官方 vLLM 位级或论文全预算复现。原始入口仍可单独运行，但其结果不能直接作为不同后端的受控比较。

12h 配置使用官方 unique-seed / IID sigma 采样规则，禁用正负配对；三个方法共享候选 ID/seed/强度档索引和相同 N/K。原始 sigma 是绝对标准差，谱方法 rho 是每头相对距离，数值相同不表示尺度相同。`gaussian_matched` 才是谱方法的同掩码、同目标 rho 机制对照；BF16 后的实际距离另外记录。主结果报告 K=50，K=1/10/20 为复用输出的描述性曲线。

`run --search-only` 只做选择／探测和冻结 Top-K，不生成测试预测；同配置 `--resume` 完成测试。`plan-budget` 从真实短测的耗时选择预先定义的共同规模，不读取准确率作决策。`summary.csv` 包含实际生成 tokens、搜索／测试耗时与距离作用域；`fair_comparison.csv` 给出同 K、配对题目的准确率差及 bootstrap 区间。单种子区间不包含搜索种子不确定性。

设某个目标头的基础矩阵为 `W0 = U0 diag(s0) V0.T`，匹配高斯候选为 `Wg = W0 + E = U1 diag(s1) V1.T`。

| kind | 实现 |
|---|---|
| `original` | 全部文本参数，原生精度逐元素绝对 sigma；每个参数重置同一候选 seed |
| `gaussian` | 每个头的 FP32 独立高斯方向，缩放到 `||E||F / ||W0||F = rho` |
| `direction` | `U1 diag(s0) V1.T`，使用同一高斯来源的方向 |
| `values` | `U0 diag(s1) V0.T`，使用同一高斯来源的幅值 |
| `orthogonal` | 在原始坐标中对行／列做小角度 Givens 旋转 |
| `spectral` | 在 U／V 的列坐标中做 Givens 旋转，Sigma 固定 |
| `complement` | 在长边奇异子空间与其正交补之间旋转，Sigma 固定 |

Method 配置字段为 `name, kind, side, band, pairing, distribution, pairs, max_angle`。`side` 为 `u/v/both`；`distribution` 为 `sign/uniform/gaussian`；`pairing` 为 `within/cross`。`band=all, pairing=within` 允许全谱内任意向量对；特定谱带限制在该分组内；`pairing=cross` 只允许不同分组之间的向量对。

所有方法共享候选描述 `(seed, rho, sign, pair_id, sigma)`；`sigma` 只用于原始基线，balanced 采样时为空。结构方法的每个目标头根据参数名与块编号派生随机种子，保证高斯、方向、幅值来自同一 E；原始方法按官方规则直接重用候选 seed。研究扩展配置的正负配对计入 N；对非线性旋转，这是角度符号配对，有限幅度的权重差不必严格互为相反数。12h 配置不启用该配对。

旋转通过搜索角度匹配权重距离，不直接缩放 `W' - W0`。角度上限默认 0.2 弧度，达不到目标距离时明确报错，建议减小 rho 或增加旋转平面数量。交叉重构不再缩放；实际距离与原高斯候选可能不同，分析中必须保留这个区别。

其中 `gaussian` / `gaussian_matched` 按头匹配距离，使用不同参数的独立随机流，承担结构对照；`original_randopt` 才承担原始算法对照。原版 `randopt.py`、`core/`、`utils/worker_extn.py` 保留，可在具备官方依赖的 GPU 环境中另作参考；不能将不同参数范围和后端的差异直接归因于保谱。

保谱针对单个目标头矩阵；不保证拼接后的整层谱、attention 输出或整个模型的函数不变。U/V 子空间内旋转和正交补旋转是不同实验，不应混称为完整奇异子空间搜索。

## 输出与分析

- `manifest.json`：配置、软件版本、代码指纹、模型 revision、数据 SHA256、各数据划分 ID、目标矩阵指纹和全部候选种子。
- `candidates/<method>/<id>.json`：选择与探测逐题答案、正确性、参考 KL/NLL、权重距离、采样耗时，以及抽样头的详细谱诊断。
- `selected/` 与 `test/`：按选择集排名的候选 ID，以及这些候选的测试预测。
- `ensembles/`：各 K 的准确率、纠正基础错误／破坏基础正确的比例，以及成员正确性分歧。
- `candidates.jsonl`、`summary.csv`、`fair_comparison.csv`、`hybrid_pairs.json`、`report.md`：汇总、同 K 配对比较与机制分析。

默认不保存完整生成文本以控制结果大小；可设置 `save_texts=true`。始终记录严格／宽松提取答案与生成长度。多数投票忽略无效数值回答，平票选择排名更靠前的成员；不会查看 ground truth 来解平票。上游宽松提取器可能返回单独的句号，扩展在其后做 Decimal 数值校验和规范化，防止无效回答参与投票。

SVD 缓存使用不含 pickle 的 NPZ，绑定模型、参数名和实际基础权重指纹；内存上限由 `svd_cache_entries` 控制，默认 16，12h 配置为 1024，减少重复磁盘读取。每个候选结束或异常时恢复基础权重的原生 dtype 副本。原始基线额外记录全部文本参数的 SHA256；断点续跑要求代码、配置、模型、数据、设备和软件版本一致，修改算法后必须新建运行目录。

诊断默认在沿层分布的 8 个目标块上做完整 SVD 对比，12h 配置使用 4 个；结构方法的所有块都记录实际距离汇总。近重根跨越分位边界时合并分组；空谱带不会静默替换成其他带。报告同时包含子空间距离和谱带重构变化，避免仅凭单个奇异向量的基变化判断语义适配。

命中率 bootstrap 按强度档分层，正负候选对作为一个簇；独立采样时每个候选各自成簇。该区间条件于固定探测题与本次各强度档的候选计数，不覆盖题目抽样和档位比例的波动；全零／全一命中时会退化，不能当作真实概率等于 0/1 的证据。小探测集准确率离散，应报告样本量，并在更大数据上复核。

本版只实现 GSM8K，不据此声称存在跨任务专家；未实现自适应头／谱带搜索、KL 等距离搜索、跨预训练检查点实验、vLLM 谱扰动 worker 或蒸馏。

## 来源与复现

- 本研究仓库：[zhuhaoxiang1/randopt-spectral](https://github.com/zhuhaoxiang1/randopt-spectral)，私有仓库，默认分支 `main`。服务器需具备读取权限；可用 `gh repo clone zhuhaoxiang1/randopt-spectral` 获取。
- 官方 RandOpt：<https://github.com/sunrainyg/RandOpt>，基础提交 `536df0a308f3990b6270c991fbb96bd0b779a58e`。
- Neural Thickets：<https://arxiv.org/abs/2603.12228>；Diffract：<https://arxiv.org/abs/2608.10850>。
- 数据来源：OpenAI `grade-school-math` 官方仓库的 `grade_school_math/data/{train,test}.jsonl`。

为节省本地流量，本仓库由已存在且经 GitHub 核验与官方 main 一致的 Git 对象独立克隆。发布后 `origin` 指向用户研究仓库，官方地址保留为 `upstream`。旧仓库中的未跟踪实验文件没有复制。本地使用稀疏检出保留根文件、core、data_handlers、utils、scripts 及新增实验目录；新克隆没有通过 alternates 依赖旧仓库。服务器常规 clone 会检出完整已提交工程。
