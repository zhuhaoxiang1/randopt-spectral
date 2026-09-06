# RandOpt

## 本地谱扰动实验扩展

本研究仓库：[zhuhaoxiang1/randopt-spectral](https://github.com/zhuhaoxiang1/randopt-spectral)（公开，默认分支 `main`）。
基于 [sunrainyg/RandOpt](https://github.com/sunrainyg/RandOpt)，保留上游代码与历史。

新增的 CPU / 单 GPU 实验入口、Qwen2.5-1.5B-Instruct 配置和离线 smoke 见
[SPECTRAL_EXPERIMENTS.md](SPECTRAL_EXPERIMENTS.md)。官方 Ray/vLLM 入口保留原样。

**48GB 单卡、最多 12 小时的服务器任务：** 按 [SERVER_EXPERIMENT_PLAN.md](SERVER_EXPERIMENT_PLAN.md) 执行。
默认 N=100、K=50，公平比较原始 RandOpt 扰动规则、同范围高斯与 U/V 谱旋转；先实测吞吐再确定共同规模。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m spectral_randopt smoke --output-dir runs/smoke
```

此命令使用随机初始化的微型 Qwen2，不下载模型、数据或依赖。

<p align="center">
  <img src="assets/neural_thickets.gif" alt="Neural Thickets" width="100%">
</p>

**Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights**

[Yulu Gan](https://yulugan.com), [Phillip Isola](https://web.mit.edu/phillipi/)

[Paper](https://arxiv.org/pdf/2603.12228)          |         [Project Page](https://thickets.mit.edu)    |   [Openreview](https://openreview.net/forum?id=92oF5bU4cU) |    Starting with a 1D Experiment: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SsBrfQ-iFKuGElWjTNiFoX4dtMaCzCGy?usp=sharing)


## News
- **[2026-07]** **Iterative RandOpt** is now on the [**`iterative-randopt`**](https://github.com/sunrainyg/RandOpt/tree/iterative-randopt) branch! Also support the `pip install`-able implementation that drops into existing **verl** or **huggingface/trl** setups.


## Requirements

### Option1: Python / Conda
```bash
(optional) conda activate your_env
pip install -r requirements.txt
```

### Option2: Docker

From the directory containing `RandOpt/`:

| Step | Command |
|------|---------|
| **Build** | `docker build -f RandOpt/docker/Dockerfile_vllm -t randopt-vllm:latest .` |
| **Run** | `docker run -it --gpus all randopt-vllm:latest bash` |
| **Run** (with data) | `docker run -it --gpus all -v /path/to/RandOpt/data:/workspace/data randopt-vllm:latest bash` |


## Run RandOpt

### Post-train on your own dataset
Please follow the instructions in [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md)

### Post-train on a standard dataset
First download the data here: [data/README.md](data/README.md)

Then, from the `RandOpt` directory:

| Mode | Command |
|------|---------|
| **Single node** | `sbatch scripts/single_node.sh` |
| **Multiple nodes** | `sbatch scripts/multiple_nodes.sh` |
| **Local** (no Slurm) | `bash scripts/local_run.sh` |

## Distill top-k models into a single model
Please follow the instructions in [distillation/README.md](distillation/README.md).

## Run Baselines
Please follow the instructions in [baselines/README.md](baselines/README.md)

## Having questions?
Open an issue @ [github.com/sunrainyg/RandOpt/issues](https://github.com/sunrainyg/RandOpt/issues/new).


## Citation
```bib
@misc{gan2026neuralthickets,
      title={Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights}, 
      author={Yulu Gan and Phillip Isola},
      year={2026},
      eprint={2603.12228},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.12228}, 
}
```
