# 基于扩散的 3D 场景生成、优化与规划

<a href="./README.md">English</a> | 中文

<p align="left">
    <a href='https://scenediffuser.github.io/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='论文 PDF'>
    </a>
    <a href='https://arxiv.org/abs/2301.06015'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='论文 arXiv'>
    </a>
    <a href='https://scenediffuser.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='项目页面'>
    </a>
    <a href='https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo'>
      <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggingFace'>
    </a>
    <a href='https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='检查点'>
    </a>
</p>

[Siyuan Huang*](https://siyuanhuang.com/),
[Zan Wang*](https://silvester.wang),
[Puhao Li](https://xiaoyao-li.github.io/),
[Baoxiong Jia](https://buzz-beater.github.io/),
[Tengyu Liu](http://tengyu.ai/),
[Yixin Zhu](https://yzhu.io/),
[Wei Liang](https://liangwei-bit.github.io/web/),
[Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/)

本仓库是论文 "Diffusion-based Generation, Optimization, and Planning in 3D Scenes" 的官方实现。

我们提出了 SceneDiffuser，一个用于 3D 场景理解的条件生成模型。SceneDiffuser 提供了一个统一模型，用于解决场景条件下的生成、优化和规划问题。与以往工作相比，SceneDiffuser 天然具备场景感知、基于物理和面向目标的特性。

[论文](https://scenediffuser.github.io/paper.pdf) |
[arXiv](https://arxiv.org/abs/2301.06015) |
[项目](https://scenediffuser.github.io/) |
[HuggingFace 演示](https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo) |
[检查点](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing)

<div align=center>
<img src='./figures/teaser.png' width=60%>
</div>

## 摘要

我们提出了 SceneDiffuser，一个用于 3D 场景理解的条件生成模型。SceneDiffuser 提供了一个统一模型，用于解决场景条件下的生成、优化和规划问题。与以往工作相比，SceneDiffuser 天然具备场景感知、基于物理和面向目标的特性。借助迭代采样策略，SceneDiffuser 通过一个基于扩散的去噪过程，以完全可微的方式联合建模场景感知生成、基于物理的优化和面向目标的规划。这样的设计缓解了不同模块之间的不一致性，以及先前场景条件生成模型中的后验坍塌问题。我们在多种 3D 场景理解任务上评估了 SceneDiffuser，包括人体姿态与动作生成、灵巧抓取生成、3D 导航路径规划以及机械臂运动规划。结果表明，与先前模型相比有显著提升，展示了 SceneDiffuser 在广泛 3D 场景理解社区中的巨大潜力。

## 新闻

- [ 2023.04 ] 我们发布了抓取生成和机械臂运动规划的代码！

## 环境配置

完整安装指南（Python 3.11 + PyTorch 2.x + CUDA 12.8）请参见[新环境安装说明](./install_new_env_cn.md)。

完成安装后，后续运行通常只需要激活这个 conda 环境：

```bash
conda activate scene-diff
```

本 README 只保留当前主分支、默认环境下可直接运行的任务：

- `pose_gen`
- `motion_gen`
- `path_planning`

`obj` 分支上的抓取生成、机械臂规划等任务需要额外依赖，不在下文展开。可选依赖见 `requirements-optional.txt`。

## 数据与检查点

### 1. 数据

你可以使用我们[预处理好的数据](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing)，或者按照[说明](./preprocessing/README.md)自行处理数据。

将数据资源放在仓库根目录的 `data/` 下，默认布局如下：

```text
data/
├── PROXD_temp/              # pose_gen, motion_gen
├── PROX/                    # pose_gen, motion_gen
├── models_smplx_v1_1/       # pose_gen, motion_gen
├── V02_05/                  # 仅 optimizer.vposer=true 时需要
└── scannet_path_planning/   # path_planning
```

数据路径可通过环境变量覆盖，详见[新环境安装说明](./install_new_env_cn.md)。

### 2. 检查点

下载我们的[预训练模型](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing)，并将其放入 `./outputs/checkpoints/`。

任务|检查点|说明
-|-|-
预训练 Point Transformer|2022-04-13_18-29-56_POINTTRANS_C_32768|
姿态生成|2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100|
动作生成|2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300|无起始位置
动作生成|2022-11-09_14-28-12_MotionGen_ddm_T200_lr1e-4_ep300_obser|有起始位置
路径规划|2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL|

如果你是第一次运行，建议先直接使用作者提供的检查点验证流程，再考虑自行训练。

## 任务 1：3D 场景中的人体姿态生成

### 训练

- 使用单张 GPU 训练

    ```bash
    bash scripts/pose_gen/train.sh ${EXP_NAME}
    ```

- 使用 4 张 GPU 训练（修改 `scripts/pose_gen/train_ddm.sh` 以指定可见 GPU）

    ```bash
    bash scripts/pose_gen/train_ddm.sh ${EXP_NAME}
    ```

### 测试（定量评估）

```bash
bash scripts/pose_gen/test.sh ${CKPT} [OPT]
# 例如：bash scripts/pose_gen/test.sh ./outputs/checkpoints/2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100/ OPT
```

- `[OPT]` 是可选项，用于优化引导采样。

### 采样（定性可视化）

```bash
bash scripts/pose_gen/sample.sh ${CKPT} [OPT]
# 例如：bash scripts/pose_gen/sample.sh ./outputs/checkpoints/2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100/ OPT
```

- `[OPT]` 是可选项，用于优化引导采样。

## 任务 2：3D 场景中的人体动作生成

**默认配置为无观测的动作生成。如果你想探索带起始观测的动作生成设置，请将 `./scripts/motion_gen/` 文件夹中所有脚本里的 `task.has_observation` 改为 `true`。**

### 训练

- 使用单张 GPU 训练

    ```bash
    bash scripts/motion_gen/train.sh ${EXP_NAME}
    ```

- 使用 4 张 GPU 训练（修改 `scripts/motion_gen/train_ddm.sh` 以指定可见 GPU）

    ```bash
    bash scripts/motion_gen/train_ddm.sh ${EXP_NAME}
    ```

### 测试（定量评估）

```bash
bash scripts/motion_gen/test.sh ${CKPT} [OPT]
# 例如：bash scripts/motion_gen/test.sh ./outputs/checkpoints/2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300/ OPT
```

- `[OPT]` 是可选项，用于优化引导采样。

### 采样（定性可视化）

```bash
bash scripts/motion_gen/sample.sh ${CKPT} [OPT]
# 例如：bash scripts/motion_gen/sample.sh ./outputs/checkpoints/2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300/ OPT
```

- `[OPT]` 是可选项，用于优化引导采样。

## 任务 3：3D 物体的灵巧抓取生成

> 该任务位于 `obj` 分支，需要额外依赖和数据准备，当前改造后的环境暂不支持。

## 任务 4：3D 场景中的路径规划

> 详细新手教程请参见 **[Path Planning 运行教程](./tutorial_path_planning_cn.md)**。

`plan.sh` 和 `sample.sh` 都会渲染图片，因此 `data/scannet_path_planning/mesh/` 中还需要准备对应的 ScanNet `.ply` 网格文件。

### 训练

- 使用单张 GPU 训练

    ```bash
    bash scripts/path_planning/train.sh ${EXP_NAME}
    ```

- 使用 4 张 GPU 训练（修改 `scripts/path_planning/train_ddm.sh` 以指定可见 GPU）

    ```bash
    bash scripts/path_planning/train_ddm.sh ${EXP_NAME}
    ```

### 测试（定量评估）

```bash
bash scripts/path_planning/plan.sh ${CKPT}
# 例如：bash scripts/path_planning/plan.sh ./outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/
```

### 采样（定性可视化）

```bash
bash scripts/path_planning/sample.sh ${CKPT} [OPT] [PLA]
# 例如：bash scripts/path_planning/sample.sh ./outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ OPT PLA
```

- 程序将根据给定的起始位置和场景生成轨迹，并将结果渲染为图像。（这些结果并不是规划结果，只是使用 diffuser 生成多样化轨迹。）
- `[OPT]` 是可选项，用于优化引导采样。
- `[PLA]` 是可选项，用于规划器引导采样。

## 任务 5：机械臂运动规划

> 该任务位于 `obj` 分支，需要额外依赖和数据准备，当前改造后的环境暂不支持。

## 引用

如果你觉得我们的项目有帮助，请考虑引用我们：

```tex
@article{huang2023diffusion,
  title={Diffusion-based Generation, Optimization, and Planning in 3D Scenes},
  author={Huang, Siyuan and Wang, Zan and Li, Puhao and Jia, Baoxiong and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Zhu, Song-Chun},
  journal={arXiv preprint arXiv:2301.06015},
  year={2023}
}
```

## 致谢

部分代码借鉴自 [latent-diffusion](https://github.com/CompVis/latent-diffusion)、[PSI-release](https://github.com/yz-cnsdqz/PSI-release)、[Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet)、[point-transformer](https://github.com/POSTECH-CVLab/point-transformer) 和 [diffuser](https://github.com/jannerm/diffuser)。

### 许可证

本项目采用 MIT License。更多细节请参见 [LICENSE](LICENSE)。本项目使用了以下数据集，并受其各自许可证约束：

- PROX 采用[非商业科学研究用途软件版权许可证](https://prox.is.tue.mpg.de/license.html)。
- LEMO 采用 [MIT License](https://github.com/sanweiliti/LEMO/blob/main/LICENSE)。
- ScanNet V2 采用 [ScanNet 使用条款](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)。
