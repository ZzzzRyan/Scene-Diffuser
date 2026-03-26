# Diffusion-based Generation, Optimization, and Planning in 3D Scenes

English | <a href="./README_CN.md">中文</a>

<p align="left">
    <a href='https://scenediffuser.github.io/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2301.06015'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://scenediffuser.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo'>
      <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggingFace'>
    </a>
    <a href='https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='Checkpoints'>
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

This repository is the official implementation of paper "Diffusion-based Generation, Optimization, and Planning in 3D Scenes".

We introduce SceneDiffuser, a conditional generative model for 3D scene understanding. SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning. In contrast to prior work, SceneDiffuser is intrinsically scene-aware, physics-based, and goal-oriented.

[Paper](https://scenediffuser.github.io/paper.pdf) |
[arXiv](https://arxiv.org/abs/2301.06015) |
[Project](https://scenediffuser.github.io/) |
[HuggingFace Demo](https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo) |
[Checkpoints](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing)

<div align=center>
<img src='./figures/teaser.png' width=60%>
</div>

## Abstract

We introduce SceneDiffuser, a conditional generative model for 3D scene understanding. SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning. In contrast to prior works, SceneDiffuser is intrinsically scene-aware, physics-based, and goal-oriented. With an iterative sampling strategy, SceneDiffuser jointly formulates the scene-aware generation, physics-based optimization, and goal-oriented planning via a diffusion-based denoising process in a fully differentiable fashion. Such a design alleviates the discrepancies among different modules and the posterior collapse of previous scene-conditioned generative models. We evaluate SceneDiffuser with various 3D scene understanding tasks, including human pose and motion generation, dexterous grasp generation, path planning for 3D navigation, and motion planning for robot arms. The results show significant improvements compared with previous models, demonstrating the tremendous potential of SceneDiffuser for the broad community of 3D scene understanding.

## News

- [ 2023.04 ] We release the code for grasp generation and arm motion planning!

## Setup

For a complete installation guide (Python 3.11 + PyTorch 2.x + CUDA 12.8), see [New Environment Installation Guide_CN](./install_new_env_cn.md).

After finishing the guide, the normal workflow is simply:

```bash
conda activate scene-diff
```

This README focuses on the tasks that are runnable on the current main branch with the default environment:

- `pose_gen`
- `motion_gen`
- `path_planning`

Tasks on the `obj` branch (for example grasp generation and robot-arm planning) need extra dependencies and are not covered below. Optional packages are listed in `requirements-optional.txt`.

## Data & Checkpoints

> If you only want to reproduce "Task 4: Path Planning in 3D Scenes", you can directly refer to the [Path Planning Tutorial](./tutorial_path_planning_cn.md) for data preparation and running instructions.

### 1. Data

You can use our [pre-processed data](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing) or process the data by yourself following the [instructions](./preprocessing/README.md).

Even when using pre-processed data, there are still some additional officially released data assets that need to be downloaded. For details, please refer to the [instructions](./preprocessing/README.md).

Place data assets under `data/` at the repository root. The default layout is:

```text
data/
├── PROXD_temp/              # pose_gen, motion_gen
├── PROX/                    # pose_gen, motion_gen
├── models_smplx_v1_1/       # pose_gen, motion_gen
├── V02_05/                  # only if optimizer.vposer=true
└── scannet_path_planning/   # path_planning
```

All data path configurations are in `configs/default.yaml`. You can modify them as needed or override them directly through environment variables.

### 2. Checkpoints

Download our [pre-trained model](https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing) and place them under `./outputs/checkpoints/`.

task|checkpoints|desc
-|-|-
Pretrained Point Transformer|2022-04-13_18-29-56_POINTTRANS_C_32768|
Pose Generation|2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100|
Motion Generation|2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300|w/o start position
Motion Generation|2022-11-09_14-28-12_MotionGen_ddm_T200_lr1e-4_ep300_obser|w/ start position
Path Planning|2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL|

For a first run, it is recommended to use the released checkpoints before training your own models.

## Task-1: Human Pose Generation in 3D Scenes

### Train

- Train with single gpu

    ```bash
    bash scripts/pose_gen/train.sh ${EXP_NAME}
    ```

- Train with 4 GPUs (modify `scripts/pose_gen/train_ddm.sh` to specify the visible GPUs)

    ```bash
    bash scripts/pose_gen/train_ddm.sh ${EXP_NAME}
    ```

### Test (Quantitative Evaluation)

```bash
bash scripts/pose_gen/test.sh ${CKPT} [OPT]
# e.g., bash scripts/pose_gen/test.sh ./outputs/checkpoints/2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

### Sample (Qualitative Visualization)

```bash
bash scripts/pose_gen/sample.sh ${CKPT} [OPT]
# e.g., bash scripts/pose_gen/sample.sh ./outputs/checkpoints/2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

## Task-2: Human Motion Generation in 3D Scenes

**The default configuration is motion generation without observation. If you want to explore the setting of motion generation with start observation, please change the `task.has_observation` to `true` in all the scripts in folder `./scripts/motion_gen/`.**

### Train

- Train with single gpu

    ```bash
    bash scripts/motion_gen/train.sh ${EXP_NAME}
    ```

- Train with 4 GPUs (modify `scripts/motion_gen/train_ddm.sh` to specify the visible GPUs)

    ```bash
    bash scripts/motion_gen/train_ddm.sh ${EXP_NAME}
    ```

### Test (Quantitative Evaluation)

```bash
bash scripts/motion_gen/test.sh ${CKPT} [OPT]
# e.g., bash scripts/motion_gen/test.sh ./outputs/checkpoints/2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

### Sample (Qualitative Visualization)

```bash
bash scripts/motion_gen/sample.sh ${CKPT} [OPT]
# e.g., bash scripts/motion_gen/sample.sh ./outputs/checkpoints/2022-11-09_12-54-50_MotionGen_ddm_T200_lr1e-4_ep300/ OPT
```

- `[OPT]` is optional for optimization-guided sampling.

## Task-3: Dexterous Grasp Generation for 3D Objects

> This task is on the `obj` branch and requires extra dependencies and data preparation, which are not currently supported by the default environment.

## Task-4: Path Planning in 3D Scenes

For a detailed beginner tutorial in Chinese, see **[Path Planning Tutorial](./tutorial_path_planning_cn.md)**.

`plan.sh` and `sample.sh` both render images, so `data/scannet_path_planning/mesh/` must contain the corresponding ScanNet `.ply` files.

### Train

- Train with single gpu

    ```bash
    bash scripts/path_planning/train.sh ${EXP_NAME}
    ```

- Train with 4 GPUs (modify `scripts/path_planning/train_ddm.sh` to specify the visible GPUs)

    ```bash
    bash scripts/path_planning/train_ddm.sh ${EXP_NAME}
    ```

### Test (Quantitative Evaluation)

```bash
bash scripts/path_planning/plan.sh ${CKPT}
# e.g., bash scripts/path_planning/plan.sh ./outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/
```

### Sample (Qualitative Visualization)

```bash
bash scripts/path_planning/sample.sh ${CKPT} [OPT] [PLA]
# e.g., bash scripts/path_planning/sample.sh ./outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ OPT PLA
```

- The program will generate trajectories with given start position and scene; rendering the results into images. (The results not the planning results, just use diffuser to generate diverse trajectories.)
- `[OPT]` is optional for optimization-guided sampling.
- `[PLA]` is optional for planner-guided sampling.

## Task-5: Motion Planning for Robot Arms

> This task is on the `obj` branch and requires extra dependencies and data preparation, which are not currently supported by the default environment.

## Citation

If you find our project useful, please consider citing us:

```tex
@article{huang2023diffusion,
  title={Diffusion-based Generation, Optimization, and Planning in 3D Scenes},
  author={Huang, Siyuan and Wang, Zan and Li, Puhao and Jia, Baoxiong and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Zhu, Song-Chun},
  journal={arXiv preprint arXiv:2301.06015},
  year={2023}
}
```

## Acknowledgments

Some codes are borrowed from [latent-diffusion](https://github.com/CompVis/latent-diffusion), [PSI-release](https://github.com/yz-cnsdqz/PSI-release), [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet), [point-transformer](https://github.com/POSTECH-CVLab/point-transformer), and [diffuser](https://github.com/jannerm/diffuser).

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details. The following datasets are used in this project and are subject to their respective licenses:

- PROX is under the [Software Copyright License for non-commercial scientific research purposes](https://prox.is.tue.mpg.de/license.html).
- LEMO is under the [MIT License](https://github.com/sanweiliti/LEMO/blob/main/LICENSE).
- ScanNet V2 is under the [ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf).
