# Task-4: Path Planning 运行教程

本教程指导你在当前仓库环境下完整运行 **Path Planning（3D 室内路径规划）** 任务，包括定量评估（plan）和定性可视化（sample）。

前提：你已按照 [新环境安装说明](./install_new_env_cn.md) 完成 `scene-diff` conda 环境的安装和验证。

---

## 1. 任务简介

Path Planning 任务的目标是：给定一个 3D 室内场景和起点/终点，使用扩散模型生成从起点到终点的导航路径。

本任务提供三种运行模式：

| 模式 | 脚本 | 说明 |
|---|---|---|
| 训练 | `scripts/path_planning/train.sh` | 从头训练模型 |
| 定量评估 | `scripts/path_planning/plan.sh` | 用预训练模型做路径规划，输出成功率和路径长度指标 |
| 定性可视化 | `scripts/path_planning/sample.sh` | 用预训练模型采样，渲染路径图片 |

---

## 2. 数据准备

### 2.1 数据总览

Path Planning 任务需要以下数据，全部放在 `data/scannet_path_planning/` 下：

```text
data/scannet_path_planning/
├── scene/          # 预处理后的场景点云 (.npy)
├── path/           # 导航路径标注 (.pkl)
├── height/         # 高度图，用于碰撞检测 (.pkl)
└── mesh/           # ScanNet 原始场景网格 (.ply)，用于结果可视化
```

其中 `scene/`、`path/`、`height/` 三个目录由项目作者提供的预处理数据覆盖，`mesh/` 需要从 ScanNet 官方获取。

### 2.2 下载预处理数据（scene / path / height）

项目作者在 Google Drive 上提供了预处理数据：

> https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing

在该 Google Drive 页面中找到 `data/scannet_path_planning.zip` 压缩包，下载并解压其中的 `scene/`、`path/`、`height/` 三个子目录，放到：

```
data/scannet_path_planning/
```

### 2.3 下载场景网格文件（mesh）

`mesh/` 目录存放的是 ScanNet-V2 数据集的原始场景网格文件，文件名格式为 `{scene_id}_vh_clean_2.ply`。这些文件在定量评估的可视化环节和定性可视化中被用于渲染路径结果图片。

**获取方式：**

mesh 文件来自 ScanNet-V2 数据集。你需要：

1. 前往 ScanNet 官方页面申请数据访问权限：

   > http://www.scan-net.org/

   填写 Terms of Use 表单，提交后会收到包含下载脚本的邮件。

2. 使用官方提供的下载脚本下载你需要的场景。本项目使用以下 61 个场景：

   ```
   scene0000_00  scene0005_00  scene0006_00  scene0008_00  scene0012_00
   scene0022_00  scene0025_00  scene0030_00  scene0040_00  scene0051_00
   scene0064_00  scene0114_00  scene0122_00  scene0132_00  scene0134_00
   scene0137_00  scene0142_00  scene0145_00  scene0151_00  scene0160_00
   scene0187_00  scene0192_00  scene0199_00  scene0202_00  scene0231_00
   scene0247_00  scene0261_00  scene0269_00  scene0276_00  scene0281_00
   scene0294_00  scene0296_00  scene0297_00  scene0309_00  scene0317_00
   scene0363_00  scene0370_00  scene0403_00  scene0420_00  scene0435_00
   scene0477_00  scene0505_00  scene0515_00  scene0536_00  scene0549_00
   scene0588_00  scene0603_00  scene0621_00  scene0626_00  scene0634_00
   scene0637_00  scene0640_00  scene0641_00  scene0645_00  scene0653_00
   scene0667_00  scene0672_00  scene0673_00  scene0678_00  scene0694_00
   scene0698_00
   ```

   ScanNet 官方下载脚本的典型用法：

   ```bash
   python download-scannet.py -o /tmp/scannet_raw --type _vh_clean_2.ply --id scene0000_00
   ```

   `--type _vh_clean_2.ply` 表示只下载该场景的网格文件。你可以写一个循环批量下载所有 61 个场景。

3. 将下载得到的 `.ply` 文件放入：

   ```
   data/scannet_path_planning/mesh/
   ```

   最终 mesh 目录内容应该类似：

   ```
   data/scannet_path_planning/mesh/
   ├── scene0000_00_vh_clean_2.ply
   ├── scene0005_00_vh_clean_2.ply
   ├── ...
   └── scene0698_00_vh_clean_2.ply
   ```

   可以用下面的命令检查文件是否准备好：

   ```bash
   ls data/scannet_path_planning/mesh/ | wc -l   # 应该是 61
   ```

> 提示：如果你在项目作者的 Google Drive 中看到了 mesh 文件夹且包含 `.ply` 文件，也可以直接从那里下载，跳过 ScanNet 申请流程。但据 zr 观察，Google Drive 上没有。

---

## 3. Checkpoint 准备

Path Planning 任务需要两个预训练模型：

| Checkpoint | 用途 | 默认位置 |
|---|---|---|
| Point Transformer | 场景编码器 | `outputs/checkpoints/2022-04-13_18-29-56_POINTTRANS_C_32768/model.pth` |
| Path Planning 模型 | 扩散模型 | `outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ckpts/model.pth` |

### 3.1 下载

从项目作者的 Google Drive 下载：

> https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing

下载以下两个文件夹：

- `2022-04-13_18-29-56_POINTTRANS_C_32768`（包含 `model.pth`）
- `2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL`（包含 `ckpts/model.pth`）

### 3.2 放置

将下载的文件夹放到 `outputs/checkpoints/` 下：

```text
outputs/
└── checkpoints/
    ├── 2022-04-13_18-29-56_POINTTRANS_C_32768/
    │   └── model.pth
    └── 2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/
        └── ckpts/
            └── model.pth
```

### 3.3 验证

```bash
ls outputs/checkpoints/2022-04-13_18-29-56_POINTTRANS_C_32768/model.pth
ls outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ckpts/model.pth
```

两个命令都应该正常输出文件路径，没有 "No such file" 错误。

---

## 4. 运行前检查

在运行任务之前，确认环境和数据都已就绪：

```bash
# 1. 激活环境
conda activate scene-diff

# 2. 确认 PyTorch 和 GPU
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 3. 确认 pointops
python -c "import pointops_cuda; print('pointops OK')"

# 4. 确认数据完整
echo "scene: $(ls data/scannet_path_planning/scene/ | wc -l)"
echo "path:  $(ls data/scannet_path_planning/path/ | wc -l)"
echo "height:$(ls data/scannet_path_planning/height/ | wc -l)"
echo "mesh:  $(ls data/scannet_path_planning/mesh/ | wc -l)"

# 5. 确认 checkpoint
ls outputs/checkpoints/2022-04-13_18-29-56_POINTTRANS_C_32768/model.pth
ls outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ckpts/model.pth
```

预期输出：PyTorch 版本为 `2.11.0+cu128`，CUDA 可用，pointops OK，四个数据目录各 61 个文件，两个 checkpoint 都存在。

---

## 5. 运行定量评估（plan）

定量评估会让模型在测试集的场景中做路径规划，统计成功率和路径长度，同时渲染前 32 个 case 的可视化结果。

### 5.1 运行命令

```bash
conda activate scene-diff  # 激活环境

bash scripts/path_planning/plan.sh outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL
```

### 5.2 输出结果

结果保存在 checkpoint 目录下的 `eval/plan/` 子目录中：

```text
outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/
└── eval/
    └── plan/
        └── 2026-xx-xx_xx-xx-xx/
            ├── plan.log              # 运行日志
            ├── metrics.json          # 定量指标（成功率、路径长度等）
            └── scene0xxx_00_xxxxx/   # 可视化渲染图（前 32 个 case）
                └── res_xxx.png
```

`metrics.json` 中的关键指标：

- `succ_average`：规划成功率
- `length_average`：平均路径长度

---

## 6. 运行定性可视化（sample）

定性可视化会用扩散模型对测试集场景进行采样，渲染生成的路径图片。

### 6.1 基本运行

```bash
bash scripts/path_planning/sample.sh outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL
```

这会运行不带任何引导的纯扩散采样。

### 6.2 带引导的运行

```bash
# 带优化器引导
bash scripts/path_planning/sample.sh outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL OPT

# 带路径规划器引导
bash scripts/path_planning/sample.sh outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL PLA

# 同时带优化器 + 规划器引导
bash scripts/path_planning/sample.sh outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL OPT PLA
```

| 参数 | 含义 |
|---|---|
| `OPT` | 启用优化器引导（optimizer guidance），帮助避免碰撞 |
| `PLA` | 启用规划器引导（planner guidance），帮助趋向目标 |

### 6.3 输出结果

结果保存在 checkpoint 目录下的 `eval/final/` 子目录中：

```text
outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/
└── eval/
    └── final/
        └── 2026-xx-xx_xx-xx-xx/
            ├── sample.log
            └── scene0xxx_00_xxxxx/
                ├── 000.png       # 第 1 条采样路径的渲染图
                ├── 001.png       # 第 2 条采样路径的渲染图
                └── ...
```

每个场景会生成多条采样路径（默认 10 条，由 `task.visualizer.ksample=10` 控制），渲染为 PNG 图片。

---

## 7. 从头训练（可选）

如果你想自己训练 Path Planning 模型：

### 7.1 单卡训练

```bash
bash scripts/path_planning/train.sh my_path_exp
```

`my_path_exp` 是你给实验起的名字，训练产出保存在 `outputs/<timestamp>_my_path_exp/`。

### 7.2 多卡训练

编辑 `scripts/path_planning/train_ddm.sh` 中的 `CUDA_VISIBLE_DEVICES` 指定你要使用的 GPU，然后：

```bash
bash scripts/path_planning/train_ddm.sh my_path_exp
```

### 7.3 用自己训练的模型做评估

训练完成后，checkpoint 保存在 `outputs/<timestamp>_my_path_exp/ckpts/model.pth`。用它替换前面的 checkpoint 路径即可：

```bash
bash scripts/path_planning/plan.sh outputs/<timestamp>_my_path_exp
bash scripts/path_planning/sample.sh outputs/<timestamp>_my_path_exp OPT PLA
```

---

## 8. 如何修改数据路径

如果你的数据或 checkpoint 不在默认位置，有以下几种方式修改。

### 8.1 通过环境变量（推荐）

```bash
# 修改 scannet 数据目录
export SCENE_DIFFUSER_SCANNET_PATH_DIR=/your/path/scannet_path_planning

# 修改 Point Transformer 权重路径
export SCENE_DIFFUSER_POINT_TRANSFORMER_CKPT=/your/path/model.pth

# 修改整个 data 根目录（影响所有任务）
export SCENE_DIFFUSER_DATA_DIR=/your/path/data

# 修改整个 checkpoints 根目录（影响所有任务）
export SCENE_DIFFUSER_CHECKPOINTS_DIR=/your/path/checkpoints
```

设置后直接运行脚本即可生效。

### 8.2 通过命令行参数

也可以在运行时通过 Hydra 参数覆盖：

```bash
python plan.py \
    exp_dir=outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL \
    diffuser=ddpm \
    model=unet \
    model.use_position_embedding=true \
    task=path_planning \
    task.dataset.repr_type=relative \
    task.env.inpainting_horizon=16 \
    task.env.eval_case_num=64 \
    task.env.robot_top=3.0 \
    task.env.env_adaption=false \
    optimizer=path_in_scene \
    optimizer.scale_type=div_var \
    optimizer.continuity=false \
    planner=greedy_path_planning \
    planner.scale=0.2 \
    planner.scale_type=div_var \
    planner.greedy_type=all_frame_exp \
    paths.scannet_path_planning_dir=/your/path/scannet_path_planning
```

### 8.3 路径解析链路

配置路径的默认解析顺序：

```
环境变量（最高优先级）
  ↓ 如果未设置
configs/default.yaml 中的 paths.* 默认值
  ↓ 展开为
${output_dir}/checkpoints  →  outputs/checkpoints
${repo_root}/data          →  data/
```

path_planning 任务涉及的具体路径映射：

| 配置键 | 默认值 | 环境变量 |
|---|---|---|
| `paths.scannet_path_planning_dir` | `data/scannet_path_planning` | `SCENE_DIFFUSER_SCANNET_PATH_DIR` |
| `paths.point_transformer_ckpt` | `outputs/checkpoints/2022-04-13_18-29-56_POINTTRANS_C_32768/model.pth` | `SCENE_DIFFUSER_POINT_TRANSFORMER_CKPT` |
| `task.dataset.data_dir` | 来自 `paths.scannet_path_planning_dir` | 同上 |
| `task.env.scannet_mesh_dir` | 来自 `task.dataset.data_dir` | 同上 |
| `task.visualizer.scannet_mesh_dir` | 来自 `task.dataset.data_dir` | 同上 |

注意 `mesh/` 子目录名是硬编码在代码中的，所以你只需要保证 `scannet_path_planning_dir` 指向的目录下有 `mesh/`、`scene/`、`path/`、`height/` 四个子目录。

---

## 9. 可视化渲染说明

可视化渲染使用 `pyrender` 做离屏渲染。在 WSL2 环境下，默认的 OpenGL 后端可能需要额外配置。

如果渲染报错（如 OpenGL 相关错误），尝试设置渲染后端为 osmesa 或 egl：

```bash
# 方式一：osmesa
export RENDERING_BACKEND=osmesa

# 方式二：egl（需要 GPU 驱动支持 EGL）
export RENDERING_BACKEND=egl
```

如果使用 osmesa，需要确保系统安装了 `libosmesa6-dev`：

```bash
sudo apt-get install libosmesa6-dev
```

---

## 10. 常见问题

### Q: 运行时报 `FileNotFoundError: ... _vh_clean_2.ply`

说明 `mesh/` 目录缺少对应场景的网格文件。请按本教程第 2.3 节获取 ScanNet 网格文件。

### Q: 运行时报 `Can't find pretrained point-transformer weights`

检查 Point Transformer checkpoint 是否存在：

```bash
ls outputs/checkpoints/2022-04-13_18-29-56_POINTTRANS_C_32768/model.pth
```

如果不存在，从 Google Drive 下载。

### Q: 运行时报 `Can't find provided ckpt`

检查 Path Planning 模型 checkpoint 是否存在：

```bash
ls outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ckpts/model.pth
```

### Q: 想减少评估数量加快测试

通过命令行覆盖 eval_case_num：

```bash
python plan.py \
    exp_dir=outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL \
    diffuser=ddpm model=unet model.use_position_embedding=true \
    task=path_planning task.dataset.repr_type=relative \
    task.env.inpainting_horizon=16 task.env.eval_case_num=8 \
    task.env.robot_top=3.0 task.env.env_adaption=false \
    optimizer=path_in_scene optimizer.scale_type=div_var optimizer.continuity=false \
    planner=greedy_path_planning planner.scale=0.2 planner.scale_type=div_var planner.greedy_type=all_frame_exp
```

将 `eval_case_num=8` 改为你需要的数量。

### Q: 想指定使用哪个 GPU

```bash
# 方式一：命令行参数
bash scripts/path_planning/plan.sh outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL
# 默认使用 gpu=0

# 方式二：在脚本命令中添加 gpu=1 参数手动覆盖
```

或者通过环境变量限制可见 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/path_planning/plan.sh outputs/checkpoints/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL
```
