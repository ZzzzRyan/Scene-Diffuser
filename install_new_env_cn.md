# Scene-Diffuser 新环境安装说明

本文档对应当前仓库这套新环境方案，目标是：

- 使用独立 conda 环境，不污染 `base`
- 适配 `Python 3.11 + PyTorch 2.11.0+cu128 + CUDA 12.8`
- 编译 `pointops`
- 把运行期所需的 CUDA / PyTorch 动态库路径固化到当前 conda 环境
- 后续直接使用仓库自带脚本运行，不再额外套一层包装脚本

本文档默认你使用的是：

- WSL2
- Miniconda / conda
- NVIDIA 显卡
- 当前仓库目录为 `Scene-Diffuser`

## 1. 当前方案支持范围

当前主环境默认支持这些任务：

- `pose_gen`
- `motion_gen`
- `path_planning`
- `pose_gen` / `motion_gen` 的优化引导版本，但要求 `optimizer.vposer=false`

当前主环境**不支持**的内容：

- `optimizer.vposer=true`
- 抓取 / 手模型 / `obj` 分支相关任务
- README 里原始 `obj` 分支的 `Isaac Gym` / `pointnet2` 教程

原因很简单：

- 当前仓库主分支真正保留并可直接运行的是人体姿态、人体运动、路径规划三条主链
- `human_body_prior`、抓取依赖等已经被拆到可选范围
- 可选依赖当前会重新解析 `torch`，容易破坏已经验证过的主环境

## 2. 已验证的软件组合

这套方案按下面的组合做过兼容性验证：

- Python：`3.11`
- PyTorch：`2.11.0+cu128`
- torchvision：`0.26.0+cu128`
- CUDA 编译工具：`12.8`
- GPU：`RTX 5080`

## 3. 依赖文件说明

当前仓库里的依赖文件含义如下：

- [requirements.txt](/home/zryan/04_Dev_Workspace/Forks/Scene-Diffuser/requirements.txt)
  核心 Python 依赖
- [requirements-cuda-ext.txt](/home/zryan/04_Dev_Workspace/Forks/Scene-Diffuser/requirements-cuda-ext.txt)
  需要针对当前 `torch + cuda` 编译的扩展，当前主要是 `pointops`
- [requirements-optional.txt](/home/zryan/04_Dev_Workspace/Forks/Scene-Diffuser/requirements-optional.txt)
  可选依赖，不建议装进当前主环境
- [environment.yml](/home/zryan/04_Dev_Workspace/Forks/Scene-Diffuser/environment.yml)
  参考性环境定义，不是本文档推荐的新手入口

## 4. 安装前检查

先确认 WSL 里 GPU 正常：

```bash
nvidia-smi
```

如果这一步不通，先处理驱动和 WSL GPU 支持，不要继续安装项目依赖。

## 5. 创建独立 conda 环境

进入仓库目录：

```bash
cd /path/to/Scene-Diffuser
```

创建环境：

```bash
conda create -n scene-diff -y \
  python=3.11 pip=25.1 setuptools wheel \
  ninja cmake make pkg-config git \
  gcc_linux-64=13 gxx_linux-64=13 \
  -c conda-forge
```

激活环境：

```bash
conda activate scene-diff
```

## 6. 安装 CUDA 编译工具

当前项目需要编译 `pointops`，只装 PyTorch 不够，还需要 `nvcc` 和 CUDA 头文件：

```bash
conda install -y -c nvidia -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8
```

这一步只会写入当前 conda 环境。

## 7. 安装 PyTorch

安装官方 `cu128` 版本：

```bash
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 \
  torchvision==0.26.0+cu128
```

建议立即检查：

```bash
python -m pip check
```

## 8. 安装项目核心依赖

先安装普通 Python 依赖：

```bash
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check
```

再安装 CUDA 扩展依赖：

```bash
bash scripts/with_scene_diffuser_cuda_env.sh \
  python -m pip install --no-cache-dir -r requirements-cuda-ext.txt
```

## 9. 把运行期环境固化到 conda 环境

安装完成后，执行一次：

```bash
bash scripts/install_scene_diffuser_conda_hooks.sh
```

这个脚本会把运行 `pointops_cuda` 所需的环境变量写入当前 conda 环境的：

- `etc/conda/activate.d/`
- `etc/conda/deactivate.d/`

写入完成后，重新激活环境：

```bash
conda deactivate
conda activate scene-diff
```

**至此，环境配置完毕！可以支持任务1、任务2、任务4的运行。**

## 10. 建议做一次安装后检查

重新激活环境后，建议检查这几项。

检查 `torch` 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

检查 `pointops_cuda`：

```bash
python -c "import pointops_cuda; print(pointops_cuda.__file__)"
```

检查仓库里的 `pointops` 封装：

```bash
python -c "from models.model import pointops; print(pointops.__file__)"
```

如果这三步都正常（没有报错），说明：

- PyTorch 版本链路正确
- 当前 conda 环境里的 CUDA 运行库路径已经生效
- 后续可以直接用仓库自带脚本运行任务

## 11. 后续运行任务的正确方式

完成前面的安装和固化后，后续使用方式就是：

```bash
conda activate scene-diff
```

然后直接运行脚本，例如：

```bash
bash scripts/pose_gen/train.sh my_pose_exp
bash scripts/pose_gen/test.sh /path/to/ckpt OPT

bash scripts/motion_gen/train.sh my_motion_exp
bash scripts/motion_gen/sample.sh /path/to/ckpt OPT

bash scripts/path_planning/plan.sh /path/to/ckpt
bash scripts/path_planning/sample.sh /path/to/ckpt OPT PLA
```

> 以上只是示例，具体如何运行任务请参见教程文档，如：[README_CN.md](./README_CN.md)、[Path Planning 运行教程](./tutorial_path_planning_cn.md) 。

## 12. 不建议你现在在主环境里做的事

因为当前主环境的目标是稳定跑通 任务1：`pose_gen`、任务2：`motion_gen`、任务4：`path_planning` 三条主链，所以以下操作不建议在当前主环境里做：

### 12.1 直接安装 `requirements-optional.txt`

不建议。

原因：

- 它会重新解析 `torch`
- 有较大概率破坏当前已经稳定的 `torch 2.11.0+cu128`

### 12.2 把 `optimizer.vposer=true` 当默认方案

不建议。

原因：

- 它依赖 `human_body_prior`
- 当前主环境没有把这条链路纳入稳定支持范围

### 12.3 把 README 里 `obj` 分支任务当作当前默认目标

不建议。

原因：

- 当前仓库并没有 README 里 `obj` 分支对应的完整脚本集合
- 这些教程不是当前主分支主环境的默认运行目标

## 13. 额外说明

- [scripts/with_scene_diffuser_cuda_env.sh](/home/zryan/04_Dev_Workspace/Forks/Scene-Diffuser/scripts/with_scene_diffuser_cuda_env.sh) 现在主要作为编译和排障工具保留
- 当前主环境目标很明确：先把当前主分支的人体姿态、人体运动、路径规划三条任务链稳定跑起来，其他任务需要 obj 分支的额外依赖和数据准备，暂不纳入当前主环境支持范围
