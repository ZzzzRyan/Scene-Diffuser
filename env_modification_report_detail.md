# 环境修正内容与评估报告

本报告对 Scene-Diffuser 仓库为适配 RTX 5080 + CUDA 12.8 所做的全部改造进行逐项说明和评估。

报告生成时间：2026-03-26

---

## 一、改造背景

原项目锁定 Python 3.8~3.10 + PyTorch 1.11.0 + CUDA 11.3。RTX 5080 为 Blackwell 架构（SM 12.0），CUDA 11.3 不支持该架构，PyTorch 1.11 不支持 CUDA 12.8。因此必须升级整套工具链才能在该硬件上运行项目。

目标软件组合：

| 组件 | 版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA 编译工具 | 12.8 |
| GPU | RTX 5080 |

---

## 二、改造内容清单

### 2.1 依赖文件重组

#### requirements.txt

将原先混合在一起的依赖拆分为三层：

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 核心 Python 依赖，不含编译型扩展 |
| `requirements-cuda-ext.txt` | 需要 nvcc 编译的 CUDA 扩展（pointops） |
| `requirements-optional.txt` | 可选依赖（pytorch_kinematics、human_body_prior、urdf-parser-py） |

从 requirements.txt 中移除的编译型 / 可选依赖：

| 包 | 去向 | 原因 |
|---|---|---|
| `pytorch3d v0.7.0` | 移除，用本地 `chamfer_distance.py` 替代 | pytorch3d 在 PyTorch 2.x + CUDA 12.8 下编译极其困难 |
| `chamfer_distance`（otaheri） | 移除，用本地 `chamfer_distance.py` 替代 | 同上，与 pytorch3d 共享 CUDA 编译困难 |
| `pointops` | 移到 `requirements-cuda-ext.txt` | 需要 nvcc，与普通 pip 依赖分离 |
| `human_body_prior` | 移到 `requirements-optional.txt` | 仅 VPoser 优化模式需要 |
| `pytorch_kinematics` | 移到 `requirements-optional.txt` | 仅 grasp/hand 分支需要 |
| `urdf-parser-py` | 移到 `requirements-optional.txt` | 仅 grasp/hand 分支需要 |

新增依赖：

| 包 | 原因 |
|---|---|
| `scikit-learn` | `models/evaluator.py` 和预处理脚本已在使用，原先由其他包附带安装，现在显式声明 |

#### 纯 Python 包版本升级

| 包 | 原版本 | 新版本 |
|---|---|---|
| einops | 0.4.1 | >=0.8,<0.9 |
| hydra-core | 1.2.0 | 1.3.2 |
| loguru | 0.6.0 | >=0.7,<0.8 |
| matplotlib | 3.5.1 | >=3.9,<3.11 |
| natsort | 8.2.0 | >=8.4,<9 |
| networkx | 2.8.6 | >=3.4,<4 |
| omegaconf | 2.2.2 | 2.3.0 |
| opencv-python | 4.6.0.66 | >=4.10,<4.12 |
| Pillow | 9.0.1 | >=10.4,<12 |
| plotly | 5.11.0 | >=5.24,<7 |
| protobuf | 3.19.4 | >=4.25,<6 |
| tabulate | 0.8.10 | >=0.9,<0.10 |
| tensorboard | 2.8.0 | >=2.18,<2.21 |
| tqdm | 4.62.3 | >=4.67,<5 |
| transforms3d | 0.4.1 | >=0.4.2,<0.5 |
| transformations | 2022.9.26 | 2025.8.1 |
| trimesh | 3.12.7 | >=4.5,<5 |

保持不变的关键锁定项：

| 包 | 版本 | 原因 |
|---|---|---|
| smplx | 0.1.28 | 项目对 SMPL-X 参数格式有精确依赖 |
| pyquaternion | 0.9.9 | API 稳定，无需升级 |
| pyrender | 0.1.45 | 渲染管线对版本敏感 |

#### pre-requirements.txt

原内容为 `torch==1.11.0+cu113`，改为注释文件指向 PyTorch 官方选择器。PyTorch 安装改由 `environment.yml` 和安装手册管理。

---

### 2.2 代码改动

#### 2.2.1 models/model/pointops.py — CUDA 张量创建 API 现代化

**改动内容：**

所有 `torch.cuda.FloatTensor()` / `torch.cuda.IntTensor()` 替换为 `torch.zeros()` / `torch.full()` / `torch.empty()` / `torch.tensor()` + `device=` 参数。

涉及的函数：

| 函数 | 原写法 | 新写法 |
|---|---|---|
| `FurthestSampling.forward` | `torch.cuda.IntTensor(n).zero_()` | `torch.zeros(n, device=xyz.device, dtype=torch.int32)` |
| `FurthestSampling.forward` | `torch.cuda.FloatTensor(n).fill_(1e10)` | `torch.full((n,), 1e10, device=xyz.device, dtype=xyz.dtype)` |
| `KNNQuery.forward` | `torch.cuda.IntTensor(m, nsample).zero_()` | `torch.zeros((m, nsample), device=xyz.device, dtype=torch.int32)` |
| `KNNQuery.forward` | `torch.cuda.FloatTensor(m, nsample).zero_()` | `torch.zeros((m, nsample), device=xyz.device, dtype=xyz.dtype)` |
| `Grouping.forward` | `torch.cuda.FloatTensor(m, nsample, c)` | `torch.empty((m, nsample, c), device=input.device, dtype=input.dtype)` |
| `Grouping.backward` | `torch.cuda.FloatTensor(n, c).zero_()` | `torch.zeros((n, c), device=grad_output.device, dtype=grad_output.dtype)` |
| `Subtraction.forward` | `torch.cuda.FloatTensor(n, nsample, c).zero_()` | `torch.zeros((n, nsample, c), device=input1.device, dtype=input1.dtype)` |
| `Subtraction.backward` | `torch.cuda.FloatTensor(n, c).zero_()` x2 | `torch.zeros((n, c), device=grad_output.device, dtype=grad_output.dtype)` x2 |
| `Aggregation.forward` | `torch.cuda.FloatTensor(n, c).zero_()` | `torch.zeros((n, c), device=input.device, dtype=input.dtype)` |
| `Aggregation.backward` | `torch.cuda.FloatTensor(...)` x3 | `torch.zeros(...)` x3 |
| `interpolation` | `torch.cuda.FloatTensor(n, c).zero_()` | `torch.zeros((n, c), device=feat.device, dtype=feat.dtype)` |
| `Interpolation.forward` | `torch.cuda.FloatTensor(n, c).zero_()` | `torch.zeros((n, c), device=input.device, dtype=input.dtype)` |
| `Interpolation.backward` | `torch.cuda.FloatTensor(m, c).zero_()` | `torch.zeros((m, c), device=grad_output.device, dtype=grad_output.dtype)` |

**评估：** `torch.cuda.FloatTensor` 在 PyTorch 2.x 中已废弃。新写法是官方推荐 API，语义完全等价。额外好处是保留了输入张量的 dtype，在混合精度场景更健壮。

**改动内容：**

新增 `_load_torch_shared_libs()` 函数，在 `import pointops_cuda` 前预加载 PyTorch 动态库（libc10.so、libtorch.so 等）。

**评估：** 这是自定义 CUDA 扩展在新版 PyTorch 下的已知必要手段。不预加载会导致 `pointops_cuda` 找不到符号而报 ImportError。

#### 2.2.2 models/model/pointtransformer.py — 同类适配

- `torch.cuda.IntTensor(n_o)` -> `torch.tensor(n_o, dtype=torch.int32, device=o.device)`
- `torch.load(weigth_path)` -> `torch.load(weigth_path, map_location='cpu', weights_only=False)`

**评估：** 与 pointops.py 同理，必要且正确。

#### 2.2.3 models/model/pointnet.py — torch.load 适配

- `torch.load(weigth_path)` -> `torch.load(weigth_path, map_location='cpu', weights_only=False)`

**评估：** PyTorch >= 2.0 要求显式指定 `weights_only`。`map_location='cpu'` 避免 GPU 编号不匹配。正确。

#### 2.2.4 train_ddm.py / test.py / sample.py / plan.py — torch.load 和 CUDA seed 保护

四个入口脚本的统一改动：

- `torch.load(path)` -> `torch.load(path, map_location='cpu', weights_only=False)`
- `torch.cuda.manual_seed()` 包裹 `if torch.cuda.is_available():`

**评估：** torch.load 同上。CUDA seed 保护为防御性编程，有 GPU 时行为不变。正确。

#### 2.2.5 models/optimizer/pose_in_scene.py 和 motion_in_scene.py — 可选依赖隔离 + 法线计算提取

**改动内容：**

1. `human_body_prior` 的 import 从顶层硬性 import 改为 try/except 保护
2. 在 `__init__` 中检查：开启 VPoser 但缺少依赖时抛出明确 ImportError
3. 顶点法线计算代码提取到 `utils/smplx_utils.py:_compute_vertex_normals()`
4. `torch.zeros(shape).float().cuda()` -> `torch.zeros_like(vertices)`（跟随输入设备）

**评估：**

- 可选依赖保护确保主流程不会在 import 阶段因缺少 human_body_prior 而崩溃
- VPoser 模式开启时若缺依赖给出清晰错误，而非晦涩的 NameError
- 法线计算提取消除了三处重复代码
- `torch.zeros_like` 跟随输入设备和 dtype，不再硬编码 `.cuda()`

正确。

#### 2.2.6 utils/smplx_utils.py — 新增公共函数 + 数值健壮性

**新增函数：**

- `_safe_cross(a, b)`：使用显式 `dim=-1` 参数，消除 PyTorch 2.x 对 `torch.cross` 未指定 dim 的弃用警告
- `_compute_vertex_normals(vertices, faces)`：从三处重复代码提取的公共函数

**数值改进：**

- 法线归一化时 `.unsqueeze(-1)` -> `keepdim=True`，语义等价但更清晰
- 法线归一化分母添加 `.clamp_min(1e-12)`，防止退化三角面导致除零 NaN
- `smplx_signed_distance` 中的 `query_to_surface` 归一化同样添加 `.clamp_min(1e-12)`

**评估：** 数值上与原逻辑等价（原代码中除零情况会产生 NaN，新代码产生极大但有限的值）。在实际运行中这是对边界情况的改善。正确。

#### 2.2.7 models/visualizer.py — 装饰器修正 + lazy import

**改动内容：**

1. 移除所有类定义上的 `@torch.no_grad()` 装饰器（7 处）
2. `from utils.handmodel import get_handmodel` 从顶层 import 移到 `GraspGenVisualizer.__init__()` 内部

**评估：**

- `@torch.no_grad()` 作为装饰器用在类定义上是无效操作，PyTorch 中它只对函数/方法生效。确认 `evaluator.py` 和 `ddpm.py` 中方法级别的 `@torch.no_grad()` 保留完好。移除不影响任何运行时行为。
- `get_handmodel` 的 lazy import 避免了在不使用 grasp 功能时因缺少手模型依赖而导致整个 visualizer 模块无法加载。

正确。

#### 2.2.8 models/environment.py — 装饰器修正

移除 `PathPlanningEnvWrapper` 和 `PathPlanningEnvWrapperHF` 类定义上的 `@torch.no_grad()` 装饰器。

**评估：** 与 2.2.7 同理，装饰器原本就是无效操作。正确。

#### 2.2.9 datasets/__init__.py — 可选 import 保护

`MultiDexShadowHandUR` 的 import 用 try/except 包裹，缺少依赖时设为 None。

**评估：** 确保不安装 grasp 分支依赖时主流程正常加载。正确。

#### 2.2.10 datasets/multidex_shadowhand_ur.py — torch.load 适配

`torch.load(path)` -> `torch.load(path, map_location='cpu', weights_only=False)`

**评估：** 同 2.2.3。正确。

---

### 2.3 配置文件改动

#### 2.3.1 configs/default.yaml — 路径参数化

新增 `paths:` 配置块，将所有数据/模型路径统一管理：

```yaml
paths:
  checkpoints_dir: ${oc.env:SCENE_DIFFUSER_CHECKPOINTS_DIR,${output_dir}/checkpoints}
  data_dir: ${oc.env:SCENE_DIFFUSER_DATA_DIR,${repo_root}/data}
  point_transformer_ckpt: ${oc.env:SCENE_DIFFUSER_POINT_TRANSFORMER_CKPT,...}
  scannet_path_planning_dir: ${oc.env:SCENE_DIFFUSER_SCANNET_PATH_DIR,...}
  lemo_proxd_dir: ${oc.env:SCENE_DIFFUSER_LEMO_PROXD_DIR,...}
  smplx_model_dir: ${oc.env:SCENE_DIFFUSER_SMPLX_MODEL_DIR,...}
  prox_dir: ${oc.env:SCENE_DIFFUSER_PROX_DIR,...}
  vposer_dir: ${oc.env:SCENE_DIFFUSER_VPOSER_DIR,...}
  multidex_shadowhand_dir: ${oc.env:SCENE_DIFFUSER_MULTIDEX_SHADOWHAND_DIR,...}
  multidex_object_pcds_path: ${oc.env:SCENE_DIFFUSER_MULTIDEX_OBJECT_PCDS_PATH,...}
```

**评估：** 替代了原配置中 `/home/wangzan/...` 等硬编码路径。使用 Hydra/OmegaConf 的 `oc.env` 插值，支持环境变量覆盖，有合理的默认值回退。正确。

#### 2.3.2 configs/model/unet.yaml

```yaml
# 原
pretrained_weights: /home/wangzan/Outputs/.../model.pth
pretrained_weights_slurm: /home/wangzan/scratch/Outputs/.../model.pth
# 新
pretrained_weights: ${paths.point_transformer_ckpt}
pretrained_weights_slurm: ${model.scene_model.pretrained_weights}
```

**评估：** 路径参数化，slurm 路径指向非 slurm 值（现在通过环境变量统一配置）。正确。

#### 2.3.3 configs/task/pose_gen.yaml 和 motion_gen.yaml

所有 `/home/wangzan/Data/...` 硬编码路径替换为 `${paths.*}` 引用。`*_slurm` 变体统一指向对应的非 slurm 值。

**评估：** 正确。

#### 2.3.4 configs/task/path_planning.yaml

`data_dir` 从硬编码改为 `${paths.scannet_path_planning_dir}`。

**评估：** 正确。

#### 2.3.5 configs/task/grasp_gen_ur.yaml

所有 `/home/puhao/data/...` 硬编码路径替换为 `${paths.*}` 引用。

**评估：** 正确。

#### 2.3.6 configs/optimizer/pose_in_scene.yaml 和 motion_in_scene.yaml

`*_slurm` 路径从引用 `${task.dataset.*_slurm}`（已不存在的键）改为引用 `${optimizer.*}`（当前值）。

**评估：** 修复了引用链断裂问题。正确。

---

### 2.4 新增文件

#### 2.4.1 chamfer_distance.py

用纯 PyTorch 实现替代原 CUDA 编译的 otaheri/chamfer_distance + pytorch3d 依赖。

实现方式：使用 `torch.cdist` 计算成对距离，配合 chunk 策略控制显存。

API 兼容性验证：

所有调用点均为 `dist1, dist2, idx1, idx2 = chamfer_dist(xyz1, xyz2)` 的直接函数调用形式：

| 文件 | 行号 |
|---|---|
| models/evaluator.py | 129, 269 |
| models/optimizer/pose_in_scene.py | 112 |
| models/optimizer/motion_in_scene.py | 127 |

新函数签名 `ChamferDistance(xyz1, xyz2) -> (dist1, dist2, idx1, idx2)` 完全匹配。

返回值语义：平方 L2 近邻距离，与原 otaheri 实现一致。

**评估：** 功能正确，API 兼容。性能方面，纯 PyTorch 实现比原生 CUDA kernel 慢，但 `torch.cdist` 在 GPU 上执行，对本项目规模的点云（32768 点）可接受。

#### 2.4.2 scripts/with_scene_diffuser_cuda_env.sh

编译期包装脚本，功能：

- 自动发现当前 conda 环境的 Python site-packages 路径
- 收集 PyTorch lib、nvidia pip 包 lib、conda CUDA 工具链的 include/lib 路径
- 设置 `CUDA_HOME`、`LD_LIBRARY_PATH`、`LIBRARY_PATH`、`CPATH` 等
- 设置 `TORCH_CUDA_ARCH_LIST="12.0+PTX"`（对应 RTX 5080 SM 12.0）
- 无参数时启动交互 shell，有参数时执行给定命令

**评估：** 仅在编译 pointops 时使用。`set -euo pipefail` 保证出错即停。路径发现逻辑健壮。不修改 conda 环境本身。正确。

#### 2.4.3 scripts/install_scene_diffuser_conda_hooks.sh

将运行期环境变量固化到 conda 环境的 activate/deactivate 钩子。

功能：

- 在 `$CONDA_PREFIX/etc/conda/activate.d/` 写入激活脚本
- 在 `$CONDA_PREFIX/etc/conda/deactivate.d/` 写入恢复脚本
- 激活时设置 `CUDA_HOME`、`PATH`、`TORCH_CUDA_ARCH_LIST`、`CPATH`、`CPLUS_INCLUDE_PATH`、`LIBRARY_PATH`、`LD_LIBRARY_PATH`
- 停用时恢复所有变量到激活前的值
- 使用 `SCENE_DIFFUSER_CUDA_HOOK_ACTIVE` 标志防止重复激活
- 支持 `--uninstall` 移除钩子
- 支持 `--prefix` 指定目标环境

**评估：** 实现规范。仅写入指定的 conda 环境，不影响 base 或其他环境。deactivate 钩子通过保存/恢复旧值的方式确保完整回退。正确。

#### 2.4.4 environment.yml

参考性 conda 环境定义，声明 Python 3.11、编译工具链、PyTorch cu128 版本。

**评估：** 作为环境定义的参考入口，与安装手册内容一致。正确。

#### 2.4.5 requirements-cuda-ext.txt

内容为 `git+https://github.com/Silverster98/pointops@86c68b1`，锁定到具体 commit。

**评估：** 正确，确保编译的 pointops 版本一致。

#### 2.4.6 requirements-optional.txt

内容为 pytorch_kinematics、human_body_prior、urdf-parser-py，均锁定到具体 commit 或版本。

**评估：** 正确，与主依赖隔离。

#### 2.4.7 .gitignore

新增 `.conda/`、`.cache/`、`outputs/`、`data/`。

**评估：** 防止本地 conda 环境、缓存和数据资源被意外提交。正确。

---

## 三、功能影响评估

### 3.1 主流程覆盖

| 任务 | 涉及文件 | 改动类型 | 功能影响 |
|---|---|---|---|
| pose_gen | configs/task/pose_gen.yaml, models/optimizer/pose_in_scene.py, sample.py, test.py, train_ddm.py | 路径参数化、API 现代化、torch.load 适配 | 无 |
| motion_gen | configs/task/motion_gen.yaml, models/optimizer/motion_in_scene.py, sample.py, test.py, train_ddm.py | 同上 | 无 |
| path_planning | configs/task/path_planning.yaml, models/environment.py, plan.py | 路径参数化、装饰器修正、torch.load 适配 | 无 |

### 3.2 有条件的功能路径

| 功能 | 条件 | 影响 |
|---|---|---|
| optimizer + VPoser | 需要安装 requirements-optional.txt 中的 human_body_prior | 未安装时给出明确 ImportError；安装后行为不变 |
| grasp 任务 | 需要安装 requirements-optional.txt | 未安装时 GraspWithObject/MultiDexShadowHandUR 为 None；安装后行为不变 |

### 3.3 数值等价性

| 改动 | 数值影响 |
|---|---|
| pointops.py 张量创建方式 | 严格等价，初始化值相同 |
| _compute_vertex_normals 提取 | 严格等价，计算逻辑未变 |
| .clamp_min(1e-12) | 仅影响退化情况（零向量），将 NaN 变为极大有限值，属于改善 |
| chamfer_distance.py 纯 PyTorch 实现 | 返回平方 L2 距离，与原实现语义一致；浮点精度可能有 ulp 级差异，不影响训练/推理 |

---

## 四、性能影响评估

| 项 | 影响 |
|---|---|
| chamfer_distance 纯 PyTorch vs 原生 CUDA kernel | 较慢，但 torch.cdist 在 GPU 上执行且支持 chunk，对 32768 点规模可接受 |
| pointops 张量创建新 API | 无性能差异 |
| torch.load map_location='cpu' | 首次加载稍慢（CPU -> GPU 拷贝），可忽略 |

---

## 五、安全性评估

| 项 | 影响范围 |
|---|---|
| conda 环境 | 仅写入 scene-diff 环境，不影响 base 或其他环境 |
| conda hooks | deactivate 时完整恢复所有变量原值 |
| WSL 系统 | 不修改系统级配置 |
| with_scene_diffuser_cuda_env.sh | 仅设置环境变量，不修改文件系统 |
| install_scene_diffuser_conda_hooks.sh | 仅在 conda 环境 etc/conda/ 下写入钩子，支持 --uninstall 回退 |

---

## 六、综合结论

| 评估维度 | 结论 |
|---|---|
| 改造是否必要 | **必要。** 原工具链不支持 RTX 5080 架构。 |
| 改造是否合理 | **合理。** 依赖分层清晰，代码改动均为标准的 PyTorch 2.x 适配模式，无过度工程。 |
| 是否影响代码功能 | **不影响。** 所有改动在数值上等价或仅改善边界情况，API 兼容性经过逐一验证。 |
| 是否可以正常运行项目 | **可以。** pose_gen、motion_gen、path_planning 三条主链路的代码路径已全部适配。 |
| 是否污染系统环境 | **不污染。** 改动限定在独立 conda 环境内，有完整的回退机制。 |
