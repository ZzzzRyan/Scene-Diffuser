# 环境修正内容与评估报告_精简

本报告对 Scene-Diffuser 仓库为适配 RTX 5080 + CUDA 12.8 所做的全部改造进行**简要**评估。

详细修正内容与评估报告参见：[环境修正内容与评估报告_详细](./env_modification_report_detail.md)。

报告生成时间：2026-03-26

## 总结：改造必要、合理、不影响功能

## 一、改造是否必要？

**必要。** 原项目锁定 `PyTorch 1.11 + CUDA 11.3`，而 RTX 5080 是 Blackwell 架构（SM 12.0），**CUDA 11.3 根本不认识这个架构**。你必须升级到 CUDA 12.8 + PyTorch 2.x 才能驱动这张卡。这不是"可选优化"，是"不改就跑不了"。

---

## 二、依赖改造分析

### 2.1 requirements.txt 拆分策略

| 原始做法 | 你的做法 | 评价 |
|---|---|---|
| 所有依赖混在一个 requirements.txt | 拆为 core / cuda-ext / optional 三层 | **合理**，核心依赖与编译依赖、可选依赖解耦 |
| `pytorch3d v0.7.0` 从源码编译 | 移除，用本地 `chamfer_distance.py` 替代 | **合理**，pytorch3d 在 PyTorch 2.x + CUDA 12.8 下编译极其困难 |
| `human_body_prior` 在主依赖中 | 移到 optional，代码中 try/except 保护 | **合理**，主流程（pose_gen/motion_gen/path_planning）不开 VPoser 时不需要 |
| `pytorch_kinematics`、`urdf-parser-py` 在主依赖中 | 移到 optional | **合理**，仅 grasp/hand 分支需要 |
| `pointops` 在 requirements.txt | 移到 `requirements-cuda-ext.txt` 单独管理 | **合理**，需要 nvcc 编译，和普通 pip 依赖分离 |
| `scikit-learn` 缺失 | 显式添加 | **正确**，`models/evaluator.py` 和预处理脚本都在用，原来是被其他包附带安装的 |

### 2.2 版本升级

纯 Python 包（einops、hydra-core、matplotlib、omegaconf 等）全部从 2022 年版本升级到当前版本，并使用区间约束而非精确锁定。这些库的 API 在大版本内保持兼容，**没有问题**。关键锁定项（`smplx==0.1.28`、`pyquaternion==0.9.9`、`pyrender==0.1.45`）保持不动，也是正确的。

---

## 三、代码改动分析

### 3.1 pointops.py — 最关键的改动

**改了什么：**
- 所有 `torch.cuda.FloatTensor()` / `torch.cuda.IntTensor()` → `torch.zeros()`/`torch.full()`/`torch.empty()`/`torch.tensor()` + `device=` 参数
- 新增 `_load_torch_shared_libs()`，在 `import pointops_cuda` 前预加载 PyTorch 动态库

**是否正确：**
- `torch.cuda.FloatTensor` 在 PyTorch 2.x 已废弃，新写法是官方推荐的现代 API，**语义完全等价**
- 新写法额外保留了输入张量的 dtype（原代码强制 float32/int32，新代码用 `dtype=input.dtype`），这在 float16/bfloat16 混合精度场景更健壮
- `_load_torch_shared_libs()` 是自定义 CUDA 扩展在新版 PyTorch 下的已知必要手段，**正确**

### 3.2 torch.load 适配

`plan.py`、`sample.py`、`test.py`、`pointnet.py`、`pointtransformer.py` 中所有 `torch.load(path)` → `torch.load(path, map_location='cpu', weights_only=False)`

- `weights_only=False`：PyTorch >= 2.0 默认行为变了，不加会有 FutureWarning，对于加载自有 checkpoint 是安全的
- `map_location='cpu'`：先加载到 CPU 再迁移，避免 GPU 内存不足或设备编号不匹配

**正确，标准做法。**

### 3.3 torch.cuda.manual_seed 保护

```python
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
```

**正确**，防御性编程，不影响有 GPU 时的行为。

### 3.4 @torch.no_grad() 从类定义上移除

你移除了 `PoseGenVisualizer`、`MotionGenVisualizer`、`PathPlanningEnvWrapper` 等**类**上的 `@torch.no_grad()` 装饰器。

**正确。** `@torch.no_grad()` 用在类定义上是无效的（它只对函数/方法生效）。我确认了 `evaluator.py` 和 `ddpm.py` 中方法级别的 `@torch.no_grad()` 保留完好，这些才是真正起作用的。

### 3.5 vertex normal 计算提取为公共函数

`pose_in_scene.py`、`motion_in_scene.py`、`smplx_utils.py` 中重复的顶点法线计算代码提取到 `utils/smplx_utils.py:_compute_vertex_normals()`。

改进：
- 消除了代码重复
- `torch.zeros(shape).float().cuda()` → `torch.zeros_like(vertices)`，跟随输入设备，不再硬编码 `.cuda()`
- `.clamp_min(1e-12)` 防止除零产生 NaN
- `torch.cross(e1, e2)` → `torch.cross(e1, e2, dim=-1)`，消除 PyTorch 2.x 的弃用警告

**正确，数值等价且更健壮。**

### 3.6 chamfer_distance.py 替代方案

用纯 PyTorch（`torch.cdist`）实现替代原 CUDA 编译的 otaheri 版本。

我确认了所有调用点（`evaluator.py:129`、`evaluator.py:269`、`pose_in_scene.py:112`、`motion_in_scene.py:127`）的调用方式都是：
```python
dist1, dist2, idx1, idx2 = chamfer_dist(xyz1, xyz2)
```

新函数签名 `ChamferDistance(xyz1, xyz2) -> (dist1, dist2, idx1, idx2)` **完全匹配**。

**注意点：** 纯 PyTorch 实现会比原生 CUDA kernel 慢一些，但 `torch.cdist` 本身在 GPU 上运行，配合 chunk 策略控制显存，对于本项目规模的点云（32768 点）可接受。

### 3.7 可选依赖的 lazy import

- `models/__init__.py`：`PoseInSceneOptimizer`、`GraspWithObject`、`MotionInSceneOptimizer` 用 try/except 包裹
- `datasets/__init__.py`：`MultiDexShadowHandUR` 用 try/except 包裹
- `models/visualizer.py`：`get_handmodel` 从顶层 import 移到 `GraspGenVisualizer.__init__()` 内部

**合理。** 确保不安装可选依赖时主流程不会在 import 阶段崩溃，且在真正需要时（如开启 VPoser）给出清晰的 `ImportError` 提示。

### 3.8 配置路径参数化

所有 `/home/wangzan/...` 硬编码路径替换为 Hydra/OmegaConf 插值 + 环境变量覆盖：
```yaml
data_dir: ${paths.lemo_proxd_dir}
data_dir_slurm: ${task.dataset.data_dir}
```

`*_slurm` 路径统一指向非 slurm 的值（因为现在都通过环境变量配置），**逻辑正确**。

---

## 四、新增基础设施

### 4.1 `scripts/with_scene_diffuser_cuda_env.sh`
编译期包装脚本，设置 `CUDA_HOME`、`LD_LIBRARY_PATH` 等，仅在 `pip install -r requirements-cuda-ext.txt` 时需要。**实现规范**，使用 `set -euo pipefail`，路径发现逻辑健壮。

### 4.2 `scripts/install_scene_diffuser_conda_hooks.sh`
将运行期环境变量写入 conda 的 `activate.d/deactivate.d`。deactivate 时正确恢复所有变量原值。支持 `--uninstall`。**实现很干净**，不会污染 base 或其他 conda 环境。

### 4.3 `TORCH_CUDA_ARCH_LIST="12.0+PTX"`
对 RTX 5080（SM 12.0）**完全正确**，`+PTX` 确保前向兼容。

---

## 五、有无遗漏或隐患？

| 项目 | 状态 |
|---|---|
| 主流程三条链（pose_gen / motion_gen / path_planning）的代码路径 | 全部适配到位 |
| optimizer 的 VPoser 模式 | 有 guard，开启时若缺依赖会报明确错误 |
| grasp 分支 | 依赖隔离在 optional，不影响主流程 |
| chamfer_distance 性能 | 纯 PyTorch 比原生 CUDA kernel 慢，但不影响正确性 |
| conda hooks 对其他环境的影响 | 仅写入 `scene-diff` 环境，deactivate 完整恢复，**不污染** |

---

## 结论

**改造必要、设计合理、不影响代码功能。** 可以正常运行 pose_gen、motion_gen、path_planning 三条主链。