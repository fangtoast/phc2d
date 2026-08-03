<!--
  Non-p4m full-cell dual-lane MPB dataset protocol.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 非 p4m 全胞双路线 MPB 数据实验

本实验建立两条物理含义独立的数据路线：连续渐变介质与二值介质/空气孔。两条路线共享全胞坐标、全布里渊区和 MPB 标签协议，但不合并数组、不混用材料语义；联合清单只链接各自通过门禁的数据集。

## 研究问题

1. 去除 p4m 投影后，是否可以用参数化解析函数稳定表达完整周期胞？
2. 连续 `0–1` 材料权重与严格二值空气孔能否通过同一真实 MPB 材料函数获得可追溯的频带标签？
3. `64` 求解分辨率相对 `96/128` 是否达到冻结的收敛阈值？

本实验不评估 surrogate 精度或 inverse-design 收益，也不把图像输入当作 MPB 几何真值。保存的多分辨率图像均由同一组解析参数重新栅格化。

## 双路线参数化

| 路线 | 生成家族 | 参数化 | 材料权重 |
|---|---|---|---|
| 连续渐变介质 | `graded_rbf_fourier` | 1/2 个周期椭圆高斯核，加低阶实 Fourier 模式，再经带温度的 sigmoid | 连续 `u(x,y)∈[0,1]` |
| 二值空气孔 | `binary_polar_holes` | 1/2 个极坐标 Fourier 星形孔，孔内为空气 | 严格 `{0,1}` |
| 二值随机形状 | `binary_spectral_threshold` | 周期低通随机场，分位数阈值含小扰动 | 严格 `{0,1}` |

介电常数统一按

```text
epsilon(x,y) = 1.0 + (8.9 - 1.0) * u(x,y)
```

构造。所有解析函数以周期包裹坐标求值，不执行旋转、镜像或楔形复制。

## 冻结数据与求解协议

| 项目 | 冻结值 |
|---|---|
| 每条路线样本数 | `64` |
| 切分 | `48/8/8`，按生成家族分层且由 `sample_id` 稳定决定 |
| 栅格 | `32×32`、`64×64`、`128×128`，由同一解析参数独立求值 |
| 晶格 | 正方晶格，`a=1` |
| k 网格 | 第一布里渊区 `9×9` 完整方格，共 `81` 个唯一点 |
| k 坐标 | MPB reduced reciprocal coordinates，整数索引 `-4…4` 除以 `9` |
| 极化 | `TE` |
| 频带 | 求解 `14` 带，升序后保留前 `10` 带 |
| MPB | `resolution=64`、`mesh_size=3`、`tolerance=1e-7` |
| 正式环境 | WSL Ubuntu + conda `meep_env`，真实 `meep.mpb.ModeSolver` |

对应配置：

- [求解器 oracle 前置验证](../../configs/research/full_cell_solver_preflight_v1.yaml)
- [连续路线 64 样本](../../configs/research/full_cell_graded_bz9_pilot64.yaml)
- [二值路线 64 样本](../../configs/research/full_cell_binary_holes_bz9_pilot64.yaml)
- [64/96/128 分辨率收敛](../../configs/research/full_cell_resolution_convergence_v1.yaml)

## HDF5 契约

每条路线独立写入 `phc2d.full_cell.v1` HDF5，主要字段如下：

| 字段 | 形状/内容 |
|---|---|
| `structure_json` | 可重建解析几何、参数名、参数向量、材料常数与统计量 |
| `params`, `param_mask` | 补齐后的参数向量及有效位掩码 |
| `material_weight_grid_{32,64,128}` | 同一结构在三种网格上的材料权重 |
| `epsilon_grid_{32,64,128}` | 对应介电常数图 |
| `Wh` | `[N,81,10]`，MPB 无量纲频率 `a/λ` |
| `k_points`, `k_grid_index` | 物理 reduced coordinates 与整数网格索引 |
| `idx_train`, `idx_val`, `idx_test` | 互斥且覆盖全部样本的固定切分 |

`structure_json` 是 MPB 几何的参数化真值；栅格只用于检查、可视化和后续模型输入。

## 门禁顺序

1. `solver_preflight`：均匀材料稳健谱一致性、圆孔原生几何对照和重复性。
2. 每条路线 `max_samples=4`：验证 81 k 点完整求解、分片恢复、HDF5 和元数据链路。
3. 每条路线生成 `64` 个结构；状态仅为 `pilot_generated_pending_convergence`。
4. 每条路线分层选择 `8` 个结构，在 9 个代表性 k 点重算 `96/128`。
5. 对每条路线及合并池分别要求：bands 2+ 的 `median MRE(64→96)≤0.5%`、`median MRE(96→128)≤0.25%`，两段 `p95≤1%`。
6. 只有两条数据集完整且收敛门通过，联合清单才可标记为可晋级试点。

任何测试注入求解器或跳过环境检查的运行只能标记为 `test_solver_smoke_only`，不得成为正式 MPB 证据。

## 执行入口

```bash
PYTHONPATH=src python -m phc2d.cli data generate --config configs/research/full_cell_solver_preflight_v1.yaml
PYTHONPATH=src python -m phc2d.cli data generate --config configs/research/full_cell_graded_bz9_pilot64.yaml --max-samples 4
PYTHONPATH=src python -m phc2d.cli data generate --config configs/research/full_cell_binary_holes_bz9_pilot64.yaml --max-samples 4
PYTHONPATH=src python -m phc2d.cli data generate --config configs/research/full_cell_graded_bz9_pilot64.yaml
PYTHONPATH=src python -m phc2d.cli data generate --config configs/research/full_cell_binary_holes_bz9_pilot64.yaml
PYTHONPATH=src python -m phc2d.cli data converge --config configs/research/full_cell_resolution_convergence_v1.yaml
PYTHONPATH=src python -m phc2d.cli data finalize \
  --graded-run-dir <graded-run-dir> \
  --binary-run-dir <binary-run-dir> \
  --convergence-config configs/research/full_cell_resolution_convergence_v1.yaml \
  --combined-manifest data/full_cell_dual_lane_pilot64_manifest.yaml
```

`data finalize` 只用于长任务的 fail-closed 收尾：等待两个明确指定的正式 run，任一路线失败即停止；两边完成后才运行收敛门，且只有收敛通过才写联合清单。正式命令必须在仓库规定的 WSL `meep_env` 中执行。Windows 本地测试只用于代码调试，不构成验收结果。
