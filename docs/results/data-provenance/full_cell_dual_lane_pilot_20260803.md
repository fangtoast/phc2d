<!--
  Full-cell dual-lane dataset implementation and pilot report.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# 非 p4m 全胞双路线数据生成：实现与试点报告

- 报告日期：2026-08-03
- 代码提交：`662df59`（主体实现）、`1032e93`（恢复分片身份校验）
- 正式环境：WSL Ubuntu + conda `meep_env`
- 求解器：Meep `1.33.0-beta`，MPB `1.12.0`
- 实验协议：[full_cell_dual_lane_data.md](../../../experiments/research/full_cell_dual_lane_data.md)
- 当前证据状态：生成规则调整后，旧 64+64 任务已作废；新规则每路线 4 样本真实 MPB 前置门已通过，正式 64+64 与分辨率收敛尚未完成

## 执行结论

双路线的代码、配置、CLI、HDF5、并行分片恢复和 64/96/128 收敛门已经实现。真实 MPB oracle 证明新解析材料函数与原生圆孔几何一致：bands 2+ MRE 为 `0.0002083`（约 `0.0208%`），同一圆孔重复求解最大差 `1.19×10⁻⁷`。

连续路线和二值路线随后各完成 4 个结构、81 个全 BZ k 点、10 个保留频带的端到端前置验证，8 个结构全部成功、失败率为 0。生成的 HDF5 均被 schema inspector 识别为 `hdf5_full_cell_band`，形状为 `[4,81,10]`，并包含 `32/64/128` 三种结构栅格。

这组结果只证明“参数化结构 → 真实 MPB → 可追溯 HDF5”的链路可用，不证明 64 样本数据集已完成，也不证明 `resolution=64` 已通过收敛门，更不涉及 surrogate 或 inverse-design 收益。

## 数据与参数化

| 路线 | 前置样本构成 | 结构真值 | 图像用途 |
|---|---|---|---|
| 连续渐变介质 | 2 个单核、2 个双核 | 周期高斯/RBF + 低阶 Fourier + sigmoid 的解析参数 | 同一参数在 `32/64/128` 重采样 |
| 二值空气孔 | 1 个单极坐标孔、1 个双孔、2 个低通阈值场 | 极坐标 Fourier 孔或周期低通场及阈值 | 严格 `{0,1}` 栅格，用于检查和模型输入 |

介电常数为 `epsilon=1+7.9u`。没有执行 p4m 楔形展开、旋转复制或镜像复制；第一布里渊区使用 `9×9` 完整方格。

## MPB 配置

| 字段 | 值 |
|---|---|
| 晶格 / 周期 | square / `a=1` |
| k 点 | `81`，索引 `(-4…4,-4…4)/9` |
| 极化 | TE |
| 介电常数 | air `1.0`，material `8.9` |
| 分辨率 / mesh | `64 / 3` |
| 带数 | 求解 `14`，升序保留 `10` |
| eigensolver tolerance | `1×10⁻⁷` |
| 代理求解器 | 禁用 |

## 已执行命令

所有正式命令均在 WSL `meep_env` 中运行：

```bash
PYTHONPATH=src python -m phc2d.cli data generate \
  --config configs/research/full_cell_solver_preflight_v1.yaml

PYTHONPATH=src python -m phc2d.cli data generate \
  --config configs/research/full_cell_graded_bz9_pilot64.yaml \
  --max-samples 4

PYTHONPATH=src python -m phc2d.cli data generate \
  --config configs/research/full_cell_binary_holes_bz9_pilot64.yaml \
  --max-samples 4
```

正式 64+64 命令曾于 2026-08-03 14:27 启动，对应运行目录：

- `runs/20260803/full_cell_graded_bz9_pilot64_202608031427/`
- `runs/20260803/full_cell_binary_holes_bz9_pilot64_202608031427/`

16:57 按新的生成规则停止了上述任务和自动收尾进程：连续路线已完成 `32/64` 个分片，二值路线已完成 `28/64` 个分片。两个运行目录均保留 `superseded.json`，明确标记为不可晋级、不可与新规则样本混合；未写正式联合清单。

新规则把连续介质的目标平均材料权重调整为 `0.60–0.82`，并把二值频谱场的截止尺度由原先的 `1.5–3.0` 提高到 `2.5–4.5`、Fourier 索引上限由 `6` 提高到 `8`。确定性 64 样本结构检查得到：连续路线材料权重的最小值/中位数/最大值为 `0.6062/0.6946/0.8034`，深材料区域占比中位数为 `0.7513`；全部样本仍满足连续性与高频能量门。

新规则每路线 4 样本真实 MPB 前置门均于 17:29 完成，状态为 `real_mpb_preflight_pass`，完成/失败均为 `4/0`，`Wh` 形状均为 `[4,81,10]`。连续路线墙钟 `1150.94 s`、单样本累计求解时间 `567.18 s`，HDF5 SHA-256 为 `a1a58526…9cc0bf82`；二值路线墙钟 `1136.86 s`、单样本累计求解时间 `560.00 s`，HDF5 SHA-256 为 `40c25711…f63a050`。证据目录分别为 `runs/20260803/full_cell_graded_newrules_preflight4_202608031708/` 与 `runs/20260803/full_cell_binary_newrules_preflight4_202608031709/`。

前置门通过后，新规则正式 64+64 于 17:31 启动，固定运行目录为 `runs/20260803/full_cell_graded_newrules_pilot64_202608031735/` 和 `runs/20260803/full_cell_binary_newrules_pilot64_202608031735/`。fail-closed 收尾进程同步启动并写入 `runs/full_cell_newrules_finalize_202608031735.log`；它只在两个正式 HDF5 完整且收敛门通过后写联合清单。

17:34 的视觉复核发现连续路线的浅色区仍被高均值偏置托高。随后把正振幅材料峰改成负振幅空气势阱，并收紧 sigmoid 温度；上述第二轮正式任务在各完成 `6/64` 个分片后停止，两个目录均写入 `superseded.json`，不可用于当前数据集。最终冻结版确定性 64 样本检查中，平均材料权重最小值/中位数/最大值为 `0.6050/0.6844/0.7734`，`u≤0.10` 区域占比为 `0.0361/0.1271/0.2491`，最小权重中位数为 `0.00032`。

用户确认恢复该版后，两条最终 64 样本真实 MPB 任务于 18:22 启动，运行目录为 `runs/20260803/full_cell_graded_final64_202608031822/` 和 `runs/20260803/full_cell_binary_final64_202608031822/`。两边均已写出 64 样本计划并进入真实 MPB 求解。自动收尾日志为 `runs/full_cell_final64_finalize_202608031822.log`；只有两个 HDF5 完整且收敛门通过后才允许写联合清单。

## 结果

### 求解器 oracle

证据目录：`runs/20260803/full_cell_solver_preflight_v1_202608031352/`

| 指标 | 结果 | 门限 | 状态 |
|---|---:|---:|---|
| uniform air median absolute difference | `0` | `≤1×10⁻⁶` | 通过 |
| uniform material median absolute difference | `0` | `≤1×10⁻⁶` | 通过 |
| uniform air match fraction at `1×10⁻⁵` | `92.22%` | `≥75%` | 通过 |
| uniform material match fraction at `1×10⁻⁵` | `87.78%` | `≥75%` | 通过 |
| circle repeat max absolute difference | `1.19×10⁻⁷` | `≤1×10⁻⁶` | 通过 |
| circle material-function/native MRE, bands 2+ | `0.02083%` | `≤0.5%` | 通过 |

均匀介质的最大逐带差仍记录为诊断量：air `0.1009`、material `0.0963`。误差集中在精确简并谱的少数带位置，MPB 在不同求解中可能返回不同简并向量/子集；因此该量不作为逐带有序谱门禁。中位差、匹配比例、非均匀圆孔原生对照和重复性共同构成当前 oracle。

### 双路线 4 样本前置验证

| 路线 | 状态 | `Wh` | 完成/失败 | 墙钟 | 单样本累计求解时间 | HDF5 SHA-256 |
|---|---|---:|---:|---:|---:|---|
| 连续 | `real_mpb_preflight_pass` | `[4,81,10]` | `4/0` | `1145.69 s` | `541.94 s` | `8a064695…b551d8fa5` |
| 二值 | `real_mpb_preflight_pass` | `[4,81,10]` | `4/0` | `1184.95 s` | `589.61 s` | `ad284069…c9da585` |

连续证据目录：`runs/20260803/full_cell_graded_bz9_pilot64_202608031353/`。

二值证据目录：`runs/20260803/full_cell_binary_holes_bz9_pilot64_202608031353/`。

两条路线的 `failed_case_ledger.jsonl` 均为空。其清单、solver provenance、sample metadata、预览图和 worker 分片均保存在各自目录中。

## 测试

在 WSL `meep_env` 执行完整回归：

```text
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src python -m pytest -q tests
PYTHONPATH=src python -m phc2d.cli --help
PYTHONPATH=src python -m phc2d.cli plot --help
```

结果：`290 passed in 61.22s`。在补充恢复分片 hash/identity 校验后，相关 11 项测试再次通过。

## 失败记录与修正

1. 首次 oracle 运行 `runs/20260803/full_cell_solver_preflight_v1_202608031349/` 被门禁阻止。原因是对均匀介质精确简并谱使用逐带最大差，少数简并带的返回顺序/子集造成假阳性。圆孔对照在该次运行中已经达到 `0.0208%` MRE。修正后保留最大差作为诊断，并以稳健谱匹配加圆孔原生对照作为门禁。
2. 初次 Windows 后台启动因命令参数转义失败，没有进入 Python 或 MPB；错误只存在于 `runs/full_cell_*_launch_20260803142643.stderr.log`。14:27 修正启动后，两个正式任务和 4 个 MPB worker 均正常运行。
3. 完整回归首次出现 1 个旧 A2 协议哈希失败。原因是把全胞专用 `mesh_size` 加入旧基础配置数据类。现已改为全胞专用子类型，旧协议哈希不变；随后完整 `290/290` 通过。

## 当前边界与下一门

截至本报告写入时，旧规则 64+64 正式试点已作废，不能将旧规则的已完成分片或 4 样本 HDF5 视为新规则训练集。下一步还必须：

1. 重启两条正式数据，并审计 `64/64` 完成率、失败账本、`[64,81,10]` 形状和 hash；
2. 运行 [full_cell_resolution_convergence_v1.yaml](../../../configs/research/full_cell_resolution_convergence_v1.yaml)；
3. 要求每条路线及合并池同时通过 bands 2+ 的中位与 p95 阈值；
4. 仅在上述条件全部满足后写联合清单，且联合清单只链接两个独立 HDF5，不合并材料语义。
