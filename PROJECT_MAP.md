<!--
  Current evidence gates for the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# PROJECT MAP

| Metadata | Value |
|---|---|
| Status | Draft current state snapshot pending human review |
| Audience | 导师、研究生与项目负责人 |
| Owner | 研究生维护，导师审阅科学主张 |
| Last reviewed | 2026-10-21 |
| Reading time | 5 minutes |

**当前工作已形成全胞正向的多域测试、分支依赖与固定预算确认链；最新训练尚未全部完成，单孔数据尚待 WSL 验收。** 本页是当前状态入口，不保存调度流水账。原始配置、日志、模型、哈希与求解器证据继续留在 runs；历史结论通过结果页追溯。

[项目入口](README.md) · [文档目录](docs/README.md) · [当前结果](docs/结果/README.md) · [实验登记](experiments/registry.yaml) · [协作治理](docs/collaboration.md)

工程入口：[架构兼容入口](docs/architecture.md) · [数据目录](data/README.md)。

## 当前研究问题

项目研究如何在明确的数据、模型和证据合同下，用神经代理加速二维光子晶体能带计算，并将候选结构交给 real-MPB/Meep 复核。低 surrogate 误差、代理排序、真实 replay 和物理成功是不同结论。

当前重点是全胞正向模型在 K 截断、Binary/Graded 路线与结构分布变化下的可靠性。Full-cell inverse 和 Field/PINO 的未通过门不因正向更新而解锁。

## 最新证据与最近一门

| Experiment ID | 已确认状态与边界 | 下一 gate |
|---|---|---|
| EXP-FWD-020 | 36 个 checkpoint、216-cell 六域 held-out 评估完成；G2 `not_passed`，仅探索性 surrogate | 人工接受[六域结果 Draft](docs/results/forward/EXP-FWD-020-全局局部带隙算子六域交叉评估.md)，不作 promotion |
| EXP-FWD-021/023 | 63 个固定 seed0 checkpoint、756/756 十二域 cells 与 7,560 条逐 band 记录；K8/K10/K15 从未进入训练、validation、normalizer 或选模 | [完整结果](docs/results/forward/EXP-FWD-023-七种训练制度十二域完整结果.md)人工审阅；新数据仍为 `cluster_audited_pending_wsl_acceptance` |
| EXP-FWD-022 | 1,080 冻结 validation 干预与 30 项独立训练完成；仅 11/30 过末窗收敛门，无 test | 人工审阅[依赖与独立能力](docs/results/forward/EXP-FWD-022-分支依赖与独立能力.md)，不解释为充分优化能力上限 |
| EXP-FWD-024 | 2026-10-21 只读核验：54 项中 44 个训练终态，11 个收敛、33 个未过末窗门；4/6 bundle 标记，未见 test manifest | 按[冻结合同](experiments/research/EXP-FWD-024-三核心模型固定预算确认.md)等待完整训练审计后才可 native test；现有数字不是最终结果 |
| EXP-FWD-025 | 同次核验：30 项中 9 个训练终态，3 个收敛、6 个未过末窗门；无 test manifest，test 保持锁定 | 等待全部训练和完整性审计；不与旧 200 epoch 结果混用 |
| EXP-DAT-007 | Bingxing 四 shard 合计 12,000/12,000，0 failure，`cluster_generated_pending_wsl_acceptance`；分辨率 pilot 为 `waived_by_user` | [单孔合同](experiments/research/EXP-DAT-007-单孔形状迁移测试.md)要求下载哈希核对与 WSL 跨环境验收；尚无新模型迁移结论 |

024/025 的现有证据定位为 N40 `runs/20261020/EXP-FWD-024_202610201920` 与 `EXP-FWD-025_202610201920` 下的训练终态、admission audit 和 bundle 标记。007 的证据定位为 Bingxing `runs/20261020/EXP-DAT-007_12000_20261020_bingxing_4x64_v1` 下的 results、metrics、campaign_status 和 shard 记录。这些是只读核验时的快照，不代表此后运行状态不变。

## 保留的研究边界

| 方向 | 现有结论 | 限制与入口 |
|---|---|---|
| 全胞数据底座 | 已有分布的 r64 通过数值门，历史 B/G 数据已有验收链 | 新单孔分布不能继承旧分布的精度结论；见[全胞数据](docs/results/全胞数据正式结果.md) |
| 历史 50k 正向 | ConvNeXt 的 B/G native 结果属于其明确模型和冻结协议 | [升级结果](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)不能借给 Global–Local；final-fit monitor 不能代替 test |
| 组件与 gap 决策 | 组件删除存在容量混杂，gap loss 有短预算负结果 | [组件结果](docs/results/forward/EXP-FWD-019-组件归因三种子确认.md)、[gap loss](docs/results/forward/EXP-FWD-014-gap决策损失短预算筛选.md)；不自动调参或用 test 追调 |
| 标签与物理选频 | 粗网格可高估 gap；低全谱 MRE 不保证 band/k 选择正确 | [标签门](docs/results/field-physics/EXP-PHY-007-validation带隙标签门.md)、[真实响应差异](docs/results/field-physics/代理选频与真实响应差异_20260825.md) |
| Field / adapted PINO | 校准与准入仍 blocked | [历史准入](docs/results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md)；不进入候选排序 |
| 生成式 inverse | V2 Objective-Bridge 仅有 seed0 replay；full-cell 正式闭环尚未完成 | [inverse 结果](docs/results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md)、[当前协议](docs/planning/全胞逆向索引.md) |

## 实现与交付入口

- 代码、CLI、配置与 tests 证明实现合同；正式工程验收使用 WSL Ubuntu / meep_env，不能自动升级科学证据。
- 模型必须同时核对[模型目录](docs/参考/模型目录.md)、[机器清单](docs/reports/phc2d_model_inventory_v1.yaml)和未注册的 inverse 方法。注册、训练、验证、test 与物理验收保持分开。
- 图件、编辑源与来源见[图件索引](docs/reports/图件索引.md)；历史 Bands 2–10 图不能混作 024 的 Bands 1–10 主表。

## 当前人工决策

人工审阅已有 Draft 结果。研究侧最近一门是完整训练审计和新单孔数据验收；任何新训练、正式物理验收或更强主张均须按各自合同推进，不能由文档整理替代。
