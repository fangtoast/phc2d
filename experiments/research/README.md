<!--
  Complete index of active phc2d research experiment profiles.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Research Profiles

本目录描述研究实验的输入、非目标、执行入口和 gate，不承诺当前结果。公共实现进入 [`../../src/phc2d`](../../src/phc2d)，结果统一从 [`../../docs/results/README.md`](../../docs/results/README.md) 读取。

## 数据与表征

- [`full_cell_k5_dual_lane_xingyi.md`](full_cell_k5_dual_lane_xingyi.md)：EXP-DAT-005 K=5 binary/graded 各 10k 的 Xingyi 生成、审计与本地/N40 交付合同。
- [`full_cell_dual_lane_data.md`](full_cell_dual_lane_data.md)：非 p4m 全胞双路线 real-MPB 数据协议。
- [`full_cell_graded_paracloud_m9.md`](full_cell_graded_paracloud_m9.md)：M9 渐变 10k 生成与 WSL 跨环境验收合同。
- [`full_cell_dual_lane_50000_xingyi.md`](full_cell_dual_lane_50000_xingyi.md)：Xingyi 二值/连续各 50k 的计划冻结、双节点提交、恢复与 WSL 验收合同。
- [`full_cell_resolution_convergence_xingyi.md`](full_cell_resolution_convergence_xingyi.md)：Xingyi 全胞 MPB 分辨率阶段 A 的 16 结构资源 pilot 与阶段 B 停止门。
- [`full_cell_dual_lane_sample_gallery.md`](full_cell_dual_lane_sample_gallery.md)：可追溯样本图册。
- [`representation_ablation.md`](representation_ablation.md)：输入表示消融。
- [`inverse_scale_preflight.md`](inverse_scale_preflight.md)：扩容前的资源、数据与 gate 预检。

## Forward、Query 与 Field

- [`a1_fair_forward_query.md`](a1_fair_forward_query.md)：A1 Full-Wh / Query 公平对照。
- [`full_cell_k20_forward_seed0.md`](full_cell_k20_forward_seed0.md)：K=20 B/G 单 seed 正向训练、34-cell native/cross-lane 评估与 N40 分波提交合同。
- [`EXP-FWD-007-K5训练与六域验证.md`](EXP-FWD-007-K5训练与六域验证.md)：K5 全模型训练、K5/K20/F50 六域 510-cell 评估与 N40 授权门。
- [`EXP-FWD-014-gap决策损失短预算筛选.md`](EXP-FWD-014-gap决策损失短预算筛选.md)：公平 K5 4k+4k 的 baseline/gap-aware seed0、20 epoch N40 配对筛选；只读 validation，不访问 test。
- [`EXP-FWD-015-gap决策近简并审计.md`](EXP-FWD-015-gap决策近简并审计.md)：复用 EXP-FWD-014 冻结检查点，区分近简并评价效应与模型决策能力。
- [`EXP-FWD-016-gap排序与完整带隙短预算筛选.md`](EXP-FWD-016-gap排序与完整带隙短预算筛选.md)：一个待授权的 N40 ranked-v2 seed0 短筛任务；通过前不进入多 seed 或 test。
- [`full_cell_forward_upgrade_10k.md`](full_cell_forward_upgrade_10k.md)：五个 Full-Wh 升级模型在 N40 的 admission 与 10k 七波训练合同。
- [`field_eigen_te.md`](field_eigen_te.md)：TE Bloch eigenfield、Field FNO 与 adapted PINO。
- [`near_field_surrogate.md`](near_field_surrogate.md)：near-field、NeurOLight 与 Maxwell residual 可行性边界。
- [`EXP-PHY-004-代理选频响应差异.md`](EXP-PHY-004-代理选频响应差异.md)：K5 Full-Wh 全量频谱误差、最大带隙选频稳定性与代表样本 real-MPB/Meep 响应差异。
- [`EXP-PHY-005-TE有效折射率与折叠频率校准.md`](EXP-PHY-005-TE有效折射率与折叠频率校准.md)：在不访问 test、不训练的合同下，用 near-Γ real-MPB 斜率校准 TE 有效折射率并审计全部相邻 gap。
- [`EXP-PHY-006-全BZ带隙边缘加密试点.md`](EXP-PHY-006-全BZ带隙边缘加密试点.md)：四个 validation 结构的局部 full-BZ band-edge 加密 pilot；先判断 9×9 标签风险，不等同于全局 gap 认证。

## Inverse 与 replay

- [`generative_inverse.md`](generative_inverse.md)：条件生成式 inverse 的候选/replay 协议历史入口；当前 mixture conditional generator 不是 cVAE 或 diffusion。
- [`a2_inverse_single_seed.md`](a2_inverse_single_seed.md)：A2 单 seed benchmark 与分阶段恢复语义。
- [`a2_inverse_objective_bridge_seed0_v1.md`](a2_inverse_objective_bridge_seed0_v1.md)：Objective-Bridge 100-target seed0。
- [`a2_inverse_objective_bridge_seed0_test625_v1.md`](a2_inverse_objective_bridge_seed0_test625_v1.md)：Objective-Bridge test625 replay。
- [`a2_inverse_surrogate_ablation_v1.md`](a2_inverse_surrogate_ablation_v1.md)：七 surrogate 固定候选池筛选与 promotion gate。

实验 profile 中的 `config_ready`、命令或预期产物不能替代实际运行。正式状态以结果报告和源 artifacts 为准。
