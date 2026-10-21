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
- [`EXP-DAT-006-K8-K10-K15配对数据生产.md`](EXP-DAT-006-K8-K10-K15配对数据生产.md)：K8/K10/K15 Binary/Graded 各 10k 的跨 K 配对、Xingyi 分 wave 生产与 WSL 验收合同。
- [`full_cell_dual_lane_data.md`](full_cell_dual_lane_data.md)：非 p4m 全胞双路线 real-MPB 数据协议。
- [`full_cell_graded_paracloud_m9.md`](full_cell_graded_paracloud_m9.md)：M9 渐变 10k 生成与 WSL 跨环境验收合同。
- [`full_cell_dual_lane_50000_xingyi.md`](full_cell_dual_lane_50000_xingyi.md)：Xingyi 二值/连续各 50k 的计划冻结、双节点提交、恢复与 WSL 验收合同。
- [`full_cell_resolution_convergence_xingyi.md`](full_cell_resolution_convergence_xingyi.md)：Xingyi 全胞 MPB 分辨率阶段 A 的 16 结构资源 pilot 与阶段 B 停止门。
- [`full_cell_dual_lane_sample_gallery.md`](full_cell_dual_lane_sample_gallery.md)：可追溯样本图册。
- [`representation_ablation.md`](representation_ablation.md)：输入表示消融。
- [`inverse_scale_preflight.md`](inverse_scale_preflight.md)：扩容前的资源、数据与 gate 预检。

## Forward、Query 与 Field

- [EXP-FWD-008-K5固定200轮训练与六域验证](EXP-FWD-008-K5固定200轮训练与六域验证.md)：历史冻结协议，当前状态以对应结果页为准。
- [EXP-FWD-009-K5权重热启动收敛续训](EXP-FWD-009-K5权重热启动收敛续训.md)：历史冻结协议，当前状态以对应结果页为准。
- [EXP-FWD-010-Joint各4k公平训练与六域验证](EXP-FWD-010-Joint各4k公平训练与六域验证.md)：历史冻结协议，当前状态以对应结果页为准。
- [EXP-FWD-011-Joint各4k续训至300轮](EXP-FWD-011-Joint各4k续训至300轮.md)：历史冻结协议，当前状态以对应结果页为准。
- [EXP-FWD-012-工作频率先验覆盖审计](EXP-FWD-012-工作频率先验覆盖审计.md)：历史冻结协议，当前状态以对应结果页为准。

- [`a1_fair_forward_query.md`](a1_fair_forward_query.md)：A1 Full-Wh / Query 公平对照。
- [`full_cell_k20_forward_seed0.md`](full_cell_k20_forward_seed0.md)：K=20 B/G 单 seed 正向训练、34-cell native/cross-lane 评估与 N40 分波提交合同。
- [`EXP-FWD-007-K5训练与六域验证.md`](EXP-FWD-007-K5训练与六域验证.md)：K5 全模型训练、K5/K20/F50 六域 510-cell 评估与 N40 授权门。
- [`EXP-FWD-014-gap决策损失短预算筛选.md`](EXP-FWD-014-gap决策损失短预算筛选.md)：公平 K5 4k+4k 的 baseline/gap-aware seed0、20 epoch N40 配对筛选；只读 validation，不访问 test。
- [`EXP-FWD-015-gap决策近简并审计.md`](EXP-FWD-015-gap决策近简并审计.md)：复用 EXP-FWD-014 冻结检查点，区分近简并评价效应与模型决策能力。
- [`EXP-FWD-016-gap排序与完整带隙短预算筛选.md`](EXP-FWD-016-gap排序与完整带隙短预算筛选.md)：一个待授权的 N40 ranked-v2 seed0 短筛任务；通过前不进入多 seed 或 test。
- [`EXP-FWD-018-K5理论频率加权全谱训练.md`](EXP-FWD-018-K5理论频率加权全谱训练.md)：已登记但不属于本次授权范围的 K5 理论频率加权配对训练合同。
- [`EXP-FWD-019-FNO-TFNO-ConvNeXt组件归因.md`](EXP-FWD-019-FNO-TFNO-ConvNeXt组件归因.md)：已授权的 F50 validation-only 冻结诊断、组件删除 seed0 漏斗、三 seed 配对确认与六域冻结评估合同。
- [`EXP-FWD-020-全局局部带隙算子桥接.md`](EXP-FWD-020-全局局部带隙算子桥接.md)：由 EXP-FWD-019 推导的固定融合频谱—周期大核桥接模型；G3 六域评估完成，G2 未通过，结果保持探索性 surrogate 边界。
- [`EXP-FWD-021-8k公平多域训练与六域验证.md`](EXP-FWD-021-8k公平多域训练与六域验证.md)：九个标准 Full-Wh 模型、七种固定 8k 多域制度、seed0 训练与六域 378-cell 探索性评估合同。
- [`EXP-FWD-023-固定模型十二域交叉测试.md`](EXP-FWD-023-固定模型十二域交叉测试.md)：复用 63 个冻结 checkpoint，在 N56 补充 K8/K10/K15 六域并汇总十二域；不重训，新数据仍待 WSL 物理验收。
- [`full_cell_forward_upgrade_10k.md`](full_cell_forward_upgrade_10k.md)：五个 Full-Wh 升级模型在 N40 的 admission 与 10k 七波训练合同。
- [`field_eigen_te.md`](field_eigen_te.md)：TE Bloch eigenfield、Field FNO 与 adapted PINO。
- [`near_field_surrogate.md`](near_field_surrogate.md)：near-field、NeurOLight 与 Maxwell residual 可行性边界。
- [`EXP-PHY-004-代理选频响应差异.md`](EXP-PHY-004-代理选频响应差异.md)：K5 Full-Wh 全量频谱误差、最大带隙选频稳定性与代表样本 real-MPB/Meep 响应差异。
- [`EXP-PHY-005-TE有效折射率与折叠频率校准.md`](EXP-PHY-005-TE有效折射率与折叠频率校准.md)：在不访问 test、不训练的合同下，用 near-Γ real-MPB 斜率校准 TE 有效折射率并审计全部相邻 gap。
- [`EXP-PHY-006-全BZ带隙边缘加密试点.md`](EXP-PHY-006-全BZ带隙边缘加密试点.md)：四个 validation 结构的局部 full-BZ band-edge 加密 pilot；先判断 9×9 标签风险，不等同于全局 gap 认证。
- [`EXP-PHY-007-validation带隙标签门.md`](EXP-PHY-007-validation带隙标签门.md)：64 个 validation 结构全部完成，但标签门失败；EXP-FWD-016 未解锁。

## Inverse 与 replay

- [`generative_inverse.md`](generative_inverse.md)：条件生成式 inverse 的候选/replay 协议历史入口；当前 mixture conditional generator 不是 cVAE 或 diffusion。
- [`a2_inverse_single_seed.md`](a2_inverse_single_seed.md)：A2 单 seed benchmark 与分阶段恢复语义。
- [`a2_inverse_objective_bridge_seed0_v1.md`](a2_inverse_objective_bridge_seed0_v1.md)：Objective-Bridge 100-target seed0。
- [`a2_inverse_objective_bridge_seed0_test625_v1.md`](a2_inverse_objective_bridge_seed0_test625_v1.md)：Objective-Bridge test625 replay。
- [`a2_inverse_surrogate_ablation_v1.md`](a2_inverse_surrogate_ablation_v1.md)：七 surrogate 固定候选池筛选与 promotion gate。

实验 profile 中的 `config_ready`、命令或预期产物不能替代实际运行。正式状态以结果报告和源 artifacts 为准。
