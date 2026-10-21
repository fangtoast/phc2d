<!--
  EXP-FWD-023 fixed-checkpoint twelve-domain evaluation contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-023：固定模型十二域交叉测试

## 结论与状态

本实验为 Draft、`authorized_pending_data_transfer_and_freeze`。2026-09-17 人工授权在 N56 测试训练好的模型，并确认“交叉验证”指跨数据集、跨 Binary/Graded 的固定模型测试，不做 K-fold 重训。

主层级为 L3。问题是：EXP-FWD-021 的九模型、七种 8k 制度、seed0 validation-best checkpoint，在 K5/K8/K10/K15/K20/F50 两路线上的完整 held-out surrogate 表现如何？本实验不调整训练、不重新挑 checkpoint，不改变已有六域测试合同。

## 冻结输入与证据边界

| 输入 | 合同 |
|---|---|
| 模型 | EXP-FWD-021 冻结 checkpoint manifest 的 63 个 validation-best 身份；逐一核对 SHA-256、factory、结构、normalizer、训练来源 |
| 原六域 | 复用 EXP-FWD-021 的六份只读 cache 与 378-cell 评估，禁止重新读取原始 test；未完成部分继续按其原合同执行 |
| 新六域 | EXP-DAT-006 的 K8/K10/K15 Binary/Graded 最终 HDF5；每域固定 `idx_test` 1,000 条，训练/验证/测试为 8,000/1,000/1,000 |
| 总矩阵 | 63×12＝756 个唯一 cell，7,560 条逐 Band 记录；旧六域 378＋新六域 378，不重复计算已验收 cell |
| 数据状态 | 新六域 `cluster_audited_pending_wsl_acceptance`；哈希/shape/split 审计不等于 WSL real-MPB 物理验收 |
| normalizer | 每个 checkpoint 自带的 pooled 训练 normalizer；不得按目标域重拟合、切换或更新 |

K8/K10/K15 同 lane 使用配对 cohort：冻结 sample_id 和 split 必须精确一致。K5/K20/F50 不与该 cohort 声称逐结构配对。新增目标的 test sample_id 和输入栅格哈希不得命中原训练或源域 validation；检查不读取源域 test 标签。

## 执行与失败门

1. 六份最终文件通过 `.part`＋SHA-256 转到 N56，保留原文件、集群 audit 与 pairing provenance；不下载 shard，不执行新 MPB/Meep。
2. 完成存在性、完整哈希、字段、shape、split 零重叠、样本唯一性、cohort 配对及 source-train/validation 泄漏审计；无标签调整。
3. 独立补充 campaign 从原 resolved config 和 manifests 导入训练身份，仅新增 provenance；freeze 只读取新六域的 test 标签一次并保存只读 cache。
4. N56 `gpu_5090`，CUDA 12.8 / PyTorch 2.7.1，FP32、AMP/TF32 关闭；1 GPU＋8 CPU、平台默认显存配套主存、24 h，数组≤16、并发≤7。先完成原六域，再补充新六域，不同时开启两套 7-GPU campaign。
5. 每批 Slurm 全部 COMPLETED/退出码 0、标准制品非空、指标有限、10 Band 记录、模型/normalizer/checkpoint/cache 身份审计通过后才可提交下一批；任一失败停止，不自动重试、取消、换分区或 N40。
6. 两套 378-cell 完整后才能冻结合并摘要和结果文档；原证据不覆盖，不把新六域写回原 EXP-FWD-021 manifest。

## 指标、统计与交付

每个 cell 同时保留 Band 1、Bands 2–10、Bands 1–10 与十条逐 Band MRE。主汇总按目标域等权，F50 的 5,000 样本不能支配其他 1,000 样本域。两个统计口径排名不同必须同时报告，不能事后忽略 Band 1。

按“制度×模型”为行、十二域为列输出 source→target 表；同时输出每制度宏平均、与冻结 FNO reference 的成对样本 bootstrap、训练源域/未见域/跨 lane/双重迁移分组、收敛状态、训练耗时/峰值显存与推理吞吐。配对 K8→K10/K15 的误差变化按相同 test sample_id 对齐计算，不把跨数据集非配对均值变化当成逐样本显著性。

原始证据保存在 `runs/`；人类可读结果完成后保存为 `docs/results/forward/EXP-FWD-023-十二域交叉测试.md`，并更新结果索引、Registry 和 PROJECT_MAP。文档保持 Draft。

唯一允许的结论是 `seed0 exploratory surrogate evidence`。新六域另保留数据物理验收未完成的限制；样本 bootstrap 不替代多 seed 稳定性，不产生 real-MPB/Meep 有效、架构普遍优越或 promotion 结论。
