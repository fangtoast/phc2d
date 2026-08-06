<!--
  Full-cell inverse Stage-A resumable runner smoke evidence.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Full-cell inverse Stage-A runner smoke（2026-08-06）

| Metadata | Value |
|---|---|
| Status | Implementation smoke passed；N40 正式训练未执行 |
| Primary layer | L4 implementation，向 L3 只提供 runner 可执行性证据 |
| Code revision | `9cd0f8e` (`feat(inverse): add resumable full-cell Stage-A training`) |
| Environment | Windows Python targeted test；WSL `Ubuntu` + conda `meep_env` targeted test |
| Evidence boundary | 人工生成小 HDF5 的 epoch/resume smoke；不是 10k 收敛、GPU、正向 surrogate、Stage B 或 MPB 证据 |

## 结论

full-cell inverse Stage-A 已有独立的 `(variant, method, seed)` 单任务训练入口。runner 能按 train/validation split 训练，按 validation composite loss 选择 best checkpoint，保存 optimizer 状态并在相同 run dir 精确续训，同时生成 `history.csv`、PNG/PDF 训练曲线和标准 run 制品。

当前 full-cell 正向替代模型尚未训练。Stage A 使用真实 HDF5 配对标签，不读取 forward checkpoint；runner 在 `results.json` 中固定记录 `forward_surrogate_status=not_trained`，并将 `stage_b_admitted`、`candidate_ranking_admitted` 和 `inverse_mpb_replay_admitted` 全部置为 `false`。

## 冻结入口

- CLI：`phc2d inverse full-cell-train --config ... --variant ... --method ... --seed ... --run-dir ...`；
- CPU smoke config：[`full_cell_inverse_stage_a_train_cpu_v1.yaml`](../../../configs/smoke/full_cell_inverse_stage_a_train_cpu_v1.yaml)；
- N40 formal config：[`full_cell_inverse_stage_a_train_n40_v1.yaml`](../../../configs/research/full_cell_inverse_stage_a_train_n40_v1.yaml)；
- N40 单任务 wrapper：[`paracloud_gpu_full_cell_inverse_stage_a_train.sh`](../../../jobs/paracloud_gpu_full_cell_inverse_stage_a_train.sh)。

正式 N40 配置冻结 200 epochs、`50/100/150/200` checkpoint、patience 20、batch/validation batch 64、FP32、AMP/TF32 off，以及 Direct `3e-4`、Single/Mixture `1e-4`。配置包含 binary/graded 数据集 SHA-256，runner 还把 config、codec、dataset、code revision 和 variant/method/seed 作为 resume identity；任一不一致即拒绝恢复。

## 验证结果

| 检查 | 结果 |
|---|---|
| Windows targeted pytest | `7 passed` |
| WSL `meep_env` codec + Stage-A targeted pytest | `17 passed` |
| Python compile | passed |
| Ruff | passed |
| N40 wrapper `bash -n` | passed |
| partial epoch 1 → same-run resume to epoch 3 | passed |
| optimizer/state identity restore | passed |
| `best.pt` / `last.pt` / `model.pt` / fixed epoch checkpoints | present in test run |
| final best checkpoint exact reload | passed |
| `history.csv` and PNG/PDF training curves | present in test run |
| test split used for training/selection | no |

这些测试中的 HDF5 和 run 目录由测试临时创建，不作为项目训练结果长期保留。真实双路线 10k 的 single-batch 合同另见 [Stage-A CPU 预检](full_cell_inverse_stage_a_cpu_preflight_20260806.md)。

## 当前门禁

1. N40 15-case batch-64 preflight 尚未执行，需先取得 GPU 作业授权；
2. 通过后才可逐个授权 45 个 Stage-A 正式 run，并用正式曲线/validation checkpoint 判断收敛；
3. full-cell forward 102-run 尚未训练和冻结，所以 Stage B、候选排序与 full-cell inverse 8-target MPB smoke 均保持 blocked；
4. 已完成的 graded 数据集 8 结构 WSL 验收不是 inverse candidate replay，不能直接扩展到整个 validation。

## 人工理解记录

```text
解决了什么问题：补齐 full-cell inverse Stage-A 的 epoch、validation、best/last/fixed checkpoint、optimizer resume、曲线和标准制品入口。
为什么采用这个方案：每个 variant/method/seed 独立单卡运行，既能映射 N40 并发资源，也能保持 seed、数据和恢复身份可审计。
改变了项目哪一层/哪一模块：L4 inverse training、CLI、配置、N40 wrapper 和自动测试；没有启动 L3 正式实验。
使用什么证据验证：Windows 7 项 targeted tests、WSL meep_env 17 项 targeted tests、静态检查和 shell syntax check。
最大风险、未知或下一 gate：尚无 N40 batch-64、正式 10k 收敛或 forward checkpoint；下一步需人工授权 N40 preflight，而不是直接提交 45 个训练或 M9 replay。
```
