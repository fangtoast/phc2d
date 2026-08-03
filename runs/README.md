<!--
  Run-output policy and current local evidence pointers for phc2d.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# runs/ 运行产物入口

脚本输出统一写入：

```text
runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/
```

标准产物包括 `run.log`、`run.meta.json`、`config.resolved.yaml`、`results.json`、`metrics.json`、`history.csv` 和按需生成的 `figures/`。完整运行目录、checkpoint 和原始数据不进入 Git；远端只同步本入口、精选结果报告、源路径和哈希。

## 当前证据入口

- Field 指标修正：`runs/20260803/` 下对应的 field metric 审计运行。
- A2 surrogate 修复后验证：`runs/20260802/` 下对应的 postrepair validation 运行。
- Field P6 seed-0 pilot：`runs/20260729/field_eigen_te_p6_seed0_target_residual_pilot_64_202607291023/`。
- A2 625-target objective bridge：`runs/20260728/a2_inverse_objective_bridge_seed0_test625_v1_202607280842/`。
- A1 forward/query 公平基线：`runs/20260724/` 和 `runs/20260726/` 下对应运行。

可引用结论统一从 [`../docs/results/README.md`](../docs/results/README.md) 进入。运行目录存在不等于实验通过；必须读取其状态、指标和验证边界。
