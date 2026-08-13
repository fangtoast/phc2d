<!--
  Honest status and entrypoint for the Wang 2026 reproduction profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Wang 2026 Reproduction Profile

本目录收口 Wang 等 2026 POD-DeepONet 对齐接口。它目前不是 faithful reproduction 完成证明：模型 inventory 中 `pod_res_mlp` 仍为 `placeholder`，当前 A1 Forward/Query 模型直接预测 `Wh` 或 query band vector，没有完成 POD basis 管线的同协议训练与评估。

| 内容 | 当前来源 |
|---|---|
| 研究路线与 POD 状态 | [`../../../docs/planning/fno_generative_network_adjustment_plan_20260617.md`](../../../docs/planning/fno_generative_network_adjustment_plan_20260617.md) |
| 模型实现状态 | [`../../../docs/reports/phc2d_model_inventory_v1.yaml`](../../../docs/reports/phc2d_model_inventory_v1.yaml) |
| 当前非 POD 公平基线 | [`../../research/a1_fair_forward_query.md`](../../research/a1_fair_forward_query.md) |
| 分级复现与允许表述 | [`../../../docs/reproducibility.md`](../../../docs/reproducibility.md) |
| 公共代码与配置 | [`../../../src/phc2d`](../../../src/phc2d) · [`../../../configs/research`](../../../configs/research) |

在 POD basis、模型实现、固定数据/split、训练、评估和结果审计全部通过前，只能称为“保留 Wang 2026 对齐与复现接口”，不得称为“完整复现 POD-DeepONet”。历史代码如需参考，只能从 Git history 读取并适配到 `src/phc2d`，不得恢复旧顶层 backend。
