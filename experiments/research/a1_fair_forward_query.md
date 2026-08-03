<!--
  A1 fair forward/query experiment profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# A1：10k V2 Forward/Query 公平对照

本实验只产生 neural-surrogate 指标，不调用 MPB/Meep，不启动 inverse、physics rerank、MaxwellNet 或 100k 数据生成。

## 冻结协议

- 数据：`data/query_dataset_v2_pilot_10000_mpb.npz`，SHA-256 与 `baseline_protocol_v1.yaml` 一致；split 固定为 `6875/2500/625`。
- Forward：`res_mlp_wide`、`lifted_cnn`、`fno2d_band_v2`；Query：`mlp_query`、`deep_onet_query_v2`、`fno2d_query_v2`。
- 每个模型/协议训练 seeds `0/1/2` 与三项 learning rate；只以 validation bands 2–10 MRE 选择单一 learning rate，最终 test 汇总严格为 selected-LR 的 `n=3` mean/std。

## 执行入口

```text
PYTHONPATH=src python -m phc2d.cli forward benchmark --config configs/research/a1_forward_fair_v1.yaml
PYTHONPATH=src python -m phc2d.cli query benchmark --config configs/research/a1_query_fair_v1.yaml
```

运行目录包含 `validation_history.csv`、`validation_selection_manifest.json`、selected-LR 的 `history.csv`、`leaderboard.csv`、per-band/per-k 指标、模型 identity、lineage 和完整性审计。任何缺少三 seed、数据/切分 hash 不匹配或 physical-k 未验证的结果均不得进入 A1 主表。
