<!--
  Wang 2026 reproduction experiment profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Wang 2026 Reproduction Profile

本目录收口 Wang 等 2026 POD-DeepONet 论文复现相关说明。旧 `legacy/reproduction` 顶层目录已从活跃树移除；后续若需要恢复旧脚本，应先把逻辑迁入 `src/phc2d`，再通过 `phc2d` CLI 暴露。

| 内容 | 当前来源 |
|---|---|
| 复现与对照配置 | [`../../../configs/research`](../../../configs/research) 中带明确实验 ID 的 YAML |
| 数据生成与 baseline 脚本 | [`../../../src/phc2d`](../../../src/phc2d) |
| 复现说明 | 本目录与 [`../../../docs`](../../../docs) |

后续迁移时，只迁移复现所需配置和薄入口，不重新引入顶层 legacy 区。
