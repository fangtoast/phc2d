<!--
  Configuration index for phc2d experiment profiles.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

> 超算平台资源合同位于 [`research/hpc_platform_xingyi_v1.yaml`](research/hpc_platform_xingyi_v1.yaml) 与 [`research/hpc_platform_paracloud_m9_v1.yaml`](research/hpc_platform_paracloud_m9_v1.yaml)；统一执行边界见 [`../docs/hpc_physics_campaigns.md`](../docs/hpc_physics_campaigns.md)。

# Configs

统一配置入口按实验用途分组：

| 目录 | 作用 |
|---|---|
| [`research/`](research/) | 后续研究配置 |
| [`smoke/`](smoke/) | 本地快速 smoke 配置 |

论文复现、正式研究与 WSL 验收配置均由 `research/` 中的明确实验 ID 区分；快速编辑期检查使用 `smoke/`。旧顶层实现目录不再作为活跃配置入口。

干净克隆的非物理 R1 使用：

- [`smoke/synthetic_dataset.yaml`](smoke/synthetic_dataset.yaml)：配置驱动的确定性合成数据；
- [`smoke/forward_synthetic_res_mlp.yaml`](smoke/forward_synthetic_res_mlp.yaml)：一轮最小 forward 训练。

这两项只能证明 smoke 链路可执行，不能进入科研结果或 MPB/Meep 验收。
