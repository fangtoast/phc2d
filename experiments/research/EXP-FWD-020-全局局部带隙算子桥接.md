<!--
  EXP-FWD-020 global-local band-operator bridge contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-FWD-020：全局—局部带隙算子桥接

> 状态：`authorized_n56_gated_campaign_pending_preflight`；主层级：L3 实验与证据。已获本轮 N56 环境、数据部署与门控训练提交授权；截至本文更新仍没有训练、validation 性能、held-out test 或 real-MPB/Meep 证据。

## 结论与问题

EXP-FWD-019 支持两个可进入桥接验证的信号：频谱主分支不可无损删除，周期大核局部分支的三种子删除敏感性最强且容量混杂较小。EXP-FWD-020 只检验一个最小问题：

> 在同一个频谱—局部算子骨架内，把周期局部核从 $3\times3$ 扩大到 $7\times7$，能否在 Binary 和 Graded 两条 F50 validation 路线上稳定改善 Bands 2--10 MRE？

这不是“物理规律已经被网络发现”的声明，而是由数据归因结果约束出的可证伪架构假设。

## 模型合同

模型身份为 `global_local_band_operator_v1`：

1. Tucker-factorized 双侧频谱分支负责全局空间混合；
2. 周期填充的 depthwise 大核与 pointwise channel MLP 负责局部空间混合；
3. 两条分支固定等权融合，不使用可学习 branch gate；
4. 不使用独立 TFNO channel 分支，不使用 lane embedding/FiLM；
5. 输出仍为 `material_weight_grid_64 -> Wh[81,10]`，沿用现有 band head 和 `normalized_full_matrix_mse`。

首个配对对照只改变 `local_kernel: 3 -> 7`。两侧均使用同一注册模型、频谱宽度、深度、Fourier modes、factorization rank、优化器、数据切分、seed 和训练预算。

## 数据解释与物理边界

| 层级 | 当前内容 | 允许的表述 |
|---|---|---|
| 已验证性质 | EXP-FWD-019 中 FNO/Joint 删除 spectral 后两路线均稳定退化；ConvNeXt `7x7 -> 1x1` 两路线三种子均大幅退化 | 这些组件在冻结 F50 validation 合同内包含有效预测信息 |
| 可检验机制 | `Wh[81,10]` 汇总全胞与全 BZ 响应，频谱混合可能适合长程周期耦合；周期大核可能适合跨边界邻域与介质界面模式 | 作为模型设计先验和实验假设 |
| 未验证物理规律 | Fourier 分支等价于 Bloch 展开；大核直接学习 Maxwell 散射定律；桥接模型一定优于 ConvNeXt | 禁止作为结论，需 matched training、跨域和 real-MPB/Meep 证据 |

## 执行门

### G0：工程门

- 注册身份、factory identity、state-dict schema 可追踪；
- 动态输入/输出 shape 和 k-point 输出合同通过；
- 周期平移不改变全局预测的数值回归通过；
- 旧 TFNO/Joint 默认 checkpoint schema 不变。

### G1：seed0 短筛

- K5、K20、F50 的 Binary/Graded 六条路线分别做 `kernel_3` 与 `kernel_7` 配对训练；
- seed 固定为 0，最多 20 epochs，只读 validation，不读取 test；
- 每条“数据集—lane”独立满足 candidate MRE 低于 baseline，且相对改善至少 1%；
- candidate/baseline 参数比不得超过 1.05；任一 lane 反向即停止。

### G2：三种子确认

G1 共 12 个 seed0、20 epoch validation-only 任务，以 `%8` 数组执行；只有各自通过的路线才进入 G2。G2 每条通过路线的两个 kernel 各做 seeds 0/1/2、200 epochs，从头训练，不修改预算或阈值。每条路线均需三种子同向，且 baseline/candidate 满足原收敛门；否则不进入跨域评估。

### G3：六域与最终 test

仅 G2 通过后冻结 checkpoint 身份，执行同一材料 lane 的 K5/K20/F50 原生与跨数据集 held-out 评估；满额为 108 个 checkpoint-target 单元。held-out test 在模型、阈值和报告模板全部冻结后至多读取一次。real-MPB/Meep 仍是独立物理验证任务。

## 证据与非目标

- 上游结果：[EXP-FWD-019 三种子确认](../../docs/results/forward/EXP-FWD-019-组件归因三种子确认.md)
- 机器合同：[EXP-FWD-020 配置](../../configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml)
- 实现：`src/phc2d/models/full_cell_operators.py`
- N56 运行时固定为 `gpu_5090`、CUDA 12.8、PyTorch 2.7.1 的版本化独立环境；提交前必须单卡验证 CUDA、PyTorch、HDF5、目标模型前向/反向与小批量读取。N56 上哈希错误的旧 F50-Graded 严格排除；正确副本从 N40 以 `.partial`、SHA256 校验和原子改名部署。非目标：不重跑 EXP-FWD-019，不调整科学判据，不自动重试、取消、追加训练或更换 GPU，不宣称性能提升或物理因果。

本页是 AI 整理的 Draft，正式远程训练需新的明确授权。
