---
title: EXP-FWD-022 全局局部分支归因
summary: 冻结 Global-Local checkpoint，通过 raw/renorm 分支关断、输出扰动和内部 attribution 区分分支依赖与尺度效应。
page_type: plan
status: draft
audience: 正向代理研究者、复现执行者与结果审阅者
owner: 研究生维护，导师审阅研究主张
last_reviewed: 2026-09-16
reading_time: 8 minutes
primary_layer: L3
---

<!--
  EXP-FWD-022 Global/Local branch-attribution contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="exp-fwd-022"></a>

# EXP-FWD-022：Global / Local 分支依赖与独立能力归因

**结论：**本实验先执行冻结 validation 归因 G0--G1，只回答现有 Full 模型在六个 target domain 上依赖 Global 还是 Local，以及退化中有多少来自 residual update scale。独立训练属于后续 G2--G3，必须在 G1 人工审阅后重新准入。

**目录：** [研究问题](#research-question) · [冻结干预](#intervention) · [指标](#metrics) · [内部归因](#internal) · [独立训练](#training) · [执行门](#gates) · [证据边界](#boundary)

---

<a id="research-question"></a>

## 一、研究问题

1. Full 模型在不同 source→target domain 上更依赖 Global 还是 Local？
2. 分支关断退化来自信息缺失，还是 residual update scale 改变？
3. 六个 block 的 activation 与 Gradient×Activation 是否和冻结干预一致？
4. 单个分支能否独立建模，Full 是否存在不能由参数量解释的组合收益？

K5/K20 是 Fourier cutoff；F50 是 50k 样本合同。K 相关主张只允许写成 K5/K20 差异，F50 作为数据规模与分布对照单列。

---

<a id="intervention"></a>

## 二、冻结干预

对 EXP-FWD-020 的 36 个冻结 checkpoint 执行六 target validation：

| 模式 | block update | 作用 |
|---|---|---|
| Full | $(G+L)/2$ | 原模型基线 |
| G-off raw | $L/2$ | 同时包含信息缺失与尺度变化 |
| L-off raw | $G/2$ | 同上 |
| G-off renorm | $L$ | 控制 residual scale 后删除 Global |
| L-off renorm | $G$ | 控制 residual scale 后删除 Local |

矩阵为 $36\times6\times5=1080$ 个 validation intervention cells。Full 前向、factory identity、state-dict schema 和 checkpoint 参数必须保持不变。

validation 是主分析。既有 EXP-FWD-020 test cache 只能在 G1 结论模板冻结后作事后探索性复用，不得驱动任何模型或判据变化。

---

<a id="metrics"></a>

## 三、核心指标

主误差为 Bands 2--10 MRE。Global 分支定义：

$$
I_G^{ctrl}=MRE_{G\text{-}off,renorm}-MRE_{Full},
$$

$$
I_G^{raw}=MRE_{G\text{-}off,raw}-MRE_{Full},
$$

$$
S_G=I_G^{raw}-I_G^{ctrl}.
$$

Local 分支对称定义。$I^{ctrl}$ 是主 dependency 指标，$S$ 单独表示尺度效应。

无标签输出扰动为：

$$
D_G=\operatorname{mean}_i
\frac{\|\hat W_{Full}^{(i)}-\hat W_{G\text{-}off,renorm}^{(i)}\|_F}
{\|\hat W_{Full}^{(i)}\|_F+\epsilon}.
$$

每个干预使用同 checkpoint、target 和样本做 10,000 次 paired bootstrap。主结论要求 95% CI 不跨 0 且三个 seed 方向一致；相对变化达到 1% 只作为 practical significance 标记。

per-band 与 per-k 指标只用于 heatmap 和稳定空间模式，执行 BH-FDR `q=0.05`，不以显著格子数量形成主结论。

---

<a id="internal"></a>

## 四、内部 attribution

核心内部指标仅保留：

$$
A_G^\ell=RMS(G_\ell),\qquad A_L^\ell=RMS(L_\ell),
$$

$$
C_G^\ell=mean\left|G_\ell\odot\frac{\partial\mathcal L}{\partial G_\ell}\right|,
\qquad
C_L^\ell=mean\left|L_\ell\odot\frac{\partial\mathcal L}{\partial L_\ell}\right|,
$$

$$
R_C^\ell=\frac{C_G^\ell}{C_G^\ell+C_L^\ell+\epsilon}.
$$

每个 checkpoint-target 固定两个 validation 样本集：随机 512 作为主 attribution，按边界密度与灰度比例抽取的结构分层 512 只作探索性机制分析。索引在运行前冻结并记录哈希。

---

<a id="training"></a>

## 五、后续独立训练合同

G1 人工通过后，Binary 与 Graded 分开使用 K5+K20 等权 mixed contract。每条 lane 训练 Full、G-only、L-only、G-only parameter-count control、L-only parameter-count control，各三个 seed，共 30 个任务。

- Local kernel 固定为 $3\times3$，仅作为 nuisance-controlled 选择。
- 参数量控制只调 width，并要求总参数量落在 Full 的 $\pm5\%$ 内；它不是 architecture-equivalent control。
- 组合收益定义为 $B_{GL}=\min(E_G,E_L)-E_{Full}$，容量控制同理。
- 该阶段不得因 G1 结果自动启动，必须重新冻结 manifest 和远程授权。

---

<a id="gates"></a>

## 六、执行门

| Gate | 完成条件 | 当前权限 |
|---|---|---|
| G0 工程门 | 五态干预、输出扰动、双样本集 attribution、兼容测试与单卡预检通过 | 已授权 |
| G1 冻结归因 | 1080/1080 validation cells、216 组内部 attribution、paired bootstrap 全部有限完整 | 已授权在 N40 执行 |
| G1 人工审阅 | 判定是否存在稳定 G/L 分化信号 | 必须人工决定 |
| G2 独立训练准入 | 30-run manifest、容量控制和训练资源重新冻结 | 尚未准入 |
| G3 能力证据 | dependency 与 capability 分层解释、组合收益完成 | 尚未准入 |

G0/G1 使用 N40 `gpu_4090`。N56 只作为 EXP-FWD-020 checkpoint 来源；传输必须保留 checkpoint SHA-256，并且不得覆盖不匹配文件。

---

<a id="boundary"></a>

## 七、证据与非目标

- 上游证据：[EXP-FWD-020 六域结果](../../docs/results/forward/EXP-FWD-020-全局局部带隙算子六域交叉评估.md)
- 机器合同：[EXP-FWD-022 配置](../../configs/research/full_cell_global_local_branch_attribution_exp_fwd_022_v1.yaml)
- 实现入口：`phc2d.forward.global_local_branch_attribution`

G1 只能形成 frozen validation branch dependency 证据，不改变 EXP-FWD-020 的 `g2_formal_status=not_passed`，不产生模型 promotion、test-confirmed、real-MPB/Meep 或物理因果主张。本页由 AI 整理，保持 Draft，等待人类负责人审阅。
