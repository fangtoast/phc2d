---
title: "全胞介电常数内容哈希与碰撞设计"
summary: "设计全胞参数、几何和多分辨率介电常数内容哈希及 split 碰撞审计。"
page_type: plan
status: draft
plan_state: active
effective_date: 2026-08-16
superseded_by: null
related_results: []
audience: "研究用户、数据实现者与导师"
owner: "导师提出方案，研究生维护与验证"
last_reviewed: 2026-08-17
reading_time: 9 minutes
primary_layer: L3
---

<!--
  Author: Ming Lyu <caref.lm@gmail.com>
  Copyright (c) 2026 Ming Lyu. All rights reserved.
-->

# 全胞数据随机碰撞与介电常数内容哈希说明

| Metadata | Value |
|---|---|
| Status | Draft method note / not implemented |
| Scope | `full_cell_v1` binary 与 graded 数据路线 |
| Purpose | 结构去重、split 泄漏审计和有效样本数统计 |

相关代码：

- [随机结构生成](https://github.com/fangtoast/phc2d/blob/main/src/phc2d/generators/full_cell.py)
- [材料场与现有 geometry hash](https://github.com/fangtoast/phc2d/blob/main/src/phc2d/data/full_cell.py)
- [样本计划与 split](https://github.com/fangtoast/phc2d/blob/main/src/phc2d/data/full_cell_generation.py)

## 一、碰撞为什么不可忽略

当前生成器把 master seed、sample ID 和生成器类型经 SHA-256 映射到 32 位随机种子。
独立样本数为 $n$、种子空间为 $M=2^{32}$ 时，至少发生一次碰撞的概率近似为：
$$
P(\mathrm{collision})\approx
1-\exp\left(-\frac{n(n-1)}{2M}\right).
$$

| 同一生成逻辑下的样本数 | 碰撞概率 |
|---:|---:|
| 12,500 | 1.8% |
| 25,000 | 7.0% |
| 50,000 | 25.3% |

只有执行相同生成逻辑的 seed 碰撞才必然导致相同随机序列，但不同 seed 也可能经
阈值化或离散化得到相同材料场。因此，扩大随机种子的位数只能降低原因层面的碰撞，
不能代替对生成结果的去重。

当前 graded 50k 固定样本计划中已经存在一组相同 `graded_core_2` 派生种子。该组位于
train 内，不构成跨 split 泄漏，但说明 32 位碰撞在本项目规模上已是实际问题。

## 二、身份哈希不等于内容哈希

现有 `geometry_hash` 包含 `sample_id`。它可以证明一条结构记录是否改变，却不能判断
两条记录的介电常数分布是否相同。合理的数据合同应区分：

| 类型 | 含义 |
|---|---|
| 样本身份 | 样本在计划中的编号与谱系 |
| 记录哈希 | ID、生成参数和材料常数记录是否被修改 |
| 内容哈希 | 实际介电常数分布是否相同 |

物理去重应依据内容哈希。内容哈希不得包含 sample ID、随机种子、split 或生成器名称；
这些字段不同但材料场相同的样本，应被识别为重复。

## 三、二值材料场的方法

二值路线只有空气与介质两个状态，可将规范栅格表示为整数 `0/1`，并直接对以下内容
计算 SHA-256：

- 固定的周期单胞采样规则；
- 固定栅格形状与轴顺序；
- 按统一顺序排列的 `0/1` 整数数组。

该哈希是精确内容哈希：任一像素的材料类别改变，哈希都应改变。样本 ID 和生成参数
只保留在记录哈希中，不进入二值内容哈希。

## 四、灰度材料场的方法

### 4.1 不直接哈希浮点数

连续介电常数使用浮点数组时，数学上等价的结果可能因平台、数学库或舍入顺序产生
极小差异。直接哈希 float32/float64 字节会把数值噪声误判为不同结构，也无法表达
项目能够接受的物理容差。

灰度路线应先把介电常数映射到固定数量的整数灰度阶，再对整数数组计算哈希。令灰度
阶数为 $L$，可采用：

$$
q=\operatorname{round}\left(
\frac{\epsilon-\epsilon_{\mathrm{air}}}
{\epsilon_{\mathrm{material}}-\epsilon_{\mathrm{air}}}(L-1)
\right),\qquad q\in\{0,\ldots,L-1\}.
$$

两个连续场只要量化后的整数灰度场相同，就具有相同的容差内容哈希。一个灰度阶对应
的介电常数间隔为：

$$
\Delta\epsilon=
\frac{\epsilon_{\mathrm{material}}-\epsilon_{\mathrm{air}}}{L-1}.
$$

因此，$L$ 不是单纯的存储参数，而是内容相等判据中的物理容差。它应根据求解器精度、
栅格分辨率和可接受的材料误差统一确定，并作为数据 schema 的一部分冻结。

### 4.2 数据集也采用定点灰度

如果内容哈希使用整数灰度阶，而 HDF5 仍把浮点介电常数作为权威数据，两者会形成两套
“相等”定义。为保持一致，graded 数据集也应把整数灰度阶作为权威存储：

- binary 使用 `uint8` 的 `0/1`；
- graded 根据所需灰度阶数使用 `uint8` 或 `uint16`；
- 数据集记录灰度阶数、空气介电常数和介质介电常数；
- 训练或求解时需要的浮点介电常数由整数灰度阶按固定比例恢复。

这样，存储、去重和模型输入共享同一离散材料语义；浮点数组只是派生视图，不再承担
样本身份。定点表示还可避免跨平台字节差异，并明确数据本身的有效精度。

若物理求解器仍直接使用未量化的解析连续场，则整数灰度哈希只能代表保存和训练使用的
离散表示，不能宣称它完整标识了解析求解输入。两种合同应选择其一，或明确分别记录。

## 五、建议的生成与审查方法

建议采用以下原则：

1. 先生成并通过现有材料占比、连通性和频谱门槛；
2. 按冻结的分辨率和灰度阶数得到规范整数材料场；
3. 计算与 sample ID 无关的内容哈希；
4. 在 split 冻结前执行全局去重；
5. 重复样本采用确定性方式重新生成，达到次数上限时 fail closed；
6. train、val、test 的内容哈希集合必须两两不交；
7. manifest 同时报告总样本数、唯一内容数、重复组数和各 split 唯一数；
8. 不同输入分辨率可分别计算表示哈希，用于发现下采样后才出现的重复。

同时建议把随机种子派生扩展到至少 64 位，以降低相同 RNG 初态的概率。但正式数据的
唯一性结论必须来自整数介电常数分布的内容哈希，而不是来自 ID 或 seed 不同。

## 六、最小验证标准

- 相同材料场、不同 sample ID：记录哈希不同，内容哈希相同；
- binary 任一像素翻转：内容哈希改变；
- graded 浮点扰动未跨越灰度阶：容差内容哈希不变；
- graded 扰动跨越灰度阶：容差内容哈希改变；
- 串行和并行生成得到相同的整数材料场与内容哈希；
- 所有 split 的内容哈希无交集；
- 数据集中存储的整数灰度场与按 schema 恢复的介电常数一致。

在上述合同实现并完成既有数据审计前，现有 `geometry_hash` 只能作为结构记录哈希，
不能作为物理材料场唯一性的证据。
