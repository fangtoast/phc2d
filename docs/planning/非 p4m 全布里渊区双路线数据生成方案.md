<!--
  Historical design snapshot for the non-p4m full-cell dual-lane data protocol.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="non-p4m-full-cell-dual-lane-data-plan"></a>

# 非 p4m 全布里渊区双路线数据生成方案

| Metadata | Value |
|---|---|
| Status | Historical Draft；保留 2026-08-03 的 64+64 pilot 原始设计，不作为当前执行状态 |
| Primary layer | L3 experiment/evidence |
| Role | 记录 `full_cell_v1` 双路线数据协议的初始结构、MPB 与验收门槛 |
| Audience | 数据生成、物理验收与后续模型训练维护人员 |
| Owner | 研究生维护，项目负责人审阅协议与证据边界 |
| Last reviewed | 2026-08-10 |
| Reading time | 8 minutes |

> 阅读边界：本文保留历史方案正文，不证明当前任务仍停留在 64+64 pilot，也不证明模型训练或 inverse replay 已完成。当前完成状态以 [`PROJECT_MAP.md`](../../PROJECT_MAP.md) 和[全胞双路线 pilot 结果](../results/data-provenance/full_cell_dual_lane_pilot_20260803.md)为准。

## 总体设计

建立独立于现有 `rho_wedge → Wh` 基线的 `full_cell_v1` 数据协议，保留旧数据与旧模型不变。新协议包含两套分别生成、分别验收的数据：

1. `graded_medium`：真实连续渐变介电常数。
2. `binary_air_holes`：高介电背景中的二值不规则空气孔。

首轮各生成 64 个结构，仅计算 TE、前 10 条 band，采用完整方形第一布里渊区 9×9 网格。两套 HDF5 数据通过总 manifest 关联，不直接混成一个文件。

这里需要澄清：原来的“三角形”实际是正方晶格的不可约布里渊区/高对称路径；新方案不再依赖结构的 p4m 对称性，直接采样完整方形 BZ。

## 结构生成与参数化

### 连续渐变介质

在归一化周期晶胞 \([-0.5,0.5)^2\) 上定义解析材料权重 \(u(x,y)\in[0,1]\)：

\[
g(x,y)=\sum_{j=1}^{J}A_jG_j(x,y)+
\lambda\sum_{\mathbf m}c_{\mathbf m}\phi_{\mathbf m}(x,y),
\qquad
u(x,y)=\operatorname{sigmoid}\frac{g(x,y)-b}{T}
\]

固定规则：

- 结构数：64；32 个单径向核、32 个双径向核。
- 径向核采用周期化椭圆高斯：
  - 尺度 \(0.08a\sim0.22a\)；
  - 长短轴比 \(0.75\sim1.33\)；
  - 幅值 \(0.8\sim1.2\)。
- 傅里叶扰动保留 \(|m_x|,|m_y|\le3\) 的低频实基，系数随频率衰减，扰动强度 \(\lambda\in[0.05,0.25]\)。
- 温度 \(T\in[0.15,0.30]\)；通过求解偏置 \(b\)，令平均材料权重落在随机目标区间 \([0.20,0.75]\)。
- 真实介电常数采用已确认的线性映射：

\[
\epsilon(x,y)=1.0+(8.9-1.0)u(x,y)
\]

- 接受条件：数值有限、严格周期、灰度像素比例 `0.05<u<0.95` 不低于 20%、高频能量比例不超过 5%。
- 不施加任何旋转或镜像投影；只记录 p4m 对称误差用于描述分布。

### 二值空气孔

介质背景为 \(\epsilon=8.9\)，孔内为空气 \(\epsilon=1.0\)。64 个结构按 family 分层：

- 32 个极坐标傅里叶孔：
  - 16 个单孔、16 个双孔；
  - 边界为
    \[
    r(\theta)=r_0\exp\left[\sum_{m=2}^{5}
    (a_m\cos m\theta+b_m\sin m\theta)\right]
    \]
  - 单孔 \(r_0\in[0.14a,0.25a]\)，双孔每孔 \(r_0\in[0.10a,0.18a]\)；
  - 系数标准差按 \(0.12/m\) 衰减；
  - 双孔允许部分重叠，空气区域取并集；首版不允许孔跨越晶胞边界。

- 32 个低通阈值自由形状：
  - 采样周期傅里叶随机场；
  - 截止尺度 \(k_c\in[1.5,3.0]\)，滤波包络为 \(\exp[-(|k|/k_c)^4]\)；
  - 目标空气占比在 \([0.10,0.50]\) 内采样；
  - 阈值由目标分位数加 \([-0.03,0.03]\) 的分位扰动确定；
  - 在 256×256 审计栅格上要求空气连通分量为 1–3，最小特征尺寸不低于 \(0.04a\)。

所有结构保存固定长度 `params`、`param_mask`、`family_id` 和生成参数；MPB、绘图及训练栅格必须由同一份参数重建，禁止从 PNG 或截图反推物理结构。

## MPB、k 空间与数据接口

- 新增不依赖 `rho_wedge` 的全晶胞结构接口，至少包含：
  - `FullCellStructureSpec`
  - `evaluate_material_weight(x, y)`
  - `rasterize(grid_shape)`
  - `geometry_hash`
  - `generator_family`、`params`、`param_mask`
- MPB 使用同一个解析材料函数：
  - 连续路线返回空间变化的 `Medium(epsilon=...)`；
  - 二值路线按孔洞判定返回空气或介质；
  - 图像栅格只作为模型输入和审计产物，不作为 MPB 物理真值。
- 当前 WSL 环境已确认 PyMPB 具备 `MaterialGrid`、`Prism` 和 `ModeSolver`；MPB 官方也支持任意位置材料函数。本方案固定使用解析材料函数，`Cylinder/Prism` 只用于一致性基准。[MPB Python API](https://mpb.readthedocs.io/en/latest/Python_User_Interface/)、[Meep material function](https://meep.readthedocs.io/en/master/Python_User_Interface/)
- MPB 固定参数：
  - square lattice，\(a=1\)
  - TE
  - resolution 64，`mesh_size=3`
  - tolerance \(10^{-7}\)
  - 求 14 条 band，频率升序后保留前 10 条
  - 不启用结构对称性约简
- 完整 BZ 网格采用：
  \[
  k_{ij}=(i/9,j/9),\quad i,j=-4,\ldots,4
  \]
  共 81 个无重复 reduced reciprocal coordinates，包含 \(\Gamma\)，不额外求高对称路径。完整二维采样也能避免只看边界路径而遗漏内部带边信息。[MPB tutorial](https://mpb.readthedocs.io/en/stable/Python_Tutorial/)
- 扩展 `phc2d data generate --config ...`，支持：
  - `--max-samples`
  - worker 数覆盖
  - 单样本 shard
  - 按 `sample_id + config_hash` 安全恢复
  - 失败账本和 fail-fast 配置
- 两套 HDF5 均保存：
  - `params[N,P]`、`param_mask[N,P]`、`family_id[N]`
  - `material_weight_grid_{32,64,128}`
  - `epsilon_grid_{32,64,128}`
  - `k_points[81,2]`、`k_grid_index[81,2]`
  - `Wh[N,81,10]`
  - `idx_train/idx_val/idx_test`
  - 每样本生成器、材料、几何、k 网格和 MPB provenance
- 每套 64 样本按 family/核数/孔数/占比区间分层为 `48/8/8`；采用基于稳定样本 ID 的哈希划分，不能按生成顺序直接切分。
- 总 manifest 只链接两套数据、哈希和协议，不复制或混合数组。现有 `DataRepresentationSpec` 的 `rho_grid`、`epsilon_grid`、`param_vector` 继续使用，但 loader 必须解除“epsilon 必须由 rho_wedge 提升”的假设。

## 分阶段门槛

1. **协议与无求解器测试**
   - 参数可完整重建结构。
   - 相同 seed/sample ID 位级复现。
   - 连续场周期边界一致；二值场只含 0/1。
   - 32/64/128 栅格来自同一解析结构。
   - 不调用 p4m wedge 提升函数。

2. **MPB 编码预检**
   - 均匀空气、均匀介质与 MPB 原生均匀材料结果一致。
   - 规则圆孔的材料函数编码与 `mp.Cylinder` 对照，bands 2–10 MRE 不超过 0.5%。
   - 各路线运行 1–4 个结构，必须零失败、无 NaN、band 有序、几何哈希可回放。
   - 未通过时状态为 `blocked`，不得启动 64+64。

3. **64+64 real-MPB pilot**
   - 每个结构作为一个包含全部 81 个 k 点的可恢复 worker job。
   - 两类数据均要求 64/64 完成，失败率为 0。
   - 每类分层选 8 个结构、每结构选 9 个代表性 k 点，在 resolution 96 和 128 复算。
   - 收敛门槛：64→96 的 bands 2–10 中位相对差不超过 0.5%，96→128 不超过 0.25%，P95 不超过 1%。失败则冻结扩大规模并调整 solver resolution。

4. **数据验收报告**
   - 报告材料/空气占比、灰度比例、连通分量、最小特征、低频能量、p4m 对称误差和 family 覆盖。
   - 展示同一结构的 32/64/128 栅格，以及代表 band 的完整 BZ 二维频率图。
   - 单列结构生成、MPB 求解、I/O 和每样本 wall time。
   - 64+64 只能标记为 `pilot_pass`，不能称为正式训练数据或证明模型收益。

5. **后续扩展**
   - 只有 pilot、分辨率收敛和数据审计全部通过后，才创建 256/1000 规模配置。
   - 扩大规模时保持生成参数分布、split 策略、k 网格和 MPB 协议不变。
   - 模型训练、逆向生成、本征场数据和 TE/TM 对比均不属于本轮实现范围。

## 测试与交付

- 为生成器、周期性、参数重建、拓扑约束、BZ 网格、HDF5 schema、split 防泄漏、shard 恢复和失败传播增加测试。
- 在 WSL `Ubuntu` + `meep_env` 运行相关单测、完整测试、compileall、CLI help 和真实 MPB 预检。
- 每个 run 必须具备标准日志、解析配置、results、metrics、history、figures、solver provenance、sample manifest 和 failure ledger。
- 按“协议/生成器”“MPB 适配与测试”“pilot 配置与报告”拆分作用域明确的提交，不纳入当前工作区已有的无关修改。
- 最近执行门槛固定为“结构协议 + MPB 材料函数一致性预检”；未通过前不启动批量生成。
