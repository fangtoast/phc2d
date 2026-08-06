<!--
  Dual-lane full-cell sample-gallery experiment contract.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-dual-lane-sample-gallery"></a>

# 全胞双路线样本与响应图实验

本文冻结渐变介质与二值介质两套全胞数据的样本抽取、能带展示和 real-MPB 本征近场回放合同。它只负责可追溯可视化，不改变数据集验收状态，也不定义远场散射问题。

**目录：** [实验问题](#一实验问题) · [数据与抽样](#二数据与抽样) · [响应定义](#三响应定义) · [运行与产物](#四运行与产物) · [证据边界](#五证据边界)

---

## 一、实验问题

图件需要说明：两类二维光子晶体的全胞介质形貌如何与同一样本的真实 MPB TE 能带及最大带隙上边缘本征场空间局域相对应。

图件属于 `image plate + quant`：介电常数图建立结构证据，采样能带图展示频率响应，近场图展示带边模态空间分布。三张图通过 `S01–S16` 保持样本身份一致。

---

## 二、数据与抽样

配置入口为 [`full_cell_dual_lane_sample_gallery_v1.yaml`](../../configs/research/full_cell_dual_lane_sample_gallery_v1.yaml)。每套数据仅从固定 `test` split 使用随机种子 `20260806` 分层无放回抽取 16 个样本。

| 路线 | 数据状态 | 分层配额 |
|---|---|---|
| `graded_medium` | `cluster_generated_pending_wsl_acceptance` | `graded_core_1/2` 各 8 个 |
| `binary_air_holes` | `wsl_cross_environment_acceptance_pass` | 单孔 4 个、双孔 4 个、谱阈值 8 个 |

介质图读取 HDF5 中的 `epsilon_grid_128`，两条路线都使用固定 $\varepsilon_r=1.0$ 到 $8.9$ 色标。抽样索引、样本 ID、分层、结构统计量与 geometry hash 写入 `selected_samples.csv`。

---

## 三、响应定义

### 3.1 采样能带

HDF5 保存 $9\times9$ 完整 BZ 网格上的 10 条 TE 频带。图中只抽取已有坐标组成的 13 点路径：

$$
\Gamma\rightarrow X_9\rightarrow M_9\rightarrow\Gamma,
\qquad X_9=(4/9,0),\quad M_9=(4/9,4/9).
$$

该路径不插值，也不把 $4/9$ 标记为真正的 BZ 边界 $1/2$。图中频率单位保持 MPB 无量纲 $a/\lambda$。

### 3.2 带隙与近场

对 bands 2–9 后的相邻频带，使用全部 81 个 k 点计算：

$$
\Delta_n=\min_k\omega_{n+1}(k)-\max_k\omega_n(k).
$$

选择最大的正带隙并回放其上边缘模态；若所有 $\Delta_n\leq0$，选择最大的伪带隙并明确标为 `pseudo_gap`。real-MPB 回放必须使用与数据生成一致的 TE、`resolution=64`、`mesh_size=3`、10 条保留带、4 条 guard bands 和 `1e-7` tolerance。

求解后先执行 $\varepsilon$ 加权 E 场 $L_2$ 归一化，再计算：

$$
u_E(x,y)=\varepsilon(x,y)\sum_c|E_c(x,y)|^2.
$$

图像将每个样本的 $u_E$ 独立除以自身最大值。该处理只用于观察空间分布，颜色不能用于比较跨样本绝对场强。回放频率与 HDF5 对应频率的 MRE 超过 1% 时运行 fail closed。

---

## 四、运行与产物

正式命令必须在 WSL Ubuntu 的 `meep_env` 中执行：

```bash
PYTHONPATH=src python -m phc2d.cli plot full-cell-gallery \
  --config configs/research/full_cell_dual_lane_sample_gallery_v1.yaml
```

单样本预检使用同一配置和固定抽样结果：

```bash
PYTHONPATH=src python -m phc2d.cli plot full-cell-gallery \
  --config configs/research/full_cell_dual_lane_sample_gallery_v1.yaml \
  --max-samples-per-dataset 1
```

运行目录遵循 `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`，包含标准日志、配置、结果、指标和历史文件，以及 `selected_samples.csv`、`band_source.csv`、`near_field_records.h5`、`figure_manifest.json` 和 6 张图版的 PNG/SVG/PDF 导出。

---

## 五、证据边界

- 二值数据可保留既有 `wsl_cross_environment_acceptance_pass` 状态。
- 渐变数据必须继续标记为 `cluster_generated_pending_wsl_acceptance`；16 个本征场回放不替代固定 WSL 验收方案。
- 本实验近场是无源 TE Bloch 本征模电能密度，不是受激散射近场。
- 没有入射源、有限结构、PML、观测面和归一化合同，因此本实验不生成或声称远场分布。

[↑ 回到页首](#full-cell-dual-lane-sample-gallery)
