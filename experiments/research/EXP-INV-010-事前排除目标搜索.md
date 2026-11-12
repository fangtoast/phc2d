<!--
  Preregistered exclusion-first inverse development protocol, draft.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-INV-010：事前排除后的目标驱动代理搜索

**状态：Draft，有限开发条件已满足，待人工审阅。** 冻结 Global–Local 的选择在 12 个来源、3 个模型种子的预注册同轨迹对照中，物理目标裕量较随机选择平均高 0.01757，分层 bootstrap 95% 区间为 [0.01041, 0.02643]。84/84 个 M9 案例通过冻结数值门，6/6 个预选案例通过 WSL 有限网格全谱对照（[物理统计](../../runs/20261102/EXP-INV-010_202611020411/source_generation_authorized/physical_complete/paired_physical_evaluation.json)、[跨环境对照](../../runs/20261102/EXP-INV-010_202611020411/source_generation_authorized/wsl_crosscheck/comparison.json)、[条件判定](../../runs/20261102/EXP-INV-010_202611020411/source_generation_authorized/development_decision.json)）。[机器参数](../../configs/research/exp_inv_010_prefrozen_exclusion_development.yaml)沿用 EXP-INV-009 的数值门与资源上限；本开发实验不声称全面历史独立，也不代表原 P2 或 P5 通过。

## 事前准入

在来源生成前冻结有限排除范围与输入哈希：保留已认证的原 EXP-INV-009 排除集、补查的 10,064 条二值样本、EXP-INV-009 全部 1,024 来源和 720 次搜索评估的可用身份，并加入已认证六库的原存 64 栅格及可用参数身份。对可用的原存 256 栅格和既有 512 身份继续保留；不再要求为所有历史结构补算 256/512。新来源和候选仍按各自的模型输入及物理栅格检查重复，并在物理求解前固定选择。清单必须保留每类身份的来源和覆盖范围，缺失或不一致不得冒充已排除。

历史楔形 NPZ 的实际几何合同、场样本的连续结构身份及其他位置的数据范围仍未完全确定。它们作为本开发实验的明确限制记录；已有对称性和像素扩展检查仅在其假设下成立，不能据此宣称全部历史结构独立。本次变更只调整尚未冻结的 010 开发准入，不修改 EXP-INV-009 的事前排除失败、原 P2 结果或任何物理数值门。

远端限定目录盘点还发现本机旧清单之外的二值结构文件：N40 的 50,000 条混合二值数据、EXP-DAT-007 的 12,000 条和 EXP-DAT-008 的 24,000 条，以及 N56 的 K8、K10、K15 各 10,000 条 Fourier 数据。六个文件的完整哈希、116,000 行结构文本与 split 已对照原 HDF 核验（[N40](../../runs/20261102/EXP-INV-010_202611020411/n40_extraction_5528990/independent_5529142/independent_hdf_audit.json)、[N56](../../runs/20261102/EXP-INV-010_202611020411/n56_extraction_1758941/independent_1758989/independent_hdf_audit.json)）；各库几何重建、相互重叠及排除身份尚未全量审核。N56 另有无结构 JSON 的 18,000 行混合文件，其栅格与频谱已逐行对照 K5/K20 上游数据（[溯源审核](../../runs/20261102/EXP-INV-010_202611020411/n56_mixed_lineage_audit.json)）。这些核验不证明远端盘点穷尽或物理独立性。

[23 个结构家族首例的工程对照](../../runs/20261102/EXP-INV-010_202611020411/geometry_hdf_family_sample_audit.json)显示，F50 两例原存 64/128 输入与同分辨率直接重建一致，与 256 栅格降采样不同；其他 21 例的原存 64/256 栅格、所检查参数及哈希一致。F50 的一例参数向量存在极小数值差异（[差异记录](../../runs/20261102/EXP-INV-010_202611020411/f50_sample_parameter_difference.json)），原因未知。排除清单须保留原存输入与重建栅格各自身份；样本对照不能替代逐条审核，也不替代来源生成前的输入审核。

F50 全部 50,000 条原存 64 栅格的无损提取已与原 HDF 逐点独立比对，并核对对应结构文本与 split（[原 HDF 对照](../../runs/20261102/EXP-INV-010_202611020411/f50_stored64_5529207/independent_5529209/f50.stored64.independent_audit.json)、[传输及来源行审核](../../runs/20261102/EXP-INV-010_202611020411/f50_stored64_5529207/transfer_audit.json)）。这些原存输入的规范 64 栅格哈希已逐行独立复算，库内 50,000 个均不同（[身份审核](../../runs/20261102/EXP-INV-010_202611020411/f50_stored64_5529207/f50.stored64.identity.audit.json)）。此结果确认原存输入身份；256/512 几何重建、跨库去重和其他历史数据边界仍待完成。

DAT-007 和 DAT-008 的原存 64/256 栅格哈希已对照各自原 HDF 独立逐行复算，分别覆盖 12,000 和 24,000 条（[DAT-007](../../runs/20261102/EXP-INV-010_202611020411/n40_stored_hashes_5529217/dat007.stored_hashes.independent_audit.json)、[DAT-008](../../runs/20261102/EXP-INV-010_202611020411/n40_stored_hashes_5529217/dat008.stored_hashes.independent_audit.json)）。[来源行及哈希集合检查](../../runs/20261102/EXP-INV-010_202611020411/n40_stored_hashes_5529217/local_stream_identity_audit.json)显示，两库在这些参数及原存栅格哈希上无精确交集；库内分别有 9,660 和 21,705 个不同的原存 256 栅格。该比较不覆盖其他库、512 栅格或连续几何等价，也不构成排除清单准入。

N56 的 K8、K10、K15 各 10,000 条原存 64/256 栅格哈希也已对照原 HDF 逐行复算（[K8](../../runs/20261102/EXP-INV-010_202611020411/n56_stored_hashes_1759199/k8.stored_hashes.independent_audit.json)、[K10](../../runs/20261102/EXP-INV-010_202611020411/n56_stored_hashes_1759199/k10.stored_hashes.independent_audit.json)、[K15](../../runs/20261102/EXP-INV-010_202611020411/n56_stored_hashes_1759199/k15.stored_hashes.independent_audit.json)）。[六库原存栅格集合检查](../../runs/20261102/EXP-INV-010_202611020411/stored_grid_six_library_overlap_audit.json)的两两精确交集在 64 栅格均为零；有原存 256 栅格的五库之间也均为零。这不包含 F50 的 256/512 重建或未盘点数据，不能推论物理等价的完整排除。

DAT-007 全部 12,000 条和 DAT-008 全部 24,000 条的几何重建已逐行对照来源及原存 64/256 栅格哈希，未发现差异；每个分片各抽取四例在 WSL 独立重建 64/256/512 栅格一致（[DAT-007 首片](../../runs/20261102/EXP-INV-010_202611020411/geometry_dat007_6659656/audit.json)、[中间分片](../../runs/20261102/EXP-INV-010_202611020411/geometry_collection_6659655_6659710/four_shards_audit.json)、[DAT-008 其余分片](../../runs/20261102/EXP-INV-010_202611020411/geometry_dat008_remaining_6660069/remaining_shards_audit.json)）。F50 已完成前 10,000 条的重建与来源核对，每个分片各有四例独立重建一致；其原存 64 与重建口径须分别保留。以上均是有限范围身份检查，不是完整排除或物理验证。

限定清单、各输入哈希、生成代码、配置、参考 k 点、检查范围及已知缺口经独立核验并固定后，才可按新种子生成 1,024 来源；再逐例检查已有排除身份及 k 点，固定网格粗筛后按预定三层各选 4 个来源。层不足则停止，不改种子补样。后续候选也必须逐条检查固定清单并保留异常；未覆盖的历史身份不得被描述为已验证无碰撞。

## 搜索、物理和统计合同

目标为 TE B4–B5、频率 $a/\lambda=0.57$，名义及 $\pm1/512$ 的共同目标裕量；12 来源分别以三 checkpoint 搜索，每条最多 20 次评估。保留原谱及带序异常；投影代理最高分与同轨迹有效、去重的非锚点集合均匀随机选择成对比较。完整锁定所有轨迹、选择、候选、六个跨环境角色及哈希，才可开始物理计算；来源与随机种子规则见参数草案。

最多 84 个去重结构接受 M9 真实 MPB 三偏移、512 几何及 192/224 分辨率自适应带边验证；六个预定角色在 WSL Ubuntu `meep_env` 以 $10^{-6}$ 全谱绝对容差对照。所有结构通过冻结数值门、六例通过、所有来源和选择可评分，且先来源内平均三个 seed、再三层等权来源 bootstrap 的双侧 95% 区间下界大于零，才满足本开发条件；任何缺失或失败保持不可判定。它不替代原 P2 和后续 P3–P5 门。

物理阶段分配核时上限 135，包含失败与重试；5 核、5 worker、12 GiB、90 分钟为拟定配置，先以首批确认实际配额、内存和成本再扩批。粗筛、GPU 与 WSL 成本另列；运行中或状态未知的作业不重复提交或取消。原 [EXP-INV-009](EXP-INV-009-目标驱动代理搜索.md) 代理证据和准入失败保持独立，不重复旧批。
