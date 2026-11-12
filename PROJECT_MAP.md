<!--
  Current evidence gates for the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->


> EXP-INV-012 来源8的[设计与物理复核结果](docs/results/inverse/EXP-INV-012-来源8梯度与物理复核.md)为 Draft：12个锁定角色对应11个硬结构，均通过 M9 局部数值门，六个预选角色的 WSL 对照通过；正梯度两个步长均改善真实裕量，Adam 终点数值上低于初始且差异小于数值容差，DE／Random 共同终点不覆盖目标频率。[EXP-INV-013 三来源结果](docs/results/inverse/EXP-INV-013-三来源纯代理对照.md)中27条 N56 轨迹已锁定，61个合格唯一硬结构完成 M9 复核，预选六角色 WSL 对照通过；Adam、DE、Random 的 MPB 目标覆盖分别为3/9、0/9、0/9。N40 原 Adam 数组 `5533238` 在排队时取消。[EXP-INV-014 反馈结果](docs/results/inverse/EXP-INV-014-物理反馈对照.md)已完成 M9 数值评价和六个预选角色的 WSL 独立复核；残差改善局部带边误差，但设计质量收益随来源变化。[EXP-INV-015 鲁棒对照](docs/results/inverse/EXP-INV-015-制造鲁棒对照.md)已完成两路线全部终点及预选 WSL 复核；纯代理和校正路线各27个终点均未在 $\pm1/512$ 三场景中形成覆盖0.57的共同带隙。[EXP-INV-016 独立来源验证](experiments/research/EXP-INV-016-独立来源与效率验证.md)已冻结方法并抽取5个测试划分来源，N56 `1774529`、`1774530` 的180条两参数与四参数纯代理轨迹已完成并核验；[独立来源配对结果](docs/results/inverse/EXP-INV-016-独立来源与物理反馈结果.md)已完成纯代理与反馈终点三场景复核及10个预选角色的 WSL 高精度对照；各路线均未覆盖目标，局部残差误差降低但未显示独立设计优势。

> 2026-11-04 逆向研究按作者要求重规划为[可微代理与物理反馈研究](docs/planning/鲁棒带隙逆向/研究总方案.md)：完整主线为纯代理可微设计＋少量 MPB 观测驱动的代理校正/微调；先交付纯代理基线，再实施物理适配、制造鲁棒与独立效率研究。[EXP-INV-012（MVP-INV-001）](experiments/research/EXP-INV-012-梯度链条开发验证.md)已完成来源8原点子批复核；工作包见[推进与交付](docs/planning/鲁棒带隙逆向/推进与交付.md)。前版方案和旧实验保留历史身份；下方旧逆向快照为历史。

> 2026-11-04 [EXP-FWD-033 等量混合分支消融](experiments/research/EXP-FWD-033-等量混合分支消融.md)：两机 48 项训练完成，训练产物的独立哈希、来源、模型身份与历史复核通过；N56 的 24 项 validation 重算通过，六目标测试数组 `1769140` 已提交。N40 的 validation 数组 `5532107` 等待调度，依赖作业 `5532109` 在重算及逐项核验通过后冻结并提交测试。新测试结果尚待完成和审计；原失败与补跑证据保留。

> 2026-11-02 [EXP-FWD-032 单域分支消融](docs/results/forward/EXP-FWD-032-单域分支消融.md)：72 个新增 checkpoint 的 432 个六目标代理测试 cell 完成；四个原域的普通 Full 三 seed 平均 MRE 均低于普通单支，容量对照不保持该顺序。N40 接管原 N56 测试并保留来源；3 个 F50 Graded 容量对照 seed 耗尽训练预算，历史 3×3 逐 cell split 哈希待回查。结果为 Draft，物理验收未包含。

> 2026-10-29 四域多指标补充评价为 partial：已冻结 1,176 个单元的范围，历史 K5 两域三模型的 18 个原域单元通过 WSL 多指标补算及 MRE 回归；历史 960 个单元的预测文件已定位，031 的 216 项需补导出预测，远端作业待资源授权。见[补充评价结果（Draft）](docs/results/forward/EXP-FWD-031-多指标补充评价.md)与[补充协议](experiments/research/EXP-FWD-031-多指标补充评价.md)。这不改变[原四域 MRE 完整报告](docs/reports/EXP-FWD-031四域完整测试结果_20261029.md)的测试完成状态。下方早期执行快照保留为历史。

> 2026-10-27 EXP-FWD-031 已登记为 Draft：[官方算子与通道融合对比](experiments/research/EXP-FWD-031-官方算子与通道融合对比.md)。模型实现与训练协议的工程验证、双服务器准入和 36 项正式提交分别核查；未完成准入前不得称训练已启动。此批仅含训练与 validation，不包含 held-out test。

> 2026-10-27 论文展示范围按作者要求改为K5/F50的Binary/Graded四域，见[四域结果对比（Draft）](docs/reports/K5与F50四域论文结果对比_20261027.md)。所有主表从逐seed结果重算；K20作为补充、原六域报告保留。此为L3展示范围调整，不修改实验或checkpoint，不将原K5+K20混合训练改称四域训练，也不根据新排名重新选模型。

> 2026-10-27 GL 7×7正式测试完成：EXP-FWD-029/030的48个冻结checkpoint已完成672/672个唯一测试cell，224组三seed均值与样本标准差经联合审计和本轮复算通过，覆盖268.8万条checkpoint×样本预测。正式推理全部在N56执行，原N40分片按执行调整接管，来源训练平台和冻结身份保留。结果已加入[全模型论文数据汇总](docs/reports/EXP-FWD-026-028全模型论文数据汇总_20261025.md)：7×7原域六域平均MRE为0.3015%，与3×3的0.2989%接近；12K/24K相对3×3均值改善，但没有跨任务一致优势。validation与test分开，mixed自身test split阻塞和物理验收边界继续保留。下方旧执行快照为历史。

> 2026-10-22 收敛训练已按用户授权提交：N40的EXP-FWD-026/027共54+30项从原300轮validation-best权重热启动；N56的EXP-FWD-028为六种其他模型、六单域8k、三个seed共108项从零训练。两平台各最多8张GPU，验证平台期停止、累计1000轮上限，原200/300轮证据保留。N40准入5509373与依赖训练5509407已提交、仍待调度；N56全部36项准入通过，修订版训练1722717已在8卡运行。两平台六域的数据、split、8k子集和train-only normalizer冻结记录一致；第301轮统一从validation-best权重重置AdamW和随机状态，正式训练源码为8b2d402；N56初始协议运行已保留并停止，不计入修订版结果。当前没有新的训练完成或收敛结论。协议、配置、失败门与证据位置见[同条件三seed收敛训练](experiments/research/EXP-FWD-026-收敛训练协议.md)。持续目标与原自动监控保持暂停，作业本身按冻结协议执行，不自动重试。

> 2026-10-22 EXP-FWD-024 评估已完成：N40 的 54 个冻结 checkpoint 均完成各自8k训练集重构、原生 held-out 测试和 EXP-DAT-007 12k / EXP-DAT-008 24k 新几何直接迁移；新评估 162 单元、237.6 万逐样本行及来源哈希经独立审计通过。F50 Binary ConvNeXt 在两批新几何的三 seed Bands 1–10 MRE 为 0.217% / 0.354%；仅13/54训练过既定收敛门。新几何分辨率收敛和正式 WSL MPB 复算仍待验证，不作物理或充分优化主张。详见[Draft 结果](docs/results/forward/EXP-FWD-024-训练集与新几何迁移测试.md)；下方早期执行快照不代表当前状态。

> 2026-10-20 补充实验：六单域各8k、三个核心模型、三个seed、从零300epoch的 EXP-FWD-024 的18项 admission 已通过；N56部署出现文件系统延迟，依据既有N40/N56授权，54项核心训练改在N40独立部署提交（5506475），最多4并发；EXP-FWD-025 的 N40 十项 admission 已通过，30项分支重训已提交（5506441），尚无完成或新性能结论。EXP-DAT-007 的正式参数计划已由用户扩展为六类孔形各2000、共12000结构；r64/r96/r128数值pilot仍是 `waived_by_user` 而非收敛通过。星仪权限失败和Bingxing旧1200计划目录均保留；旧solve `6258385` 与finalize `6258386` 在未求解任何标签前取消。新运行时代码提交 `88a621d` 部署至Bingxing M9，冻结目录为 `/publicfs10/fs10-m9/home/m9s005801/runs/20261020/EXP-DAT-007_12000_20261020_bingxing_4x64_v1`，采用4个独立shard、每shard 64 CPU/64 GB。plan任务 `6259235` 已完成：12000行、六族各2000、0哈希碰撞；solve array `6259236_[0-3]` 与finalize `6259237` 已提交。当前只可称“参数计划完成、求解已提交”，集群生成、传输哈希与WSL验收仍须分别记录。详见[固定预算确认](experiments/research/EXP-FWD-024-三核心模型固定预算确认.md)与[单孔形状迁移协议](experiments/research/EXP-DAT-007-单孔形状迁移测试.md)。

> 2026-09-20 EXP-FWD-022 状态：1,080 个冻结 validation 干预单元与 30/30 独立训练均保持完整，终态与哈希审计已通过。联合对照限定 K5/K20、Binary/Graded、3×3：按三 seed 均值为 2 域同向、2 域反向；加入 seed 方向门后为 1 域稳定同向、2 域稳定反向、1 域冻结排序不稳定。重训能力在两 lane 均稳定表现为 K5 的 G-only 更好、K20 的 L-only 更好；Full 在固定 200 epoch mixed 主指标上仍优于普通及参数量控制单支，但仅 11/30 通过末段收敛门。结论保持 validation-only、capability_complete_pending_human_review，不作 test/physical、充分优化能力或 promotion 主张；下一 gate 为人工审阅联合判读。详见[分支依赖与独立能力 Draft](docs/results/forward/EXP-FWD-022-分支依赖与独立能力.md)。

> 2026-08-07 工程状态：通用超算物理 campaign 控制面与 `full_cell_band` 参考适配器已实现；query/k-path、field-eigen、resolution-convergence 与 inverse replay 仅完成适配器登记，尚未完成多节点迁移。详见 [`docs/hpc_physics_campaigns.md`](docs/hpc_physics_campaigns.md)。这不是新的物理验证结论。

> 2026-08-10 升级评估状态：五个 Full-Wh 升级模型已完成 50k 正式主评估。ConvNeXt 是 B/G native held-out test 共同最优，Mixer 与 TFNO 也通过直接基线 promotion；Gated ResMLP 退化，dual-lane 未达到联合替代门。threshold proxy 因 `AssocMaxSubmitJobLimit` 未提交成功；没有 real-MPB inverse replay。详见[升级结果](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)与[全胞模型报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md)。

> 2026-08-13 维护状态：CLI、模型与 inverse runtime 模块化已分别提交为 `082ac2d`、`5c01bf9`，架构边界记录于 `75bab09`。WSL `Ubuntu` + conda `meep_env` 下 compileall、基础 CLI、`models list`、40 项定向测试与完整 `482 passed` 回归均通过；这只证明 L4 工程兼容性，不新增模型性能或物理验证结论。同期生成的多份 `paper_like_64` representation benchmark 仅为重复小样本 smoke，不构成新的正式模型或物理结论。

> 2026-08-25 EXP-PHY-004 状态：K5-B/G 各 1000 条 frozen test 的 Full-Wh 频谱审计与 6 条代表结构的 real-MPB/18 条 real-Meep response sensitivity 已完成。整体 MRE 较低不保证最大带隙 band/k 选择稳定；该结论不等于 field surrogate 已建立，也不升级 inverse 状态。详见[代理选频与真实响应差异](docs/results/field-physics/代理选频与真实响应差异_20260825.md)。

> 2026-08-25 EXP-FWD-009 状态：五个 epoch-200 权重热启动身份均通过主收敛门，最终 17 个 K5 身份完成六域 510-cell、408/272 数值回归和 102-cell paired bootstrap。最终 inventory 仅 9/17 满足主收敛门，另 8 个依据人工豁免纳入；不得写成原 EXP-FWD-008 的 17/17 收敛通过。详见[EXP-FWD-009 结果](docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md)。

> 2026-08-26 EXP-FWD-010/011 状态：joint 改为 K5 Binary/Graded 各 4,000 条后完成 seed0 200 轮从零训练、200→300 权重续训和两组六域评估。300 轮时 Binary 收敛通过，Graded 与等权未通过；相对 200 轮为五域显著改善、K20 Graded 退化，只在 K5 Binary 超过最佳单路线。该结果仍是 surrogate seed0 负对照，不支持 joint 全域 promotion。详见[EXP-FWD-009/010/011 汇总](docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md)。

> 2026-08-27 EXP-FWD-012 状态：公平 4k+4k K5 train/validation 的工作频率先验审计完成。$X/M$ 一阶 Bragg 尺度对两条 lane 的最大 gap-center 命中均为 0，未通过相对 train-only 固定中心改善门；EXP-FWD-013 的四个 loss-only 训练任务未启动。后续 L4 已实现不依赖 Bragg 中心的可选 `equal_lane_gap_aware_v1`，并通过真实 K5 CPU admission smoke；该状态仅为 `implemented + smoke_pass`，尚未登记或执行正式性能实验。详见[工作频率先验负结果](docs/results/forward/EXP-FWD-012-工作频率先验覆盖审计.md)。

> 2026-08-28 EXP-FWD-014 状态：两个 N40 seed0、20 epoch 配对任务正常完成。新损失把 Binary/Graded joint band-k accuracy 分别提高 `10.9/15.1` 个百分点，并改善多数 gap 指标；但 Bands 2–10 MRE 分别恶化 `13.44%/12.65%`，且 complete-gap 子集状态命中仍为 `0`，因此 `gap_screen_fail`，停止自动三 seed 扩展。详见[EXP-FWD-014 结果](docs/results/forward/EXP-FWD-014-gap决策损失短预算筛选.md)。

> 2026-08-28 EXP-FWD-015 状态：对 EXP-FWD-014 两个冻结检查点完成 K5 validation-only 近简并审计。train-only 容差最多只提高 joint accuracy `3.0` 个百分点，排除近简并后 v1 仍只有 Binary/Graded `27.7%/24.8%`；结论为模型决策受限，而非指标近简并主导。`equal_lane_gap_ranked_v2` 已实现并通过 WSL CPU admission smoke，但 EXP-FWD-016 的一个 N40 短筛任务尚未授权或提交。详见[EXP-FWD-015 结果](docs/results/forward/EXP-FWD-015-gap决策近简并审计.md)。

> 2026-08-28 EXP-PHY-005 状态：192 结构 near-Γ TE real-MPB 正式验证完成，数值门通过，但 Binary 方向差异 P90 `5.36%` 超过 `5%`，终态为 `scalar_anisotropy_fail`；TE 调和候选在 Binary/Graded 的 median slope 误差为 `24.81%/20.71%`，显著劣于旧公式 `7.77%/5.83%`。`x` 虽稳定选择 B1–B2，却未落入 complete gap，不能视为已找到工作频率。EXP-FWD-017 不解锁。详见[正式结果](docs/results/field-physics/EXP-PHY-005-TE有效折射率与折叠频率校准.md)。

> 2026-08-29 EXP-PHY-006 状态：4 个冻结 validation 结构的 WSL real-MPB 局部 band-edge 加密 pilot 已完成。现有 $9\times9$ 标签在 32/36 个相邻 pair 上高估 gap 超过预注册阈值，并出现 1 次 complete→pseudo 翻转；4/4 的 B1–B2 均为局部认证 pseudo-gap。该证据证明粗网格风险，但不是连续全 BZ 的全局认证；相关训练继续锁定。详见[试点结果](docs/results/field-physics/EXP-PHY-006-全BZ带隙边缘加密试点.md)。

> 2026-09-02 EXP-PHY-007 状态：K5 Binary/Graded 全部 64 个冻结 validation 结构完成 WSL real-MPB 自适应 band-edge 加密，0 solver failure；但仅 18/64 个结构同时满足全部九个 pair 的连续 edge 收敛与 resolution 64→96 稳定，终态为 `adaptive_label_gate_fail`。EXP-FWD-016 未提交，loss 路线按计划停在 L0。详见[标签门结果](docs/results/field-physics/EXP-PHY-007-validation带隙标签门.md)。

> 2026-09-08 EXP-FWD-019 状态：C0、21项 seed0 接受谱系和42项 F50 validation-only 三种子确认已完成。ConvNeXt $7\times7\to1\times1$ 在 B/G 两路线、三种子上稳定大幅退化，是当前容量混杂最小的组件信号；FNO/Joint 删除 spectral 也稳定退化，但参数量同时减少约90%/86%；TFNO `no_channel` 未通过 Graded 方向一致性门。23/42项终态未满足收敛门，六域冻结评估、held-out test 和物理验证均未执行。详见[三种子确认结果](docs/results/forward/EXP-FWD-019-组件归因三种子确认.md)。

> 2026-09-12 EXP-FWD-020 状态：36 个冻结 checkpoint（7 个原 G2 validation-best、29 个 epoch-300 continuation validation-best）已完成 K5/K20/F50 Binary/Graded 的 216-cell held-out surrogate 交叉评估。native Bands 2--10 MRE 均值为 `0.297%`，同数据集跨 lane、跨数据集同 lane、双重跨域依次为 `2.703%`、`3.295%`、`5.409%`；7x7 对 3x3 没有跨域一致优势。G2 仍为 `not_passed`，结果仅为探索性 surrogate 证据，不作 promotion、real-MPB/Meep 或物理结论。详见[六域交叉评估 Draft](docs/results/forward/EXP-FWD-020-全局局部带隙算子六域交叉评估.md)。

> 2026-11-01 [非生成式鲁棒带隙逆向长期计划（Draft）](docs/planning/鲁棒带隙逆向/README.md)仍在探索性 P2 审计：[逆向可用性结果](docs/results/inverse/鲁棒带隙逆向可用性审计.md)中，原 B9–B10 开发批在 ±1/256 偏移下 0/60 有正共同区间；B4–B5、512 栅格及 ±1/512 的另一开发合同下，60 候选有 45 个过门、4 个过门正裕量，且一个锁定选择的候选通过 M9 与 WSL 单样本复核。两批目标与扰动合同不同；P2 扩大搜索条件尚未满足。另一独立开发批 60 候选中 59 个过数值门，但全来源统计不可用且跨环境复核为 3/6 通过。更严格容差和分辨率的新开发批已完成 1,024 来源粗筛、12 来源选择和三个冻结模型预测；一个来源的锚点带序异常，已不能满足全来源可评分门。该批 60/60 候选通过冻结有限数值门、6/6 预选 WSL 对照通过，但唯一正裕量来源因目标带序异常不可评分，全来源选择统计不可判定，P2 尚未通过。

> 后续 [EXP-INV-008 独立开发批](experiments/research/EXP-INV-008-投影校正独立开发.md)60 候选中 58 个通过冻结数值门，预选六例 M9–WSL 全谱对照通过；因两个非锚点数值失败，完整来源统计不可判定。该批不满足全部候选数值门，不改变上一批 P2 未通过结论。[条件判定](runs/20261101/EXP-INV-008_202611011652/development_decision.json)

> [EXP-INV-009 目标驱动代理搜索](experiments/research/EXP-INV-009-目标驱动代理搜索.md)冻结前遗漏两份共 10,064 条 forward 数据，事后未发现精确身份碰撞，但未满足事前完整排除条件，84 例物理验证未准入。[EXP-INV-010](experiments/research/EXP-INV-010-事前排除目标搜索.md)的限定开发条件已满足：84/84 M9 案例通过冻结数值门，预选 6/6 案例通过 WSL 有限网格全谱对照；12 来源的代理选择相对同轨迹随机对照，目标裕量平均差为 0.01757，分层 bootstrap 95% 区间为 [0.01041, 0.02643]。历史覆盖仍有限，原 P2 失败与 P5 未完成状态不变。

> [EXP-INV-011 名义物理反馈开发预检](experiments/research/EXP-INV-011-名义物理反馈开发预检.md)为 Draft，拟比较三个已观察来源的静态代理与多轮反馈；输入身份绑定、配对基线及可恢复交接尚待验证，尚无执行锁或新物理结果。


# PROJECT MAP

| Metadata | Value |
|---|---|
| Status | Draft current state snapshot pending human review |
| Audience | 导师、研究生与项目负责人 |
| Owner | 研究生维护，导师审阅科学主张 |
| Last reviewed | 2026-10-21 |
| Reading time | 5 minutes |

**当前工作已形成全胞正向的多域测试、分支依赖与固定预算确认链；最新训练尚未全部完成，单孔数据尚待 WSL 验收。** 本页是当前状态入口，不保存调度流水账。原始配置、日志、模型、哈希与求解器证据继续留在 runs；历史结论通过结果页追溯。

[项目入口](README.md) · [文档目录](docs/README.md) · [当前结果](docs/结果/README.md) · [实验登记](experiments/registry.yaml) · [协作治理](docs/collaboration.md)

工程入口：[架构兼容入口](docs/architecture.md) · [数据目录](data/README.md)。

## 当前研究问题

项目研究如何在明确的数据、模型和证据合同下，用神经代理加速二维光子晶体能带计算，并将候选结构交给 real-MPB/Meep 复核。低 surrogate 误差、代理排序、真实 replay 和物理成功是不同结论。

当前重点是全胞正向模型在 K 截断、Binary/Graded 路线与结构分布变化下的可靠性。Full-cell inverse 和 Field/PINO 的未通过门不因正向更新而解锁。

## 最新证据与最近一门

| Experiment ID | 已确认状态与边界 | 下一 gate |
|---|---|---|
| EXP-FWD-020 | 36 个 checkpoint、216-cell 六域 held-out 评估完成；G2 `not_passed`，仅探索性 surrogate | 人工接受[六域结果 Draft](docs/results/forward/EXP-FWD-020-全局局部带隙算子六域交叉评估.md)，不作 promotion |
| EXP-FWD-021/023 | 63 个固定 seed0 checkpoint、756/756 十二域 cells 与 7,560 条逐 band 记录；K8/K10/K15 从未进入训练、validation、normalizer 或选模 | [完整结果](docs/results/forward/EXP-FWD-023-七种训练制度十二域完整结果.md)人工审阅；新数据仍为 `cluster_audited_pending_wsl_acceptance` |
| EXP-FWD-022 | 1,080 冻结 validation 干预与 30 项独立训练完成；仅 11/30 过末窗收敛门，无 test | 人工审阅[依赖与独立能力](docs/results/forward/EXP-FWD-022-分支依赖与独立能力.md)，不解释为充分优化能力上限 |
| EXP-FWD-024 | 2026-10-21 只读核验：54 项中 44 个训练终态，11 个收敛、33 个未过末窗门；4/6 bundle 标记，未见 test manifest | 按[冻结合同](experiments/research/EXP-FWD-024-三核心模型固定预算确认.md)等待完整训练审计后才可 native test；现有数字不是最终结果 |
| EXP-FWD-025 | 同次核验：30 项中 9 个训练终态，3 个收敛、6 个未过末窗门；无 test manifest，test 保持锁定 | 等待全部训练和完整性审计；不与旧 200 epoch 结果混用 |
| EXP-DAT-007 | Bingxing 四 shard 合计 12,000/12,000，0 failure，`cluster_generated_pending_wsl_acceptance`；分辨率 pilot 为 `waived_by_user` | [单孔合同](experiments/research/EXP-DAT-007-单孔形状迁移测试.md)要求下载哈希核对与 WSL 跨环境验收；尚无新模型迁移结论 |

024/025 的现有证据定位为 N40 `runs/20261020/EXP-FWD-024_202610201920` 与 `EXP-FWD-025_202610201920` 下的训练终态、admission audit 和 bundle 标记。007 的证据定位为 Bingxing `runs/20261020/EXP-DAT-007_12000_20261020_bingxing_4x64_v1` 下的 results、metrics、campaign_status 和 shard 记录。这些是只读核验时的快照，不代表此后运行状态不变。

## 保留的研究边界

| 方向 | 现有结论 | 限制与入口 |
|---|---|---|
| 全胞数据底座 | 已有分布的 r64 通过数值门，历史 B/G 数据已有验收链 | 新单孔分布不能继承旧分布的精度结论；见[全胞数据](docs/results/全胞数据正式结果.md) |
| 历史 50k 正向 | ConvNeXt 的 B/G native 结果属于其明确模型和冻结协议 | [升级结果](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)不能借给 Global–Local；final-fit monitor 不能代替 test |
| 组件与 gap 决策 | 组件删除存在容量混杂，gap loss 有短预算负结果 | [组件结果](docs/results/forward/EXP-FWD-019-组件归因三种子确认.md)、[gap loss](docs/results/forward/EXP-FWD-014-gap决策损失短预算筛选.md)；不自动调参或用 test 追调 |
| 标签与物理选频 | 粗网格可高估 gap；低全谱 MRE 不保证 band/k 选择正确 | [标签门](docs/results/field-physics/EXP-PHY-007-validation带隙标签门.md)、[真实响应差异](docs/results/field-physics/代理选频与真实响应差异_20260825.md) |
| Field / adapted PINO | 校准与准入仍 blocked | [历史准入](docs/results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md)；不进入候选排序 |
| 生成式 inverse | V2 Objective-Bridge 仅有 seed0 replay；full-cell 正式闭环尚未完成 | [inverse 结果](docs/results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md)、[当前协议](docs/planning/全胞逆向索引.md) |

架构调整计划处于 Draft，首批拟整理正向 Dataset 和模型调用接口，验收以兼容及行为对照为准。[架构调整与迁移计划](docs/planning/项目代码结构重构/架构调整与迁移计划.md)

## 实现与交付入口

- 代码、CLI、配置与 tests 证明实现合同；正式工程验收使用 WSL Ubuntu / meep_env，不能自动升级科学证据。
- 模型必须同时核对[模型目录](docs/参考/模型目录.md)、[机器清单](docs/reports/phc2d_model_inventory_v1.yaml)和未注册的 inverse 方法。注册、训练、验证、test 与物理验收保持分开。
- 图件、编辑源与来源见[图件索引](docs/reports/图件索引.md)；历史 Bands 2–10 图不能混作 024 的 Bands 1–10 主表。

## 当前人工决策

人工审阅已有 Draft 结果。研究侧最近一门是完整训练审计和新单孔数据验收；任何新训练、正式物理验收或更强主张均须按各自合同推进，不能由文档整理替代。
