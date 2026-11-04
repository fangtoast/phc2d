<!--
  Concise project map for the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

> EXP-INV-012 来源 8 四个 N56 设计分片已完成并通过身份、梯度和硬选型核验；[设计 run 与候选锁](runs/20261105/EXP-INV-012_202611050000/)保留 12 个角色，其中 11 个不同硬结构。M9 物理复核数组 `6758052` 的四个分片已提交；物理结果尚待返回。方向诊断 `1773777` 和 Adam `1773778` 接管了 N40 原作业 `5533061`、`5533062`，后者在排队时取消且运行时间为零；DE 与 Random 分别为 `1773091`、`1773093`。设计代码 `050ac8cb` 经[跨平台预检](runs/20261104/EXP-INV-012_202611041803/preflight/)通过，接管控制代码为 `cc7d32ef`。013 三来源批次和 014 多轮反馈尚需实现与验证，物理及鲁棒结论尚未产生。

> 2026-11-04 逆向研究按作者要求重规划为[可微代理与物理反馈研究](docs/planning/鲁棒带隙逆向/研究总方案.md)：完整主线为纯代理可微设计＋少量 MPB 观测驱动的代理校正/微调；先交付纯代理基线，再实施物理适配、制造鲁棒与独立效率研究。[EXP-INV-012（MVP-INV-001）](experiments/research/EXP-INV-012-梯度链条开发验证.md)的来源 8 原点子批已锁定，下一步收集并核验 MPB 结果；工作包见[推进与交付](docs/planning/鲁棒带隙逆向/推进与交付.md)。前版方案和旧实验保留历史身份；下方旧逆向快照为历史。

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

> 2026-10-21 耗时证据冻结：EXP-DAT-007/008共36000结构已全部生成、失败0并交付N40；18族逐结构计时已与HDF5样本顺序核对，旧/新批次求解实际工期分别为11分53秒/21分02秒。详见[MPB形状数据计算耗时汇总](docs/results/data-provenance/MPB形状数据计算耗时汇总.md)。下方提交时快照不再代表当前任务状态；正式WSL MPB复算验收仍pending，新几何分辨率收敛未评估。

> 2026-10-21 数据补充：EXP-DAT-007 的12000条真实MPB标签已完成M9生成并交付N40，三端SHA-256一致；WSL正式验收状态仍须单列。用户已认可新图谱并授权 [EXP-DAT-008 PCSEL十二族](experiments/research/EXP-DAT-008-PCSEL多晶格测试数据.md) 各2000点、共24000结构在M9求解并交付N40。全量几何门通过，427组相同栅格输入保留不同解析身份；12族WSL真实MPB接口准入及74项定向回归通过。代码 `f3f7360` 已隔离部署，冻结运行 `EXP-DAT-008_202610211514` 的plan `6284297` 已完成，solve `6284298_[0-3]` 四片均运行，finalize `6284299` 等待依赖；每片64 CPU/64GB。30分钟监控 `mpb-24k-n40` 已启用，完成后核验哈希并交付N40。当前不称新数据生成或交付完成，不宣称新族r64数值收敛或完整器件复现，不执行模型评价。

| Metadata | Value |
|---|---|
| Status | Current state snapshot |
| Audience | 导师、研究生与项目负责人 |
| Owner | 研究生维护，导师审阅科学主张 |
| Last reviewed | 2026-08-28 |
| Reading time | 8 minutes |

[项目封面](README.md) · [书籍式文档目录](docs/README.md) · [当前实验进展](docs/results/README.md) · [协作治理](docs/collaboration.md)

> 本文件是“当前状态”的唯一短入口。“已实现”不等于“已完成研究验证”；surrogate、field residual 与 real-MPB replay 证据不得混用。具体数值、配置和运行命令继续留在下层文档。

## 项目要解决什么问题

`phc2d` 面向二维光子晶体，研究如何用神经网络/神经算子可靠加速 Maxwell/MPB 求解，并从目标能带反向生成可由 real-MPB 验证的结构。

## 当前研究主线

当前逆向研究为**可微 GL 纯代理基线 + MPB 反馈驱动的代理校正/微调 + 制造共同带隙**，见[研究协议入口](docs/planning/鲁棒带隙逆向/研究总方案.md)：

`目标 → 可微 Fourier/GL 优化 → 纯代理选型与复核基线 → 少量 MPB 批次适配代理 → 再优化 → 制造共同带隙与独立复核`

正向研究提供全胞数据与 Global–Local 底座；新逆向阶段先完成纯代理设计，再以 MPB 观测适配代理并比较收益，随后扩展制造鲁棒及独立评估。旧生成式 Replay-Calibrated Hierarchical Inverse Design 保留独立历史与准入合同；V2 p4m 的既有 replay 不能替代新路线验证。

## 模块与当前进度

| 模块 | 负责什么 | 当前做到哪里 | 详细入口 |
|---|---|---|---|
| 数据与 MPB 物理底座 | 结构、k 路径、标签、provenance | V2 p4m 已物理复核；全胞 B/G 100 结构五档 MPB resolution 技术门通过并推荐 r64，待负责人冻结；二值与渐变两套 10k 均已通过 WSL 验收 | [数据目录](data/README.md) · [全胞试点](docs/results/data-provenance/full_cell_dual_lane_pilot_20260803.md) · [MPB 分辨率验证](docs/results/data-provenance/全胞MPB分辨率收敛验证_20260817.md) |
| 表征与任务协议 | `rho`/`epsilon`、Full-Wh、Query、Field | spec、target 与 NPZ/HDF5 schema inspection 已实现；小样本表征实验仅属诊断 | [研究计划](docs/planning/fno_generative_network_adjustment_plan_20260617.md) · [公式手册](docs/reports/phc2d_physics_data_formula_guide_20260803.md) |
| Forward / Query | 完整能带与任意 `(band,k)` 查询 | 10k 参考矩阵和 50k baseline 冻结评估均完成；五个升级模型的 50k native/cross-lane/resolution 主评估完成，ConvNeXt native 最优，threshold proxy 仍受 QoS 阻塞 | [全胞模型报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md) · [升级结果](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md) · [50k baseline](docs/results/forward/full_cell_dual_lane_50000_forward_training_20260808.md) · [10k 结果](docs/results/forward/full_cell_dual_lane_forward_training_20260806.md) |
| Field / adapted PINO | TE Bloch 本征场与物理残差 | U-Net、Field FNO、adapted PINO 已实现；介电加权指标需重算，residual–frequency 门未过，inverse 准入仍 `blocked` | [指标修正](docs/results/field-physics/field_electric_metric_correction_20260803.md) · [准入结果](docs/results/field-physics/field_eigen_te_b4_dense_k_admission_20260722.md) |
| 可微代理与物理反馈逆向 | 纯代理梯度、MPB 适配与制造共同带隙 | 规划按工作包整理；优先交付来源 8 的纯代理 MVP，再实施可微残差及批次反馈；具体 checkpoint、执行预算和新结果仍待完成 | [路线](docs/planning/鲁棒带隙逆向/研究总方案.md) · [模块复用](docs/planning/鲁棒带隙逆向/实施与资源.md) |
| 生成式逆向与 replay | 多候选、筛选、真实 MPB 复核 | objective-bridge 已完成 seed0 的 100-target 与 test625 replay；full-cell codec 与真实 10k 的 15-case Stage-A CPU 合同已通过，N40/正式训练/replay 尚未准入 | [逆向 seed0](docs/results/inverse/a2_inverse_objective_bridge_seed0_v1_20260727.md) · [Stage-A CPU 预检](docs/results/inverse/full_cell_inverse_stage_a_cpu_preflight_20260806.md) |
| 工程与证据治理 | CLI、配置、run 产物、审计 | 统一源码/配置/实验/运行结构已建立；CLI、模型与 inverse runtime 模块化已通过 WSL 完整工程回归；N40 已证明可同时分配 3×RTX4090，N50 登录仍受平台阻塞 | [架构说明](docs/architecture.md) · [协作治理](docs/collaboration.md) · [GPU 准备报告](docs/results/data-provenance/paracloud_gpu_preflight_20260806.md) · [仓库约定](docs/conventions.md) |

## 当前最重要的结论

1. 新逆向主张按“参数梯度 → 纯代理真实设计 → MPB 代理适配收益 → 制造鲁棒与独立效率”检验；当前是 Draft 规划，只有相应求解和数值验收能支持物理结论。
2. Forward / Query 及旧生成式 inverse 已有各自对照；010 提供有限开发合同下的代理选择证据，尚不支持新路线的参数梯度、反馈校正或普遍优越。
3. residual 下降不等于频率或 inverse 改善；adapted PINO 在校准通过前不得进入候选排序。
4. 二值与渐变全胞 10k 均已通过固定 WSL real-MPB 跨环境验收；这只解除数据门，不等于模型训练已完成。
5. M9 已补齐 binary/graded 两份 10k 数据并与 N40 使用同一静态部署 commit `b93cc2d`；这只完成文件/代码交接，不等于新的 M9 environment preflight 或 inverse replay。
6. 50k 全胞 Forward 的 40k held-out test 中，FNO 在全部 12 个方向/epoch 组均为最低三 seed 均值；binary 100/200 epoch 均为约 `0.630%`，continuous 200 epoch 为约 `0.160%`。50k final-fit 和 threshold 结果只属于训练重叠 monitor。
7. CNN/FNO 能接受 32/128 输入不等于分辨率不变；相对 64 的 156 个 paired comparisons 中有 151 个 MRE CI 显著变差。该结果不是 MPB 分辨率收敛或物理验证。
8. Full-Wh 升级 50k held-out test 中，ConvNeXt 的 B/G MRE 为约 `0.321%/0.050%`；Mixer 的 32/128 直接输入诊断总体最稳。Gated ResMLP 负结果和 dual-lane 未通过替代门必须保留。
9. K5 Full-Wh 的整体 test MRE 不能替代最大带隙选择审计；EXP-PHY-004 的 6 条代表结构中，ConvNeXt 仅 1/6、joint 仅 4/6 同时保持 real-MPB band 与 k 选择，选频差异可显著改变冻结 real-Meep 响应。
10. EXP-FWD-019在F50 validation上支持ConvNeXt大核的稳定贡献信号，并确认FNO/Joint频谱分支不可无损删除；后两者仍有显著容量混杂，TFNO channel不具备两路线一致证据。

## 当前最重要的未解决问题

- EXP-FWD-009 已完成五个身份的权重热启动续训与六域 510-cell held-out surrogate 评估；54 身份 preflight、10 shards、408/272 数值回归和 102-cell paired bootstrap 均闭合。最终 inventory 仍只有 9/17 满足主收敛门，8 个身份经人工豁免纳入，因此当前需要人工决定是否接受其为探索性 K5 对照；这不构成 real-MPB 或物理验证。详见[EXP-FWD-009 结果](docs/results/forward/EXP-FWD-009-K5权重热启动与六域验证.md)。
- EXP-FWD-010/011 已完成 joint 各 4k 的 seed0 200 轮训练、续训至 300 轮与两组六域增量评估；300 轮时 Graded 与等权收敛曲线未过门，六域只在 K5 Binary 击败最佳单路线。当前需人工决定是否接受其为负对照；若要 promotion 或数据量因果结论，仍缺多 seed、同训练与优化器状态合同的成对实验。
- EXP-FWD-012 已否定当前 $X/M$ 一阶 Bragg 工作频率先验对最大 gap-center 的准入价值；不得扩大窗口或读取 test 追调。该负结果不否定 EXP-FWD-014 直接监督真实 gap 决策的独立假设。
- EXP-FWD-014 已完成两个 N40 seed0 短预算任务并触发负 gate：k-profile 对应 k 点方向有效，但当前辅助权重和 complete-gap 状态约束不合格；不得自动调参重跑或进入三 seed/test。
- EXP-FWD-015 已确认低命中主要来自模型候选排序与 complete-gap 召回，而非近简并评价口径；EXP-FWD-016 只登记了一个 ranked-v2 N40 短筛任务，等待单独远端训练授权。
- N50R5 登录仍被平台 `loginCluster` 权限拒绝；N40 已部署当前 inverse Stage-A runner/orchestrator（`b93cc2d`），但当前正式训练只能以既有三卡分配证据为下限，9/18 卡仍需真实 admission 作业证明。
- Full-cell 50k forward 计算和自动审计已完成，但 AI 整理的 Draft 报告仍需人工接受；inverse 仍未在全胞、多形态、全 BZ 数据上完成正式 45-run Stage-A 或 soft-surrogate Stage B。
- EXP-FWD-019的42项三种子确认已完成，但23项未满足末窗收敛门，FNO/Joint缺parameter-matched control，六域冻结评估尚未执行；当前需人工决定哪些候选可进入下一门，不能直接宣布完整组件因果归因。
- Full-Wh 升级 native/cross-lane/resolution 主评估已完成，但 `k_conditioned` 头未被选择，threshold proxy 尚无作业或结果；这些 surrogate 结果尚未进入 full-cell inverse Stage B 或 real-MPB replay。
- inverse 缺 seeds 1/2 和满足 candidate-health/Top-5 coverage 的中立候选池。
- adapted PINO 尚未通过修正指标、residual–frequency、dense-k 与跨分辨率门。
- OOD、cross-dataset、cross-surrogate 统计和失败分析仍不完整。

## 下一步优先任务

1. **P0：人工审阅 50k 全胞 Forward 与升级冻结结果**；确认 held-out、final-fit、native、cross-lane 和 resolution 角色，接受或要求修复 [baseline 报告](docs/results/forward/full_cell_dual_lane_50000_forward_training_20260808.md)、[升级报告](docs/results/forward/full_cell_forward_model_upgrade_50k_evaluation_20260810.md)与[模型综合报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md)。
2. **P0：冻结可微逆向 MVP 的输入与预算**；按 [EXP-INV-012](experiments/research/EXP-INV-012-梯度链条开发验证.md)绑定来源 8、固定融合 3×3 权重、可微 Fourier 标准化和软硬表示，先完成纯代理梯度链、候选锁定及末尾物理方向开发验证。旧 Stage-A 训练不作为当前路线的前置任务。
3. **P0：保持 M9-only 物理计算边界**；threshold MPB/Meep preflight、pilot 与 replay 均只能在 M9 执行且需要单独授权，本轮未提交。
4. **P1：重做 Field/PINO 校准**；任一 frequency、dense-k、跨分辨率或 residual–error 门失败即保持 blocked。
5. **P1：汇总 OOD / cross-dataset / cross-surrogate**；统一报告统计量、运行时间、失败案例和 surrogate–replay gap。
6. **生成式参考：full-cell Stage-B 的 ranker 与数据版本**仍按旧协议单独冻结；该历史任务不自动启动，也不约束新可微路线的执行次序。
7. **P1：把 gap-selection 纳入 forward→inverse gate**；先对完整 test split 做低成本 band/k/gap 一致率审计，再决定是否按失配类型扩大 real-MPB/Meep replay。

## 详细内容去哪里查看

- 研究范围、Gate、失败判据：[研究计划](docs/planning/fno_generative_network_adjustment_plan_20260617.md)
- 逐项任务：[Research TODO](docs/planning/research_todo_20260617.md)（早期勾选可能滞后）
- 模型实现与证据状态：[模型清单](docs/reports/phc2d_model_inventory_v1.yaml)
- Full-Wh 升级代码合同：[实现规格](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_implementation_plan.md)
- Full-Wh 升级实验合同：[10k/50k 训练验证方案](docs/planning/full_cell_forward_inverse_training/full_cell_forward_model_upgrade_10k_50k_training_validation_plan.md)
- Full-Wh baseline、升级与 inverse 架构：[全胞双路线模型报告](docs/reports/全胞双路线正向与逆向模型报告_20260810.md)
- 方法、结果与边界：[完整项目报告](docs/reports/phc2d_project_full_introduction_20260724.md)
- 物理、数据与损失公式：[公式手册](docs/reports/phc2d_physics_data_formula_guide_20260803.md)
- 最新实验：[结果索引](docs/results/README.md)
- 渐变 10k 验收：[M9 验收合同](experiments/research/full_cell_graded_paracloud_m9.md)
- GPU 环境与三卡准备：[N40/N50/M9 全链路预检](docs/results/data-provenance/paracloud_gpu_preflight_20260806.md)
