<!--
  Pure-surrogate MVP with locked selection and terminal MPB diagnostics.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# EXP-INV-012：梯度链条开发验证

**状态：Draft，纯代理接口开发中，正式批次未运行。** 别名 MVP-INV-001。前版 63 组/150 核时及物理 FD 引导方向方案不再采用；无已有正式执行锁或 run 被覆盖。研究问题是：冻结 GL 的参数梯度能否指导结构优化，使预先选定的硬结构在末尾 MPB 中呈现真实改善？本批只作开发诊断，不证明独立泛化。

入口见[当前目录](../../docs/planning/鲁棒带隙逆向/README.md)、[纯代理方法](../../docs/planning/鲁棒带隙逆向/纯代理方法.md)与[实验复核](../../docs/planning/鲁棒带隙逆向/实验与复核.md)。所有优化、候选过滤、评分和选型使用 GL；本批 MPB 成绩在选择锁之后揭示，不用于续跑、调参或补选。

本实验交付完整研究的首个纯代理基线；后续 MPB 代理校正/微调属于 EXP-INV-014，按[推进与交付](../../docs/planning/鲁棒带隙逆向/推进与交付.md)实施，不改变本批候选的锁定规则。

## 原点子批的运行入口

[原点子批配置](../../configs/research/exp_inv_012_origin_v1.json)绑定来源 8 文件、K5 Binary 固定融合 3×3 seed 0 checkpoint 和 k 点哈希，并设定两参数范围、优化预算、物理协议及预检容差。该 checkpoint 来自 EXP-FWD-020 的 epoch-300 validation-best 探索续训；它与 010/011 门控模型分开。配置用于开发批，正式执行前须完成两端预检并固定实际资源。

N40 和 N56 各用 `jobs/paracloud_inverse_differentiable_mvp.sh` 的 `preflight` 角色输出模型身份、中心谱、参数梯度、有限差分及耗时；`python -m phc2d.inverse.differentiable_preflight --input N40_JSON --input N56_JSON --output GATE_JSON` 合并准入。正式分片使用同一部署 commit、配置、来源和 checkpoint：N40 承担 `directions`、`adam`，N56 承担 `de`、`random`。`jobs/paracloud_submit_inverse_differentiable.sh` 按角色提交并保存回执，正式分片读取已通过的预检 gate；提交前核验资源和已有回执。N40 的 `gpu_4090` 分区使用每卡默认内存，调度器拒绝自定义内存参数；按实际分配记录费用。

四个分片返回后，`python -m phc2d.inverse.differentiable_lock --directions DIRECTIONS_JSON --adam ADAM_JSON --de DE_JSON --random RANDOM_JSON --output LOCK_JSON` 校验身份、中心谱、参数梯度和各自硬评分 Top-1，锁定 12 个角色。`python -m phc2d.inverse.differentiable_handoff --config CONFIG_JSON --lock LOCK_JSON --output-directory REQUEST_DIR` 生成现有 `audit_perturb_refine` 所需的 nominal 零偏移 MPB 请求。相同 512 硬栅格只求解一次并保留角色映射；相同 64 代理输入不代表相同硬几何。

使用 `python -m phc2d.inverse.differentiable_archive --config CONFIG_JSON --lock LOCK_JSON --directions DIRECTIONS_JSON --adam ADAM_JSON --de DE_JSON --random RANDOM_JSON --output RUN_DIR` 将四个原始分片和锁按项目 run 合同归档；归档状态为 `design_locked`，物理验证状态另行更新。

求解后用 `python -m phc2d.inverse.differentiable_compare --lock LOCK_JSON --manifest MANIFEST_JSON --protocol PROTOCOL_JSON --result CASE_JSON --output COMPARISON_JSON` 比较原始同 k 频谱与加密带边；多个 case 重复传入 `--result`，缺失角色保留状态。采样和分辨率结果保留在原始 solver 输出，不与同 k 频谱误差合并。

## 输入与冻结

| 项目 | 草案合同 |
|---|---|
| 来源 | EXP-INV-010 来源池编号 8，完整源 spec/hash；不是训练行号；已观察开发来源，不因结果换来源 |
| 目标 | TE、B4–B5、$a/\lambda=0.57$，精确有符号裕量 |
| 主模型 | 固定 0.5/0.5 融合的 global_local_band_operator_v1、local_kernel=3、K5 Binary；具体 checkpoint/seed/hash 待冻结 |
| 身份差异 | 010/011 门控 GL 的 hash 与成绩不归入固定融合模型 |
| 变量 | modes[0].cos 与标准化潜场 threshold，实际模态向量从源 spec 读取，其余固定 |
| 坐标 | $z=(\alpha-\alpha_0)/[0.2,0.1]$，草案箱域 $[-0.5,0.5]^2$ |
| 几何 | 重算均值/总体标准差，Sigmoid 松弛和 >= 硬阈值；512 主栅格按合同降至 64 |
| 材料 | 核对原 spec 的空气 1、材料 8.9、材料权重及填充率 [0.5,0.9]；不符先核查 |
| 优化 | 固定温度 Adam、DE、Random，同约束与软目标；硬 GL 精确评分选 Top-1 |
| 不包含 | 在线 MPB、残差、物理 FD 引导搜索、制造三场景、骨干微调 |

先核验源/数据/模型/k/normalizer/参考输入输出；再测试 tensor Fourier、总体标准差导数、软硬映射与 block mean。代理参数 AD–FD 使用多个步长，初始可考察归一化步长 $10^{-2},3\times10^{-3},10^{-3}$；具体容差、步长、温度、平滑参数、学习率、步数及种子在物理求解前冻结。需要 MPB 校准时使用单列开发批，校准后另锁本批，不把校准成绩作为独立测试。

## 候选与选择锁

| 部分 | 规则 | 硬几何评价组上限 |
|---|---|---:|
| 中心 | $z=0,\pm0.05e_1,\pm0.05e_2$ | 5 |
| 方向 | 各中心半径 0.05、0.10 的正/负 GL 梯度及正/负预固定随机方向 | 40 |
| 短程终点 | 原点 Adam、DE、Random 各最多 20 次目标评价；按硬 GL 排名各选 Top-1 | 3 |
| 总计 | 重复、非法、同栅格保留状态，不抽点替换 | **48** |

先完成全部 GL 轨迹、合法性筛选、硬评分与 Top-1，然后锁定参数/spec、硬栅格、模型输入、完整原始预测、排名、代码/模型/配置 hash、物理清单与复核角色，再执行评价。相同分数按起点与轨迹序号；相同硬几何保留最早身份。梯度零/非有限、边界裁剪或无合法候选分别记录，不扩大预算补选。

若仅执行原点链路，须在锁前单独登记 12 组子批（原点、8 方向、3 终点），其余四中心是后续预注册扩展；不看成功结果再冒称同一个完整 48 组批。物理 FD 如后续需要，仅解释已经锁定的方向并另记费用，不能据此新增优化候选。

三种算法分别评价自己锁定的 Top-1；方向诊断不参与最终选型，不从全部 MPB 结果中挑最好结构冒称 GL 输出。来源 8 初始可能已为正裕量，须报告相对初值变化。Top-M 若另行研究，事先冻结 M 和聚合规则；MPB best-of-M 与纯代理 Top-1 分列。

## MPB 复核与成本

先以相同硬结构、材料、81 k、极化和带序计算 GL–MPB 全谱误差及 B4/B5 带边误差。再做自适应密 k 与空间分辨率审查，采样差异不混入同 k 模型误差。拟核对旧合同的 512 几何、192/224 分辨率、mesh_size=7、eigensolver tolerance $10^{-9}$、10 条带加 4 条 guard band、带边/分辨率容差 $5\times10^{-4}$；完整配置须在锁内冻结。

WSL Ubuntu/meep_env 预选原点、原点半径 0.10 的正梯度/负梯度/正随机、Adam 终点和 DE 终点六个角色，拟按 $10^{-6}$ 全谱绝对差复核；重复只报覆盖，不按成绩换角色。跨环境一致不等于 k 或空间分辨率已收敛。旧零偏移谱只有在结构和完整配置/谱身份核对后才可作评价参考，不能输入设计器。

48 为候选组上限，不是 solver 调用数。CPU 核时、GPU 卡时、WSL 复核核时、内存、并发、墙钟、失败/重试权限按同合同实测在准入前冻结；不继承前版额度。设计、在线物理（主方法为零）、最终复核成本分列，forward/backward、重启、FD、硬重预测、加密与重复任务全部记账；达到上限记 partial，不自动扩容。

[原点子批资源提案](../../configs/research/exp_inv_012_resources_v1.json)为四个 GPU 分片各一张卡、最多 30 分钟；M9 物理复核最多四个三案例分片，各 5 CPU、12 GiB、2 小时，合计上限 40 核时。M9 参数参考 EXP-INV-010 同分辨率、三扰动计算的历史账单；本批仅有名义场景，实际案例数以锁定后的去重清单为准。`python -m phc2d.inverse.differentiable_resources --manifest MANIFEST_JSON --resources RESOURCES_JSON --output RESOURCE_PLAN_JSON` 可生成准确分片范围和预留核时，不提交作业。WSL 六个角色的复核成本单列；资源提案在两端预检和求解环境核对后冻结。

## gate 与交付

| gate | 证据 |
|---|---|
| 身份与数学 | 原始输入/预测复现、冻结权重、参数 AD–FD 与步长稳定性按预冻结容差通过 |
| 选型 | 所有轨迹、无效/重复记录、Top-1 和锁可重建；MPB 未参与决策 |
| 真实方向 | 全部预选正/负/随机及优化终点的真实变化、软硬差、谱误差与数值噪声完整 |
| 数值和资源 | 全部评价覆盖、预选跨环境角色、数值门、失败及总成本完整 |

改善须超过预冻结数值噪声；混合或负结果可以结束 MVP。零梯度、同硬栅格、带序异常或未定义余弦单列，不填零、不静默排序修复。失败和未完成保留在全注册分母。正式多来源对照另由 013 检验，不能把五个局部点当五个独立来源。

run 使用 runs/{yyyymmdd}/EXP-INV-012_{YYYYMMDDHHmm}/，保存标准 run.log、run.meta.json、config.resolved.yaml、results.json、metrics.json、history.csv，附参数梯度、完整轨迹、selection lock、候选/求解清单、原始谱/栅格与资源账本。复核失败不改原候选；改方法须新开发版本与新评估批。科学结论或 gate 改变后才建 Draft 结果页。
