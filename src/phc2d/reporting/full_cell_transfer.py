"""Render a reviewable Draft from a finalized twelve-domain surrogate matrix.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

from phc2d.forward.full_cell_multidomain_contracts import read_csv
from phc2d.forward.full_cell_multidomain_evaluation import load_evaluation_campaign


def render_transfer_report(campaign_dir: str | Path) -> Path:
    """Write only a run-local Draft; canonical publication requires separate review."""
    campaign, profile = load_evaluation_campaign(campaign_dir)
    combined = campaign / "evaluation/combined"
    summary = json.loads((combined / "results.json").read_text(encoding="utf-8"))
    if summary.get("status") != "finalized_twelve_domain_exploratory_evidence" or summary.get("evaluation_cell_count") != 756:
        raise RuntimeError("a complete audited 756-cell summary is required")
    matrix = read_csv(combined / "cell_matrix.csv")
    macros = read_csv(combined / "twelve_domain_macro.csv")
    resources = read_csv(combined / "training_resources.csv")
    lookup = {(r["regime"], r["model"], r["target_domain"]): r for r in matrix}
    resource_map = {(r["regime"], r["model"]): r for r in resources}
    targets = [f"k{k}_{lane}" for k in (5, 8, 10, 15, 20) for lane in ("binary", "graded")] + [f"f50_{lane}" for lane in ("binary", "graded")]
    if len(lookup) != 756 or any(target not in {r["target_domain"] for r in matrix} for target in targets):
        raise RuntimeError("report source matrix identity is incomplete")
    passed = sum(r["training_status"] == "formal_training_success" for r in resources)
    best210 = min(macros, key=lambda r: float(r["bands_2_10_mre_pct"]))
    best110 = min(macros, key=lambda r: float(r["bands_1_10_mre_pct"]))
    date = campaign.parent.name
    date = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 and date.isdigit() else "2026-09-17"
    lines = ["---", "title: EXP-FWD-023 十二域交叉测试", "summary: 固定 seed0 的 63 个模型在十二个 held-out 域上的跨数据集与跨路线测试。",
             "page_type: evidence", "status: draft", "audience: 研究用户与项目负责人", "owner: 研究生维护，导师审阅",
             f"last_reviewed: {date}", "reading_time: 12 minutes", "primary_layer: L3", f"evidence_date: {date}",
             "evidence_scope: seed0 exploratory surrogate evidence; new datasets pending WSL physical acceptance", "---", "",
             "<!--", "  Fixed-checkpoint twelve-domain Draft; numerical evidence remains in runs.", "  Author: fangtoast", "  Email: fangtoast@foxmail.com",
             "  Copyright (c) 2026 fangtoast. All rights reserved.", "-->", "", "# EXP-FWD-023：十二域交叉测试", "", "## 结果与限制", "",
             f"已审计 63 个 validation-best checkpoint、756 个 cell 和 7,560 条逐 Band 记录。十二域等权 Bands 2–10 最低组合为 `{best210['regime']}/{best210['model']}`（{float(best210['bands_2_10_mre_pct']):.6f}%）；Bands 1–10 最低组合为 `{best110['regime']}/{best110['model']}`（{float(best110['bands_1_10_mre_pct']):.6f}%）。",
             "", "不同统计口径不一致时保留排名冲突，不选择性宣布冠军。单 seed 与单 checkpoint-role 只构成探索性 surrogate 证据，不是多 seed 稳定性、物理有效性或模型 promotion。",
             "", "K8/K10/K15 六域已通过集群哈希、schema、split、配对和 source-train/validation 泄漏审计，但 **WSL real-MPB 数据验收未完成**；相关结论保持该限制。",
             "", f"训练 63/63 完成固定 200 epochs，其中 {passed} 个通过收敛门、{63-passed} 个为 `training_complete_convergence_not_met`。未重训、未按目标域改 normalizer，未用本轮 test 重新选 checkpoint。",
             "", "## 测试与比较合同", "", "K5/K8/K10/K15/K20 的两路线各使用冻结 test 1,000 条；F50 的两路线各 5,000 条。每条结构预测完整 Wh[81,10]。旧六域复用原 cache，新增六域各只读入一次 cache。",
             "", "对照基线为同一训练制度下的 FNO；不能把不同制度的差异归因为架构。目标域均等权，禁止按 F50 样本数量加权。K8/K10/K15 同 lane 按 sample_id 配对；K5/K20/F50 不声称与该 cohort 逐结构配对。",
             "", "## 训练来源→目标域", "", "以下数值均为 MRE%；每格按 **Bands 2–10 / Bands 1–10** 排列。Band 1 和十条逐 Band 指标始终保留在审计后的机器证据中，不事后剔除。", ""]
    labels = [t.replace("_binary", "-B").replace("_graded", "-G").upper() for t in targets]
    for regime in profile["regimes"]:
        source_domains = ", ".join(profile["regimes"][regime]["domains"])
        lines.extend([f"### {regime}", "", f"训练来源：`{source_domains}`。", "", "| 模型 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(targets)])
        for model in profile["models"]:
            values = [lookup[(regime, model, target)] for target in targets]
            lines.append(f"| {model} | " + " | ".join(f"{float(v['bands_2_10_mre_pct']):.4f} / {float(v['bands_1_10_mre_pct']):.4f}" for v in values) + " |")
        lines.append("")
    lines.extend(["## 十二域等权汇总与收敛", "", "| 制度 | 模型 | Band 1 MRE% | Bands 2–10 MRE% | Bands 1–10 MRE% | 收敛门 |", "|---|---|---:|---:|---:|---|"])
    for row in macros:
        status = resource_map[(row["regime"], row["model"])]["training_status"]
        lines.append(f"| {row['regime']} | {row['model']} | {float(row['band1_mre_pct']):.4f} | {float(row['bands_2_10_mre_pct']):.4f} | {float(row['bands_1_10_mre_pct']):.4f} | {'通过' if status == 'formal_training_success' else '未通过'} |")
    lines.extend(["", "## 统计解释与下一决策", "", "模型 bootstrap 在相同目标域、相同测试结构上与固定 FNO reference 比较；配对 K-shift 使用同一 checkpoint 的 K8→K10/K15 样本误差。95% 区间反映样本不确定性，不能替代跨 seed 稳定性。训练源域、未见域、跨 lane 和双重迁移分组必须结合域数量解释，不能用总宏平均掩盖目标域退化。",
                  "", "下一步由人类审阅逐域表、两种排名冲突、收敛状态和置信区间，决定是否接受探索性结果；正式数据物理接受仍需单独完成六份 WSL 验收。不得据此自动 promotion、调参或追加 seed。",
                  "", "## 原始证据", "", f"完整合并证据：`{combined}`。", "", "| 证据 | 用途 |", "|---|---|",
                  "| cell_matrix.csv / twelve_domain_macro.csv | 756-cell 数值、十二域等权汇总与目标域数据状态 |",
                  "| per_band_metrics.csv | 7,560 条逐 Band 记录 |", "| paired_model_bootstrap.csv / paired_k_shift_bootstrap.csv | 模型与配对 K-shift 区间 |",
                  "| transfer_groups.csv / ranking_conflicts.csv | 转移分组与每制度两口径排名冲突 |",
                  "| training_resources.csv / 两套 cell metrics.json | 训练时间、峰值显存、推理吞吐与单样本延迟 |",
                  "| results.json / isolation_audit.json（campaign 根） | 完整性、输入哈希、泄漏和数据验收边界 |", ""])
    output = combined / "EXP-FWD-023-十二域交叉测试.md"
    if output.exists():
        raise FileExistsError("Draft report already exists; no automatic overwrite")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
