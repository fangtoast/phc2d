"""Summarize the V2 inverse 625-test real-MPB replay shards.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


EXPERIMENT_ID = "inverse_mpb_replay_mixture_v2_10000_test625_summary"
DENOM_FLOOR = 1.0e-6
DEFAULT_BOOTSTRAP_SEED = 20260706
REQUIRED_SHARD_ARTIFACTS = [
    "run.log",
    "run.meta.json",
    "config.resolved.yaml",
    "results.json",
    "metrics.json",
    "history.csv",
    "per_target_replay_metrics.csv",
    "per_band_replay_metrics.csv",
    "per_target_band_replay_metrics.csv",
    "replay_candidates.jsonl",
    "target_bands.npy",
    "surrogate_pred_bands.npy",
    "mpb_replay_bands.npy",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_ci(values: Sequence[float], *, repeats: int, seed: int) -> dict[str, float | int | None]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None, "bootstrap_repeats": repeats}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for index in range(repeats):
        means[index] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "bootstrap_repeats": int(repeats),
    }


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and arr[order[end]] == arr[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x_values: Sequence[float], y_values: Sequence[float], *, spearman: bool = False) -> dict[str, float | int | None]:
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return {"n": len(pairs), "r": None}
    x = np.asarray([item[0] for item in pairs], dtype=float)
    y = np.asarray([item[1] for item in pairs], dtype=float)
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"n": len(pairs), "r": None}
    return {"n": len(pairs), "r": float(np.corrcoef(x, y)[0, 1])}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return payload if isinstance(payload, dict) else {}


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def inspect_shard(shard: dict[str, Any], root: Path) -> dict[str, Any]:
    run_dir = resolve_path(root, shard["run_dir"])
    missing = [name for name in REQUIRED_SHARD_ARTIFACTS if not (run_dir / name).exists()]
    metrics = read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    return {
        **shard,
        "run_dir_abs": str(run_dir),
        "exists": run_dir.exists(),
        "missing_artifacts": missing,
        "complete": run_dir.exists() and not missing,
        "metrics": metrics,
    }


def add_shard_columns(rows: list[dict[str, Any]], shard: dict[str, Any]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.setdefault("shard_id", shard.get("shard_id"))
        item.setdefault("shard_name", shard.get("shard_name"))
        enriched.append(item)
    return enriched


def concatenate_jsonl(paths: Sequence[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line)
                        if not line.endswith("\n"):
                            out.write("\n")
                        count += 1
    return count


def aggregate_per_band(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_band: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        band = row.get("band") or row.get("band_one_based")
        try:
            by_band[int(float(str(band)))] .append(row)
        except (TypeError, ValueError):
            continue
    result: list[dict[str, Any]] = []
    for band in sorted(by_band):
        band_rows = by_band[band]
        item: dict[str, Any] = {"band": band, "n": len(band_rows)}
        for key in [
            "mpb_replay_mre_pct",
            "replay_mre_pct",
            "surrogate_mre_pct",
            "surrogate_replay_gap_pct",
            "mre_pct",
            "mae",
            "rmse",
        ]:
            values = numeric_values(band_rows, key)
            if values:
                item[f"{key}_mean"] = summarize(values)["mean"]
                item[f"{key}_median"] = summarize(values)["median"]
        result.append(item)
    return result


def summarize_metric_section(records: Sequence[dict[str, Any]], section: str, keys: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        values = numeric_values(records, key)
        item = {"section": section, "metric": key, **summarize(values)}
        rows.append(item)
    return rows


def load_candidate_preflight(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    candidates_value = manifest.get("candidates_path")
    if not candidates_value:
        return {"available": False, "reason": "manifest has no candidates_path"}
    candidates_path = resolve_path(root, candidates_value)
    if not candidates_path.exists():
        return {"available": False, "reason": f"candidates_path missing: {rel(candidates_path, root)}"}
    sample_metrics_path = candidates_path.parent / "metrics.json"
    sample_metrics = read_json(sample_metrics_path) if sample_metrics_path.exists() else {}
    candidate_payload = read_json(candidates_path)
    targets = candidate_payload.get("targets") if isinstance(candidate_payload, dict) else []
    if not isinstance(targets, list):
        targets = []
    metric_keys = [
        "n_candidates",
        "validity_rate",
        "unique_rate",
        "diversity_hamming",
        "nearest_neighbor_hamming_min",
        "nearest_neighbor_hamming_p10",
        "nearest_neighbor_hamming_mean",
        "duplicate_cluster_count",
        "largest_duplicate_cluster_fraction",
        "fill_fraction",
    ]
    filter_keys = [
        "filter_pass_ratio",
        "candidate_count_raw",
        "candidate_count_filtered",
        "candidate_separation_enabled",
        "candidate_separation_deduplicate_binary",
        "candidate_separation_min_hamming",
    ]
    raw_records = [record.get("raw_metrics", {}) for record in targets if isinstance(record, dict)]
    filtered_records = [record.get("filtered_metrics", {}) for record in targets if isinstance(record, dict)]
    surrogate_records = [record.get("surrogate_metrics", {}) for record in targets if isinstance(record, dict)]
    filter_records = [record.get("filter_metrics", {}) for record in targets if isinstance(record, dict)]
    table_rows = (
        summarize_metric_section(raw_records, "raw_candidates", metric_keys)
        + summarize_metric_section(filtered_records, "filtered_candidates", metric_keys)
        + summarize_metric_section(surrogate_records, "surrogate_ranked_candidates", metric_keys)
        + summarize_metric_section(filter_records, "candidate_filter", filter_keys)
    )
    global_keys = [
        "targets",
        "validity_rate",
        "unique_rate",
        "diversity_hamming",
        "nearest_neighbor_hamming_min",
        "nearest_neighbor_hamming_p10",
        "nearest_neighbor_hamming_mean",
        "duplicate_cluster_count",
        "largest_duplicate_cluster_fraction",
        "morphology_score",
        "filter_pass_ratio",
        "candidate_sampling_time_s",
        "surrogate_ranking_time_s",
        "inference_time_per_sample_ms",
    ]
    global_metrics = {key: sample_metrics.get(key) for key in global_keys if key in sample_metrics}
    primary = {
        row["metric"]: row
        for row in table_rows
        if row["section"] == "surrogate_ranked_candidates"
        and row["metric"]
        in {
            "validity_rate",
            "unique_rate",
            "nearest_neighbor_hamming_min",
            "nearest_neighbor_hamming_p10",
            "nearest_neighbor_hamming_mean",
            "fill_fraction",
        }
    }
    return {
        "available": True,
        "candidates_path": rel(candidates_path, root),
        "sample_metrics_path": rel(sample_metrics_path, root) if sample_metrics_path.exists() else None,
        "target_count": len(targets),
        "global_metrics": global_metrics,
        "primary_section": "surrogate_ranked_candidates",
        "primary_metrics": primary,
        "table_rows": table_rows,
    }


def best_worst(rows: Sequence[dict[str, Any]], key: str, n: int = 20) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = [(to_float(row.get(key)), row) for row in rows]
    valid = [(score, row) for score, row in scored if score is not None]
    ordered = sorted(valid, key=lambda item: item[0])
    return [dict(row) for _score, row in ordered[:n]], [dict(row) for _score, row in ordered[-n:][::-1]]


def copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def plot_histogram(values: Sequence[float], title: str, xlabel: str, path_base: Path) -> None:
    if not values:
        return
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=160)
    ax.hist(values, bins=30, color="#2364AA", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Target count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"))
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def plot_per_band(rows: Sequence[dict[str, Any]], path_base: Path) -> None:
    value_key = "mpb_replay_mre_pct_mean"
    bands = [int(row["band"]) for row in rows if "band" in row and row.get(value_key) is not None]
    values = [float(row[value_key]) for row in rows if "band" in row and row.get(value_key) is not None]
    if not bands:
        return
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=160)
    ax.bar(bands, values, color="#3A7D44")
    ax.set_title("Per-band real-MPB replay MRE")
    ax.set_xlabel("Band")
    ax.set_ylabel("MRE (%)")
    ax.set_xticks(bands)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"))
    fig.savefig(path_base.with_suffix(".svg"))
    plt.close(fig)


def artifact_info(path: Path, root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": rel(path, root), "exists": path.exists()}
    if not path.exists():
        return info
    info["size_bytes"] = int(path.stat().st_size)
    if path.suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            info["columns"] = next(reader, [])
            info["row_count"] = sum(1 for _ in reader)
    elif path.suffix == ".json":
        payload = read_json(path)
        info["top_level_keys"] = list(payload)[:50] if isinstance(payload, dict) else []
    elif path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        info["shape"] = [int(item) for item in arr.shape]
        info["dtype"] = str(arr.dtype)
    return info


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    output_dir: Path,
    root: Path,
) -> None:
    status = summary["status"]
    replay = summary["metrics"]["best_of_k_replay_mre_bands_2_10_pct"]
    ci = summary["metrics"]["bootstrap_ci95"]["best_of_k_replay_mre_bands_2_10_pct"]
    gap = summary["metrics"]["surrogate_replay_gap_bands_2_10_pct"]
    preflight = summary.get("candidate_preflight", {})
    preflight_primary = preflight.get("primary_metrics", {}) if isinstance(preflight, dict) else {}
    runtime = summary.get("runtime", {})

    def preflight_mean(metric: str) -> Any:
        row = preflight_primary.get(metric) if isinstance(preflight_primary, dict) else None
        return row.get("mean") if isinstance(row, dict) else None

    lines = [
        "# Inverse MPB Replay Mixture V2 10000 Test625 Report",
        "",
        "## Evidence Boundary",
        "",
        f"- Status: `{status}`.",
        "- No training was run in this replay summary path; candidates come from the fixed inverse generator checkpoint and frozen query/full-Wh surrogates.",
        "- Replay backend is required to be `real_mpb`; Windows-local surrogate or smoke results are not counted as formal validation.",
        "- Target split is the pure dataset `idx_test` split: 625 targets, expected index range 9375..9999.",
        "- Headline metrics use bands 2-10 only; band1/full-band views are retained only as risk diagnostics.",
        "",
        "## Candidate Preflight",
        "",
        f"- Target count: {summary['target_count_processed']} / {summary['target_count_expected']}.",
        f"- Replay top-M: {summary['replay_top_m']}.",
        f"- Shard count complete: {summary['complete_shard_count']} / {summary['shard_count']}.",
        f"- Primary candidate pool: `{preflight.get('primary_section', 'unavailable')}`.",
        f"- Validity / unique rate: {preflight_mean('validity_rate')} / {preflight_mean('unique_rate')}.",
        f"- Nearest-neighbor Hamming min / p10 / mean: {preflight_mean('nearest_neighbor_hamming_min')} / {preflight_mean('nearest_neighbor_hamming_p10')} / {preflight_mean('nearest_neighbor_hamming_mean')}.",
        f"- Fill fraction mean: {preflight_mean('fill_fraction')}.",
        f"- Candidate preflight table: `{rel(output_dir / 'summary_tables' / 'candidate_preflight_metrics.csv', root)}`.",
        "",
        "## Real-MPB Replay",
        "",
        f"- Mean bands 2-10 replay MRE: {replay.get('mean')}.",
        f"- Median / p10 / p90: {replay.get('median')} / {replay.get('p10')} / {replay.get('p90')}.",
        f"- Min / max: {replay.get('min')} / {replay.get('max')}.",
        f"- Bootstrap 95% CI for mean: [{ci.get('ci95_low')}, {ci.get('ci95_high')}].",
        "",
        "## Runtime",
        "",
        f"- Sum MPB replay time: {runtime.get('mpb_replay_time_s_sum')} s.",
        f"- Sum shard wall time: {runtime.get('wall_time_total_s_sum')} s.",
        f"- MPB replay time per target / candidate: {runtime.get('mpb_replay_time_s_per_target')} s / {runtime.get('mpb_replay_time_s_per_candidate')} s.",
        "",
        "## Gap Analysis",
        "",
        f"- Mean surrogate-replay gap bands 2-10: {gap.get('mean')}.",
        f"- Pearson r: {summary['metrics']['gap_correlations']['pearson'].get('r')}.",
        f"- Spearman r: {summary['metrics']['gap_correlations']['spearman'].get('r')}.",
        "",
        "## Worst/Best Cases",
        "",
        f"- Worst 20 table: `{rel(output_dir / 'summary_tables' / 'worst_20_targets.csv', root)}`.",
        f"- Best 20 table: `{rel(output_dir / 'summary_tables' / 'best_20_targets.csv', root)}`.",
        "",
        "## Baseline Comparison",
        "",
        "- Baseline is the current 40-target formal replay, so it is a scale/reference comparison only, not a same-N statistical replacement.",
    ]
    if baseline:
        lines.append(f"- 40-target baseline mean replay MRE bands 2-10: {baseline.get('mean_replay_mre_bands_2_10_pct')}.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary run: `{rel(output_dir, root)}`.",
            f"- Artifact index: `{rel(output_dir / 'artifact_index.json', root)}`.",
            f"- Metrics summary: `{rel(output_dir / 'metrics_summary.json', root)}`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_baseline(root: Path, baseline_metrics: Path | None) -> dict[str, Any] | None:
    if baseline_metrics is None:
        default = root / "runs/20260706/inverse_mpb_replay_mixture_v2_10000_40_202607060853/metrics.json"
        baseline_metrics = default if default.exists() else None
    if baseline_metrics is None or not baseline_metrics.exists():
        return None
    payload = read_json(baseline_metrics)
    return {
        "path": rel(baseline_metrics, root),
        "target_count": payload.get("targets"),
        "mean_replay_mre_bands_2_10_pct": payload.get("best_of_k_replay_mre_bands_2_10_pct"),
        "replay_backend": payload.get("replay_backend"),
        "validation_boundary": payload.get("validation_boundary"),
    }


def summarize_runtime(shard_infos: Sequence[dict[str, Any]], *, target_count: int, candidate_count: int) -> dict[str, Any]:
    metrics = [item.get("metrics", {}) for item in shard_infos]
    mpb_times = [to_float(item.get("mpb_replay_time_s")) for item in metrics]
    wall_times = [to_float(item.get("wall_time_total_s")) for item in metrics]
    mpb_values = [float(item) for item in mpb_times if item is not None]
    wall_values = [float(item) for item in wall_times if item is not None]
    total_mpb = float(sum(mpb_values))
    total_wall = float(sum(wall_values))
    return {
        "shard_count": int(len(shard_infos)),
        "mpb_replay_time_s_sum": total_mpb,
        "wall_time_total_s_sum": total_wall,
        "mpb_replay_time_s_per_target": total_mpb / target_count if target_count else None,
        "mpb_replay_time_s_per_candidate": total_mpb / candidate_count if candidate_count else None,
        "wall_time_total_s_per_target": total_wall / target_count if target_count else None,
        "mpb_replay_time_s_by_shard": summarize(mpb_values),
        "wall_time_total_s_by_shard": summarize(wall_values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True, type=Path, help="Directory containing shard_manifest.json.")
    parser.add_argument("--output-dir", type=Path, help="Summary run output directory.")
    parser.add_argument("--report-path", type=Path, help="Markdown report path.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write an incomplete summary if shard artifacts are missing.")
    parser.add_argument("--expected-target-count", default=625, type=int)
    parser.add_argument("--expected-nk", default=31, type=int)
    parser.add_argument("--expected-nb", default=10, type=int)
    parser.add_argument("--bootstrap-repeats", default=1000, type=int)
    parser.add_argument("--bootstrap-seed", default=DEFAULT_BOOTSTRAP_SEED, type=int)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--timestamp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    shard_root = resolve_path(root, args.shard_root)
    output_dir = resolve_path(root, args.output_dir) if args.output_dir else root / "runs" / timestamp[:8] / f"{EXPERIMENT_ID}_{timestamp}"
    report_path = resolve_path(root, args.report_path) if args.report_path else root / "docs" / "results" / f"inverse_mpb_replay_mixture_v2_10000_test625_{timestamp[:8]}.md"

    manifest = read_json(shard_root / "shard_manifest.json")
    shard_infos = [inspect_shard(shard, root) for shard in manifest.get("shards", [])]
    incomplete = [item for item in shard_infos if not item["complete"]]
    if incomplete and not args.allow_incomplete:
        first = incomplete[0]
        raise FileNotFoundError(f"Incomplete shard {first.get('shard_name')} missing {first.get('missing_artifacts')}")

    complete_shards = [item for item in shard_infos if item["complete"]]
    per_target_rows: list[dict[str, Any]] = []
    per_target_band_rows: list[dict[str, Any]] = []
    jsonl_paths: list[Path] = []
    target_arrays: list[np.ndarray] = []
    surrogate_arrays: list[np.ndarray] = []
    replay_arrays: list[np.ndarray] = []

    for shard in complete_shards:
        run_dir = Path(shard["run_dir_abs"])
        per_target_rows.extend(add_shard_columns(read_csv(run_dir / "per_target_replay_metrics.csv"), shard))
        per_target_band_rows.extend(add_shard_columns(read_csv(run_dir / "per_target_band_replay_metrics.csv"), shard))
        jsonl_paths.append(run_dir / "replay_candidates.jsonl")
        target_arrays.append(np.load(run_dir / "target_bands.npy"))
        surrogate_arrays.append(np.load(run_dir / "surrogate_pred_bands.npy"))
        replay_arrays.append(np.load(run_dir / "mpb_replay_bands.npy"))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tables = output_dir / "summary_tables"
    figures = output_dir / "figures"

    per_band_rows = aggregate_per_band(per_target_band_rows)
    candidate_preflight = load_candidate_preflight(manifest, root)
    best_rows, worst_rows = best_worst(per_target_rows, "best_of_k_replay_mre_bands_2_10_pct", n=20)
    write_csv(summary_tables / "per_target_replay_metrics.csv", per_target_rows)
    write_csv(summary_tables / "per_target_band_replay_metrics.csv", per_target_band_rows)
    write_csv(summary_tables / "per_band_replay_metrics.csv", per_band_rows)
    write_csv(summary_tables / "candidate_preflight_metrics.csv", candidate_preflight.get("table_rows", []))
    write_csv(summary_tables / "best_20_targets.csv", best_rows)
    write_csv(summary_tables / "worst_20_targets.csv", worst_rows)
    replay_candidate_count = concatenate_jsonl(jsonl_paths, output_dir / "replay_candidates.jsonl") if jsonl_paths else 0

    if target_arrays:
        target_bands = np.concatenate(target_arrays, axis=0)
        surrogate_pred_bands = np.concatenate(surrogate_arrays, axis=0)
        mpb_replay_bands = np.concatenate(replay_arrays, axis=0)
        np.save(output_dir / "target_bands.npy", target_bands)
        np.save(output_dir / "surrogate_pred_bands.npy", surrogate_pred_bands)
        np.save(output_dir / "mpb_replay_bands.npy", mpb_replay_bands)
    else:
        target_bands = surrogate_pred_bands = mpb_replay_bands = np.empty((0, args.expected_nk, args.expected_nb), dtype=np.float32)

    expected_shape = (args.expected_target_count, args.expected_nk, args.expected_nb)
    shape_ok = tuple(target_bands.shape) == expected_shape and tuple(mpb_replay_bands.shape) == expected_shape
    target_count_ok = len(per_target_rows) == args.expected_target_count
    metrics_ok = all(
        shard.get("metrics", {}).get("replay_backend") == "real_mpb"
        and shard.get("metrics", {}).get("validation_boundary") == "formal_inverse_candidate_mpb_replay"
        for shard in complete_shards
    )
    status = "complete" if not incomplete and shape_ok and target_count_ok and metrics_ok else "incomplete"
    if status != "complete" and not args.allow_incomplete:
        raise ValueError(
            f"Summary acceptance failed: status={status} shape={tuple(target_bands.shape)} "
            f"expected_shape={expected_shape} target_rows={len(per_target_rows)}"
        )

    replay_key = "best_of_k_replay_mre_bands_2_10_pct"
    surrogate_key = "best_of_k_surrogate_mre_bands_2_10_pct"
    gap_key = "surrogate_replay_gap_bands_2_10_pct"
    replay_values = numeric_values(per_target_rows, replay_key)
    surrogate_values = numeric_values(per_target_rows, surrogate_key)
    gap_values = numeric_values(per_target_rows, gap_key)
    metrics = {
        replay_key: summarize(replay_values),
        surrogate_key: summarize(surrogate_values),
        gap_key: summarize(gap_values),
        "bootstrap_ci95": {
            replay_key: bootstrap_ci(replay_values, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed),
            surrogate_key: bootstrap_ci(surrogate_values, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed + 1),
            gap_key: bootstrap_ci(gap_values, repeats=args.bootstrap_repeats, seed=args.bootstrap_seed + 2),
        },
        "gap_correlations": {
            "pearson": correlation(surrogate_values, replay_values),
            "spearman": correlation(surrogate_values, replay_values, spearman=True),
        },
    }
    runtime_summary = summarize_runtime(
        complete_shards,
        target_count=len(per_target_rows),
        candidate_count=replay_candidate_count,
    )
    baseline = load_baseline(root, resolve_path(root, args.baseline_metrics) if args.baseline_metrics else None)
    summary = {
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "validation_boundary": "formal_inverse_candidate_mpb_replay" if status == "complete" else "incomplete_inverse_candidate_mpb_replay",
        "replay_backend": "real_mpb",
        "target_split": "test",
        "target_count_expected": int(args.expected_target_count),
        "target_count_processed": int(len(per_target_rows)),
        "band_array_shape": [int(item) for item in target_bands.shape],
        "expected_band_array_shape": list(expected_shape),
        "shape_ok": bool(shape_ok),
        "target_count_ok": bool(target_count_ok),
        "metrics_boundary_ok": bool(metrics_ok),
        "shard_count": int(len(shard_infos)),
        "complete_shard_count": int(len(complete_shards)),
        "missing_shards": [
            {"shard_name": item.get("shard_name"), "missing_artifacts": item.get("missing_artifacts")} for item in incomplete
        ],
        "replay_top_m": 5,
        "replay_candidate_jsonl_rows": int(replay_candidate_count),
        "metrics": metrics,
        "runtime": runtime_summary,
        "candidate_preflight": candidate_preflight,
        "baseline_comparison": baseline,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    write_json(output_dir / "shard_manifest.json", {**manifest, "summary_status": status, "shards": shard_infos})
    copy_artifact(shard_root / "shard_manifest.json", output_dir / "source_shard_manifest.json")

    plot_histogram(replay_values, "Real-MPB replay MRE, bands 2-10", "MRE (%)", figures / "replay_mre_bands_2_10_hist")
    plot_histogram(gap_values, "Surrogate-replay gap, bands 2-10", "Gap (%)", figures / "surrogate_replay_gap_hist")
    plot_per_band(per_band_rows, figures / "per_band_replay_mre")

    figure_artifacts = sorted(figures.glob("*.png")) + sorted(figures.glob("*.svg"))
    artifacts = [
        output_dir / "metrics_summary.json",
        output_dir / "shard_manifest.json",
        output_dir / "source_shard_manifest.json",
        summary_tables / "per_target_replay_metrics.csv",
        summary_tables / "per_target_band_replay_metrics.csv",
        summary_tables / "per_band_replay_metrics.csv",
        summary_tables / "candidate_preflight_metrics.csv",
        summary_tables / "best_20_targets.csv",
        summary_tables / "worst_20_targets.csv",
        output_dir / "replay_candidates.jsonl",
        output_dir / "target_bands.npy",
        output_dir / "surrogate_pred_bands.npy",
        output_dir / "mpb_replay_bands.npy",
    ] + figure_artifacts
    write_json(output_dir / "artifact_index.json", {"artifacts": [artifact_info(path, root) for path in artifacts]})
    write_report(report_path, summary=summary, baseline=baseline, output_dir=output_dir, root=root)
    print(f"Wrote summary: {rel(output_dir, root)} status={status}")
    print(f"Wrote report: {rel(report_path, root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
