"""Frozen spectral-evidence preparation for EXP-PHY-004.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.plot.full_cell_gallery import sampled_gxm_path, select_largest_gap_edge
from phc2d.plot.full_cell_surrogate_figures import render_band_plates, render_spectral_summary
from phc2d.plot.full_cell_surrogate_support import (
    SOURCES,
    decode_text,
    hash_file,
    paired_bootstrap_mean_difference,
    read_csv_rows,
    relative_path,
    repo_path,
    sample_mre_pct,
    select_quantile_positions,
    write_csv_rows,
    write_history,
)
from phc2d.runs import ensure_figures_dir, write_metrics, write_results


def resolve_evaluation_rows(config: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    """Resolve the frozen four-cell ConvNeXt/joint evidence matrix."""

    evidence = dict(config["evidence"])
    campaign = repo_path(evidence["campaign_dir"])
    rows = read_csv_rows(campaign / "evaluation_cells.csv")
    resolved: dict[tuple[str, str], dict[str, str]] = {}
    for lane, dataset in dict(config["datasets"]).items():
        target_domain = str(dataset["target_domain"])
        for model_key, model in dict(evidence["models"]).items():
            expected_lane = lane if model.get("source_lane") == "match_target" else str(model["source_lane"])
            matches = [
                row
                for row in rows
                if row.get("source_family") == str(evidence["source_family"])
                and row.get("source_seed") == str(evidence["source_seed"])
                and row.get("source_model") == str(model["source_model"])
                and row.get("source_lane") == expected_lane
                and row.get("target_domain") == target_domain
                and row.get("status") == "completed"
            ]
            if len(matches) != 1:
                raise RuntimeError(f"expected one completed {lane}/{model_key} evaluation cell, found {len(matches)}")
            row = matches[0]
            archive = campaign / "shards" / f"shard_{int(row['shard_id']):02d}" / f"{row['evaluation_id']}.npz"
            if not archive.is_file():
                raise FileNotFoundError(f"evaluation archive is missing: {archive}")
            resolved[(lane, str(model_key))] = {**row, "archive_path": str(archive)}
    if set(resolved) != {("B", "convnext"), ("B", "joint"), ("G", "convnext"), ("G", "joint")}:
        raise RuntimeError("evaluation evidence does not resolve to the frozen four-cell matrix")
    return resolved


def load_spectral_evidence(
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and load the two HDF5 targets and four prediction archives."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("h5py is required for EXP-PHY-004 spectral evidence") from exc

    evaluations = resolve_evaluation_rows(config)
    sampling = dict(config["sampling"])
    quantiles = tuple(float(value) for value in sampling["quantiles"])
    expected_test_count = int(sampling.get("expected_test_count", 1000))
    lane_payloads: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for lane, dataset_raw in dict(config["datasets"]).items():
        dataset = dict(dataset_raw)
        path = repo_path(dataset["path"])
        digest = hash_file(path)
        if digest != str(dataset["expected_dataset_sha256"]):
            raise RuntimeError(f"{lane} dataset hash mismatch: {digest}")
        archives: dict[str, dict[str, np.ndarray]] = {}
        for model in ("convnext", "joint"):
            row = evaluations[(lane, model)]
            archive_path = Path(row["archive_path"])
            with np.load(archive_path, allow_pickle=False) as loaded:
                if set(loaded.files) != {"prediction", "sample_mre", "target"}:
                    raise ValueError(f"unexpected evidence arrays in {archive_path}: {loaded.files}")
                archives[model] = {name: np.asarray(loaded[name]) for name in loaded.files}
            source_rows.append(
                {
                    "lane": lane,
                    "model_key": model,
                    "evaluation_id": row["evaluation_id"],
                    "source_model": row["source_model"],
                    "source_lane": row["source_lane"],
                    "target_domain": row["target_domain"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "archive_path": relative_path(archive_path),
                    "archive_sha256": hash_file(archive_path),
                }
            )
        expected_shape = (expected_test_count, 81, 10)
        if archives["convnext"]["target"].shape != expected_shape:
            raise ValueError(f"{lane} evidence must have shape {expected_shape}")
        if not np.array_equal(archives["convnext"]["target"], archives["joint"]["target"]):
            raise RuntimeError(f"{lane} ConvNeXt and joint targets differ")
        target = archives["convnext"]["target"].astype(np.float32, copy=False)
        for model in ("convnext", "joint"):
            recomputed = sample_mre_pct(archives[model]["prediction"], target)
            if not np.allclose(recomputed, archives[model]["sample_mre"], rtol=1.0e-5, atol=1.0e-5):
                raise RuntimeError(f"{lane}/{model} sample-MRE evidence does not recompute")

        with h5py.File(path, "r") as handle:
            if decode_text(handle.attrs.get("schema_version")) != "phc2d.full_cell_dataset.v2":
                raise ValueError(f"{lane} dataset schema mismatch")
            idx_test = np.asarray(handle["idx_test"], dtype=np.int64)
            if idx_test.shape != (expected_test_count,) or np.any(np.diff(idx_test) <= 0):
                raise ValueError(f"{lane} test split must be a sorted {expected_test_count}-index vector")
            stored_target = np.asarray(handle["Wh"][idx_test], dtype=np.float32)
            if not np.array_equal(stored_target, target):
                raise RuntimeError(f"{lane} archived target does not exactly match the frozen HDF5 test split")
            k_points = np.asarray(handle["k_points"], dtype=np.float64)
            k_grid_index = np.asarray(handle["k_grid_index"], dtype=np.int64)
            path_indices, path_ticks, path_tick_labels = sampled_gxm_path(k_grid_index)
            sample_ids = [decode_text(handle["sample_id"][int(index)]) for index in idx_test]
            geometry_hashes = [decode_text(handle["geometry_hash"][int(index)]) for index in idx_test]
            structure_json = [decode_text(handle["structure_json"][int(index)]) for index in idx_test]
            conv_mre = archives["convnext"]["sample_mre"].astype(np.float64)
            joint_mre = archives["joint"]["sample_mre"].astype(np.float64)
            difficulty = 0.5 * (conv_mre + joint_mre)
            epsilon_by_position: dict[int, np.ndarray] = {}
            for quantile, position, target_quantile in select_quantile_positions(difficulty, quantiles):
                dataset_index = int(idx_test[position])
                material = np.asarray(handle[str(dataset["grid_field"])][dataset_index], dtype=np.float32)
                epsilon_by_position[position] = 1.0 + 7.9 * np.clip(material, 0.0, 1.0)
                selected_rows.append(
                    {
                        "lane": lane,
                        "target_domain": str(dataset["target_domain"]),
                        "display_name": str(dataset["display_name"]),
                        "quantile": quantile,
                        "quantile_label": f"p{int(round(100.0 * quantile))}",
                        "quantile_target_mre_pct": target_quantile,
                        "sample_label": f"{lane}-p{int(round(100.0 * quantile)):02d}",
                        "test_position": position,
                        "dataset_index": dataset_index,
                        "sample_id": sample_ids[position],
                        "geometry_hash": geometry_hashes[position],
                        "convnext_mre_pct": float(conv_mre[position]),
                        "joint_mre_pct": float(joint_mre[position]),
                        "difficulty_mre_pct": float(difficulty[position]),
                        "structure_json": structure_json[position],
                    }
                )

        conv_prediction = archives["convnext"]["prediction"].astype(np.float32, copy=False)
        joint_prediction = archives["joint"]["prediction"].astype(np.float32, copy=False)
        lane_payloads[lane] = {
            "display_name": str(dataset["display_name"]),
            "dataset_path": path,
            "dataset_sha256": digest,
            "idx_test": idx_test,
            "k_points": k_points,
            "k_grid_index": k_grid_index,
            "path_indices": path_indices,
            "path_ticks": path_ticks,
            "path_tick_labels": path_tick_labels,
            "target": target,
            "convnext": conv_prediction,
            "joint": joint_prediction,
            "convnext_sample_mre_pct": conv_mre,
            "joint_sample_mre_pct": joint_mre,
            "convnext_per_band_mre_pct": np.mean(
                100.0 * np.abs(conv_prediction - target) / np.maximum(np.abs(target), 1.0e-8), axis=(0, 1)
            ),
            "joint_per_band_mre_pct": np.mean(
                100.0 * np.abs(joint_prediction - target) / np.maximum(np.abs(target), 1.0e-8), axis=(0, 1)
            ),
            "epsilon_by_test_position": epsilon_by_position,
        }
    selected_rows.sort(key=lambda row: (row["lane"], float(row["quantile"])))
    return lane_payloads, selected_rows, source_rows


def band_metric_rows(
    lane_payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-band and per-k full-test metric rows."""

    per_band: list[dict[str, Any]] = []
    per_k: list[dict[str, Any]] = []
    for lane, payload in lane_payloads.items():
        target = np.asarray(payload["target"], dtype=np.float64)
        for model in ("convnext", "joint"):
            relative = 100.0 * np.abs(np.asarray(payload[model]) - target) / np.maximum(np.abs(target), 1.0e-8)
            for band in range(relative.shape[2]):
                per_band.append(
                    {"lane": lane, "model": model, "band_one_based": band + 1, "mre_pct": float(np.mean(relative[:, :, band]))}
                )
            for k_index in range(relative.shape[1]):
                per_k.append(
                    {
                        "lane": lane,
                        "model": model,
                        "k_index": k_index,
                        "k_x": float(payload["k_points"][k_index, 0]),
                        "k_y": float(payload["k_points"][k_index, 1]),
                        "bands_2_10_mre_pct": float(np.mean(relative[:, k_index, 1:])),
                    }
                )
    return per_band, per_k


def band_edge_rows(
    lane_payloads: Mapping[str, Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    gap_range: tuple[int, int],
) -> list[dict[str, Any]]:
    """Record real and surrogate maximum-gap upper-edge choices."""

    rows: list[dict[str, Any]] = []
    for selected in selected_rows:
        lane, position = str(selected["lane"]), int(selected["test_position"])
        payload = lane_payloads[lane]
        truth_edge = select_largest_gap_edge(
            payload["target"][position], lower_band_min_one_based=gap_range[0], lower_band_max_one_based=gap_range[1]
        )
        for source in SOURCES:
            values = payload["target" if source == "real_mpb" else source][position]
            edge = select_largest_gap_edge(
                values, lower_band_min_one_based=gap_range[0], lower_band_max_one_based=gap_range[1]
            )
            rows.append(
                {
                    "lane": lane,
                    "sample_label": selected["sample_label"],
                    "sample_id": selected["sample_id"],
                    "dataset_index": selected["dataset_index"],
                    "source": source,
                    **asdict(edge),
                    "frequency_relative_error_vs_real": abs(edge.frequency - truth_edge.frequency)
                    / max(abs(truth_edge.frequency), 1.0e-12),
                    "gap_absolute_error_vs_real": abs(edge.gap - truth_edge.gap),
                    "band_selection_matches_real": edge.upper_band_one_based == truth_edge.upper_band_one_based,
                    "k_selection_matches_real": edge.k_index == truth_edge.k_index,
                }
            )
    return rows


def sample_metric_rows(lane_payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build one held-out metric row per lane and test position."""

    rows: list[dict[str, Any]] = []
    for lane, payload in lane_payloads.items():
        for position, dataset_index in enumerate(payload["idx_test"]):
            conv = float(payload["convnext_sample_mre_pct"][position])
            joint = float(payload["joint_sample_mre_pct"][position])
            rows.append(
                {
                    "lane": lane,
                    "test_position": position,
                    "dataset_index": int(dataset_index),
                    "convnext_bands_2_10_mre_pct": conv,
                    "joint_bands_2_10_mre_pct": joint,
                    "mean_model_difficulty_mre_pct": 0.5 * (conv + joint),
                }
            )
    return rows


def prepare_spectral_evidence(
    config: Mapping[str, Any], out: Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate, summarize, persist, and render all cheap spectral evidence."""

    started = time.perf_counter()
    figures_dir = ensure_figures_dir(out)
    lane_payloads, selected_rows, source_rows = load_spectral_evidence(config)
    gap_range = tuple(int(value) for value in config["bands"]["gap_lower_bands_one_based"])
    band_edges = band_edge_rows(lane_payloads, selected_rows, gap_range)
    per_band, per_k = band_metric_rows(lane_payloads)
    sample_rows = sample_metric_rows(lane_payloads)
    replicates = int(config["statistics"]["paired_bootstrap_replicates"])
    bootstrap_rows = []
    for index, lane in enumerate(("B", "G")):
        payload = lane_payloads[lane]
        bootstrap_rows.append(
            {
                "lane": lane,
                "comparison": "joint_minus_convnext_bands_2_10_sample_mre",
                "sample_count": len(payload["joint_sample_mre_pct"]),
                "replicates": replicates,
                **paired_bootstrap_mean_difference(
                    payload["joint_sample_mre_pct"],
                    payload["convnext_sample_mre_pct"],
                    replicates=replicates,
                    seed=int(config["statistics"]["bootstrap_seed"]) + index,
                ),
            }
        )
    for name, rows in (
        ("source_evidence.csv", source_rows),
        ("selected_samples.csv", selected_rows),
        ("sample_metrics.csv", sample_rows),
        ("per_band_metrics.csv", per_band),
        ("per_k_metrics.csv", per_k),
        ("paired_bootstrap.csv", bootstrap_rows),
        ("band_edge_frequency_manifest.csv", band_edges),
    ):
        write_csv_rows(out / name, rows)
    arrays: dict[str, np.ndarray] = {}
    for selected in selected_rows:
        lane, position, label = str(selected["lane"]), int(selected["test_position"]), str(selected["sample_label"])
        payload = lane_payloads[lane]
        for source in ("target", "convnext", "joint"):
            arrays[f"{label}_{source}"] = np.asarray(payload[source][position], dtype=np.float32)
        arrays[f"{label}_epsilon"] = np.asarray(payload["epsilon_by_test_position"][position], dtype=np.float32)
    np.savez_compressed(out / "selected_spectral_source_data.npz", **arrays)

    output = dict(config["output"])
    formats, dpi = tuple(str(value) for value in output["formats"]), int(output["dpi"])
    figure_paths = render_spectral_summary(lane_payloads, selected_rows, figures_dir=figures_dir, formats=formats, dpi=dpi)
    figure_paths += render_band_plates(lane_payloads, selected_rows, figures_dir=figures_dir, formats=formats, dpi=dpi)
    elapsed = time.perf_counter() - started
    write_metrics(
        out,
        {
            "spectral_wall_time_s": elapsed,
            "full_test_sample_count_per_lane": int(lane_payloads["B"]["target"].shape[0]),
            "selected_sample_count": len(selected_rows),
            "source_evaluation_cell_count": len(source_rows),
            "spectral_figure_file_count": len(figure_paths),
            "convnext_bands_2_10_mre_pct_B": float(np.mean(lane_payloads["B"]["convnext_sample_mre_pct"])),
            "convnext_bands_2_10_mre_pct_G": float(np.mean(lane_payloads["G"]["convnext_sample_mre_pct"])),
            "joint_bands_2_10_mre_pct_B": float(np.mean(lane_payloads["B"]["joint_sample_mre_pct"])),
            "joint_bands_2_10_mre_pct_G": float(np.mean(lane_payloads["G"]["joint_sample_mre_pct"])),
        },
    )
    write_results(
        out,
        {
            "status": "spectral_complete_physics_pending",
            "evidence_level": "surrogate_held_out_plus_stored_real_mpb_targets",
            "selected_sample_count": len(selected_rows),
            "physical_solver_execution_completed": False,
            "field_surrogate_prediction_included": False,
        },
    )
    write_history(out / "history.csv", [{"phase": "spectral", "wall_time_s": elapsed, "record_count": len(sample_rows)}])
    return lane_payloads, selected_rows


__all__ = ["prepare_spectral_evidence"]
