"""Run-directory, logging, and JSON result helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.runs.core import (
    RUN_LOG_NAME,
    RUN_FIGURES_DIR,
    RUN_METRICS_NAME,
    RUN_META_NAME,
    RUN_RESOLVED_CONFIG_NAME,
    RUN_RESULTS_NAME,
    add_throughput_metrics,
    ensure_figures_dir,
    get_logger,
    make_run_dir,
    measure_wall_time,
    resolve_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_json,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)

__all__ = [
    "RUN_LOG_NAME",
    "RUN_FIGURES_DIR",
    "RUN_METRICS_NAME",
    "RUN_META_NAME",
    "RUN_RESOLVED_CONFIG_NAME",
    "RUN_RESULTS_NAME",
    "add_throughput_metrics",
    "ensure_figures_dir",
    "get_logger",
    "make_run_dir",
    "measure_wall_time",
    "resolve_run_dir",
    "setup_logging",
    "validate_run_artifacts",
    "write_json",
    "write_metrics",
    "write_resolved_config",
    "write_results",
    "write_run_meta",
]
