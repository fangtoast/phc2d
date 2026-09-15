"""Command-line control plane for EXP-FWD-021.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from phc2d.forward.full_cell_multidomain_contracts import (
    audit_training_runs,
    freeze_campaign,
)
from phc2d.forward.full_cell_multidomain_evaluation import (
    evaluation_status,
    finalize_evaluation,
    freeze_evaluation,
    run_evaluation_task,
)
from phc2d.forward.full_cell_multidomain_training import run_training_task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze datasets, subsets, and 63 tasks")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--repository-root", default=".")
    freeze.add_argument("--dataset-path-key")
    freeze.add_argument("--skip-file-hash", action="store_true")

    train = subparsers.add_parser("run-training-task", help="run one manifest training task")
    train.add_argument("--campaign", required=True)
    train.add_argument("--task-index", type=int, required=True)
    train.add_argument("--admission-probe", action="store_true")
    train.add_argument("--device")
    train.add_argument("--max-train-batches", type=int)
    train.add_argument("--max-validation-samples", type=int)

    audit = subparsers.add_parser("audit-training", help="audit a contiguous formal task slice")
    audit.add_argument("--campaign", required=True)
    audit.add_argument("--start-index", type=int, default=0)
    audit.add_argument("--end-index", type=int)

    evaluate_freeze = subparsers.add_parser(
        "freeze-evaluation", help="freeze checkpoints and read-once held-out caches"
    )
    evaluate_freeze.add_argument("--campaign", required=True)

    evaluate = subparsers.add_parser("run-evaluation-task", help="run one checkpoint-target cell")
    evaluate.add_argument("--campaign", required=True)
    evaluate.add_argument("--cell-index", type=int, required=True)
    evaluate.add_argument("--device")

    status = subparsers.add_parser("status", help="inspect evaluation completeness")
    status.add_argument("--campaign", required=True)

    finalize = subparsers.add_parser("finalize", help="finalize a complete 378-cell matrix")
    finalize.add_argument("--campaign", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        output = freeze_campaign(
            args.config,
            args.output,
            repository_root=args.repository_root,
            dataset_path_key=args.dataset_path_key,
            verify_file_hash=not args.skip_file_hash,
        )
        print(output)
    elif args.command == "run-training-task":
        output = run_training_task(
            args.campaign,
            args.task_index,
            admission_probe=args.admission_probe,
            device_override=args.device,
            max_train_batches=args.max_train_batches,
            max_validation_samples=args.max_validation_samples,
        )
        print(output)
    elif args.command == "audit-training":
        campaign = Path(args.campaign).resolve()
        result = audit_training_runs(
            campaign / "task_manifest.csv",
            start_index=args.start_index,
            end_index=args.end_index,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "freeze-evaluation":
        print(freeze_evaluation(args.campaign))
    elif args.command == "run-evaluation-task":
        print(run_evaluation_task(args.campaign, args.cell_index, device_override=args.device))
    elif args.command == "status":
        print(json.dumps(evaluation_status(args.campaign), indent=2))
    elif args.command == "finalize":
        print(finalize_evaluation(args.campaign))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
