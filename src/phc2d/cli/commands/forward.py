"""Forward command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    forward = sub.add_parser("forward", help="Forward surrogate training and evaluation.")
    forward_sub = forward.add_subparsers(dest="action", required=True)
    forward_train = forward_sub.add_parser(
        "train",
        help="Train a registered forward surrogate.",
        description="Train a registered forward surrogate.",
    )
    forward_train.add_argument("--config", type=Path, required=True)
    forward_train.set_defaults(command="forward train")
    forward_eval = forward_sub.add_parser(
        "eval",
        help="Evaluate a frozen forward surrogate checkpoint.",
        description="Evaluate a frozen forward surrogate checkpoint.",
    )
    forward_eval.add_argument("--config", type=Path, required=True)
    forward_eval.set_defaults(command="forward eval")
    forward_benchmark = forward_sub.add_parser(
        "benchmark",
        help="Run a small representation-model benchmark matrix.",
        description="Run a small representation-model benchmark matrix and write leaderboard + per-band summaries.",
    )
    forward_benchmark.add_argument("--config", type=Path, required=True)
    forward_benchmark.set_defaults(command="forward benchmark")


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command == "forward train":
        from phc2d.forward import run_forward_train

        print(run_forward_train(ns.config))
        return True
    if ns.command == "forward eval":
        from phc2d.forward import run_forward_eval

        print(run_forward_eval(ns.config))
        return True
    if ns.command == "forward benchmark":
        from phc2d.forward import run_forward_benchmark

        print(run_forward_benchmark(ns.config))
        return True
    return False
