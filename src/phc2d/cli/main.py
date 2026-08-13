"""Top-level command dispatch for the phc2d CLI.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from phc2d.cli.commands import (
    data,
    forward,
    hpc,
    inverse,
    models,
    plot,
    query,
    replay,
    report,
)
from phc2d.cli.parser import build_parser

CommandRunner = Callable[[Any], bool]

_RUNNERS: tuple[CommandRunner, ...] = (
    plot.run,
    hpc.run,
    data.run,
    models.run,
    forward.run,
    query.run,
    replay.run,
    inverse.run,
    report.run,
)


def _not_migrated(command: str) -> None:
    raise SystemExit(
        f"`phc2d {command}` is reserved for the unified pipeline, but the executable "
        "backend has not been migrated into src/phc2d yet."
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse and dispatch one CLI command."""

    ns = build_parser().parse_args(argv)
    for run in _RUNNERS:
        if run(ns):
            return
    _not_migrated(ns.command)
