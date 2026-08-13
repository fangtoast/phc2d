"""Lightweight top-level parser assembly for the phc2d CLI.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse

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

_CONFIGURATORS = (
    data.configure,
    forward.configure,
    query.configure,
    models.configure,
    inverse.configure,
    replay.configure,
    hpc.configure,
    plot.configure,
    report.configure,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without importing training or solver backends."""

    parser = argparse.ArgumentParser(
        prog="phc2d",
        description="Unified 2D photonic-crystal workflow CLI.",
    )
    parser.add_argument("--version", action="version", version="phc2d 0.2.0")
    sub = parser.add_subparsers(dest="domain", required=True)
    for configure in _CONFIGURATORS:
        configure(sub)
    return parser
