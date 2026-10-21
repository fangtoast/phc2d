"""Thin CLI for evaluation-only target extension and audited result merging.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--repository-root", type=Path, default=Path("."))
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--campaign", type=Path, required=True)
    render = sub.add_parser("render-report")
    render.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        from phc2d.forward.full_cell_transfer_contracts import freeze_transfer_evaluation
        print(freeze_transfer_evaluation(args.config, args.output, repository_root=args.repository_root))
    elif args.command == "finalize":
        from phc2d.forward.full_cell_transfer_summary import finalize_transfer_summary
        print(finalize_transfer_summary(args.campaign))
    else:
        from phc2d.reporting.full_cell_transfer import render_transfer_report
        print(render_transfer_report(args.campaign))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
