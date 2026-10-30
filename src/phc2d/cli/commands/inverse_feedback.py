"""CLI surface for inspecting inverse-feedback journals.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def configure_feedback_inspect(sub: Any) -> None:
    """Register the read-only feedback inspection action."""
    parser = sub.add_parser(
        'feedback-inspect',
        help='Verify and inspect a robust feedback journal or ordered chain.',
        description=('Recompute saved feedback receipts without dispatching jobs. '
                     'The initial incumbent requires independent verification.'),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--directory', type=Path, help='Inspect one journal directory.')
    source.add_argument('--initial-state', type=Path, help='Initial state for an ordered chain.')
    parser.add_argument('--round', type=Path, action='append', default=[],
                        help='Journal directory in chronological order; repeat for each round.')
    parser.set_defaults(command='inverse feedback-inspect')


def run_feedback_inspect(ns: Any) -> None:
    """Recompute a saved journal or ordered chain without submitting work."""
    from phc2d.inverse.feedback_journal import summarize_robust_chain, summarize_robust_round

    if ns.directory is not None:
        if ns.round:
            raise SystemExit('--round requires --initial-state.')
        summary = summarize_robust_round(ns.directory)
    else:
        from phc2d.inverse.feedback_state import restore_feedback

        initial = restore_feedback(json.loads(ns.initial_state.read_bytes()))
        summary = summarize_robust_chain(initial, ns.round)
    print(json.dumps(summary, indent=2, allow_nan=False))
