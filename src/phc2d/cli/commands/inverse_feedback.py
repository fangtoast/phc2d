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
    source.add_argument('--dispatch-directory', type=Path, help='Inspect a persisted dispatch journal.')
    parser.add_argument('--round', type=Path, action='append', default=[],
                        help='Journal directory in chronological order; repeat for each round.')
    parser.add_argument('--initial-sha256', help='Externally retained dispatch journal initial hash.')
    parser.add_argument('--head-sha256', help='Externally retained dispatch journal head hash.')
    parser.set_defaults(command='inverse feedback-inspect')


def run_feedback_inspect(ns: Any) -> None:
    """Recompute a saved journal or ordered chain without submitting work."""
    from phc2d.inverse.feedback_journal import (summarize_dispatch_journal, summarize_robust_chain,
                                               summarize_robust_round)

    if ns.dispatch_directory is not None:
        if ns.round or not ns.initial_sha256 or not ns.head_sha256:
            raise SystemExit('--dispatch-directory requires --initial-sha256 and --head-sha256, without --round.')
        summary = summarize_dispatch_journal(ns.dispatch_directory, initial_sha256=ns.initial_sha256,
                                             expected_head=ns.head_sha256)
    elif ns.initial_sha256 or ns.head_sha256:
        raise SystemExit('--initial-sha256 and --head-sha256 require --dispatch-directory.')
    elif ns.directory is not None:
        if ns.round:
            raise SystemExit('--round requires --initial-state.')
        summary = summarize_robust_round(ns.directory)
    else:
        from phc2d.inverse.feedback_state import restore_feedback

        initial = restore_feedback(json.loads(ns.initial_state.read_bytes()))
        summary = summarize_robust_chain(initial, ns.round)
    print(json.dumps(summary, indent=2, allow_nan=False))
