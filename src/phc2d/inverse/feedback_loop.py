"""Budgeted orchestration for solver-gated inverse feedback.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from phc2d.inverse.feedback_state import FeedbackState, finish_feedback, propose_feedback


class SolverAttemptFailed(Exception):
    """A recorded physical attempt failed and consumes one candidate budget unit."""


def run_feedback_loop(state: FeedbackState, *,
                      propose: Callable[[FeedbackState], Mapping[str, Any] | None],
                      evaluate: Callable[[FeedbackState], Mapping[str, Any]],
                      persist: Callable[[FeedbackState], None]) -> tuple[FeedbackState, str]:
    """Advance confirmed attempts, persisting the pending state before each solve.

    An already pending candidate requires reconciliation with its external
    solver record; it is never evaluated a second time automatically. The
    evaluator must persist its own raw solver artifacts before returning.
    Only an explicit SolverAttemptFailed is consumed as a solver failure;
    unexpected errors propagate with the last pending state persisted.
    """
    if state.pending is not None:
        return state, 'pending_reconciliation'
    while state.remaining_evaluations > 0:
        candidate = propose(state)
        if candidate is None:
            return state, 'proposal_exhausted'
        pending = propose_feedback(state, **candidate)
        persist(pending)
        try:
            observation = evaluate(pending)
        except SolverAttemptFailed as exc:
            message = str(exc).strip()
            if not message:
                raise ValueError('solver failure must have a reason') from exc
            result = finish_feedback(pending, failure_reason=message)
        else:
            result = finish_feedback(pending, observation=observation)
        persist(result)
        state = result
    return state, 'budget_exhausted'
