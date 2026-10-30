"""Durable single-writer replay for the synchronous physical-search baseline.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from phc2d.inverse.feedback_loop import SolverAttemptFailed


def _serialize(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _write_once(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class DirectSearchJournal:
    """Replay completed physical attempts while refusing ambiguous attempts.

    One controller owns a directory. The immutable contract must include the
    source identity, complete search and solver configuration, code and SciPy
    identities, and any resource budget. It is the caller's responsibility to
    persist raw solver evidence before returning an evaluation, verify that
    evidence, and account for the initial incumbent and scheduler allocation.
    This journal does not submit, reconcile, retry, or lock external jobs.
    When supplied, ``verify_evidence`` must be read-only and verify the
    candidate, immutable evidence reference and stored observation on both initial use and replay;
    it must raise on failure and return ``None`` on success.
    Without it, replay alone cannot establish that raw evidence still exists.

    Use ``before_solve``, ``evaluate`` and ``after_solve`` as the callbacks to
    ``search_fourier_physically``; call ``finish`` after a normal return. On a
    restart, replay starts from the same seeded search and consumes recorded
    observations without invoking the solver. An attempt without a recorded
    result is ambiguous and must be reconciled externally; it is never retried.
    """

    def __init__(self, directory: str | Path, *, contract: Mapping[str, Any],
                 evaluator: Callable[[Any], Mapping[str, Any]],
                 verify_evidence: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], None] | None = None) -> None:
        if (not isinstance(contract, Mapping) or not contract or not callable(evaluator)
                or (verify_evidence is not None and not callable(verify_evidence))):
            raise ValueError('nonempty frozen contract and evaluator are required')
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.verify_evidence = verify_evidence
        self._active: dict[str, Any] | None = None
        self._visited: set[str] = set()
        identity = {'schema': 'direct_search_journal_v1', 'contract': dict(contract)}
        if verify_evidence is not None:
            identity['requires_evidence_verification'] = True
        contract_bytes = _serialize(identity)
        contract_path = self.path / 'contract.json'
        if contract_path.exists():
            if contract_path.read_bytes() != contract_bytes:
                raise ValueError('search journal contract mismatch')
        else:
            if any(self.path.iterdir()):
                raise ValueError('nonempty search journal has no contract')
            _write_once(contract_path, contract_bytes)

    def before_solve(self, event: Mapping[str, Any]) -> None:
        if self._active is not None:
            raise ValueError('previous search attempt is not reconciled')
        index = event.get('candidate_index')
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError('invalid search attempt index')
        name = f'case_{index:08d}'
        if name in self._visited:
            raise ValueError('search attempt already visited')
        directory = self.path / name
        encoded = _serialize(dict(event))
        fresh = not directory.exists()
        if fresh:
            directory.mkdir(exist_ok=False)
            _write_once(directory / 'request.json', encoded)
        elif (directory / 'request.json').read_bytes() != encoded:
            raise ValueError('replayed search trajectory differs')
        result = directory / 'result.json'
        if not fresh and not result.exists():
            raise RuntimeError('unresolved prior solver attempt requires external reconciliation')
        self._active = {'directory': directory, 'event': encoded, 'fresh': fresh}
        self._visited.add(name)

    def evaluate(self, candidate: Any) -> Mapping[str, Any]:
        active = self._active
        if active is None:
            raise ValueError('search attempt has not been recorded')
        if active.get('evaluated'):
            raise ValueError('search attempt already evaluated')
        active['evaluated'] = True
        event = json.loads(active['event'])
        if getattr(candidate, 'parameter_hash', None) != event.get('parameter_hash'):
            raise ValueError('search candidate identity differs')
        path = active['directory'] / 'result.json'
        if active['fresh']:
            try:
                evaluated = self.evaluator(candidate)
            except SolverAttemptFailed as exc:
                reason = str(exc).strip()
                if not reason:
                    raise ValueError('solver failure must have a reason') from exc
                _write_once(path, _serialize({'kind': 'failure', 'reason': reason}))
                raise
            if not isinstance(evaluated, Mapping) or not isinstance(evaluated.get('observation'), Mapping):
                raise ValueError('evaluator must return a gateable observation')
            # Raw traces remain in the evaluator's own immutable evidence store.
            record = {'kind': 'observation', 'status': evaluated.get('status'),
                      'observation': dict(evaluated['observation'])}
            if 'evidence' in evaluated:
                if not isinstance(evaluated['evidence'], Mapping):
                    raise ValueError('evidence reference must be a mapping')
                record['evidence'] = dict(evaluated['evidence'])
            self._verify_evidence(candidate, record)
            _write_once(path, _serialize(record))
        else:
            record = json.loads(path.read_bytes())
            if record.get('kind') == 'observation':
                self._verify_evidence(candidate, record)
        if record.get('kind') == 'failure' and isinstance(record.get('reason'), str) and record['reason'].strip():
            raise SolverAttemptFailed(record['reason'])
        if record.get('kind') != 'observation' or not isinstance(record.get('observation'), dict):
            raise ValueError('invalid recorded search result')
        return {'status': record.get('status'), 'observation': record['observation']}

    def _verify_evidence(self, candidate: Any, record: Mapping[str, Any]) -> None:
        if self.verify_evidence is not None:
            if not isinstance(record.get('evidence'), Mapping) or not record['evidence']:
                raise ValueError('recorded physical evidence reference is required')
            decision_input = {'status': record.get('status'), 'observation': record.get('observation')}
            if self.verify_evidence(candidate, record['evidence'], decision_input) is not None:
                raise ValueError('evidence verifier must return None on success')

    def after_solve(self, event: Mapping[str, Any], outcome: Mapping[str, Any]) -> None:
        active = self._active
        if active is None or _serialize(dict(event)) != active['event']:
            raise ValueError('search outcome has no matching request')
        result_bytes = (active['directory'] / 'result.json').read_bytes()
        receipt = _serialize({'result_sha256': hashlib.sha256(result_bytes).hexdigest(),
                              'outcome': dict(outcome)})
        path = active['directory'] / 'receipt.json'
        if path.exists():
            if path.read_bytes() != receipt:
                raise ValueError('replayed search outcome differs')
        else:
            _write_once(path, receipt)
        self._active = None

    def finish(self) -> None:
        """Reject incomplete attempts and previously recorded divergent paths."""
        if self._active is not None:
            raise ValueError('search attempt remains unreconciled')
        names = {path.name for path in self.path.iterdir() if path.name != 'contract.json'}
        if names != self._visited:
            raise ValueError('search replay did not consume the complete journal')
        if any(not (self.path / name / 'receipt.json').is_file() for name in names):
            raise ValueError('search journal has an incomplete receipt')
