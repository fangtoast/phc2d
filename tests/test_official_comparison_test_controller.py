"""GPU capacity accounting for incremental EXP-FWD-031 tests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from types import SimpleNamespace

from jobs.paracloud_official_comparison_test_controller import gpu_jobs


def test_gpu_jobs_counts_running_and_pending_allocations(monkeypatch):
    monkeypatch.setenv("USER", "test-user")

    def fake_run(command, **kwargs):
        assert command[:3] == ["squeue", "-u", "test-user"]
        return SimpleNamespace(stdout="gpu:1\ngres/gpu:1\nN/A\ngpu:1\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert gpu_jobs() == 3
