"""Record peak resident memory for one Linux process tree.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _process_table() -> tuple[dict[int, int], dict[int, int]]:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text(encoding="utf-8").split()
            pid, parent = int(fields[0]), int(fields[3])
            status = stat.with_name("status").read_text(encoding="utf-8")
            rss_kib = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
        parents[pid] = parent
        rss[pid] = rss_kib * 1024
    return parents, rss


def _tree_rss(root_pid: int) -> tuple[bool, int]:
    parents, rss = _process_table()
    descendants = {int(root_pid)}
    changed = True
    while changed:
        before = len(descendants)
        descendants.update(pid for pid, parent in parents.items() if parent in descendants)
        changed = len(descendants) != before
    return int(root_pid) in parents, sum(rss.get(pid, 0) for pid in descendants)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    peak = 0
    while True:
        alive, current = _tree_rss(args.pid)
        peak = max(peak, current)
        if not alive:
            break
        time.sleep(max(args.interval, 0.1))
    args.output.write_text(f"{peak}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
