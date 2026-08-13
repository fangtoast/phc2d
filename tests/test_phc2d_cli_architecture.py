"""Architecture contracts for the modular CLI package.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import phc2d.cli
from phc2d.cli import build_parser
from phc2d.cli import main as public_main
from phc2d.cli.main import main as modular_main


EXPECTED_COMMANDS = {
    "data accept-cluster",
    "data converge",
    "data convert",
    "data finalize",
    "data generate",
    "data inspect",
    "data make-smoke",
    "data prepare-full-cell-plan",
    "forward benchmark",
    "forward eval",
    "forward train",
    "hpc adapters",
    "hpc audit",
    "hpc benchmark",
    "hpc finalize",
    "hpc plan",
    "hpc solve-shard",
    "hpc status",
    "inverse benchmark",
    "inverse design-diagnose",
    "inverse diagnose",
    "inverse eval",
    "inverse full-cell-preflight",
    "inverse full-cell-train",
    "inverse objective-train",
    "inverse rerank",
    "inverse sample",
    "inverse train",
    "models list",
    "plot",
    "query audit",
    "query benchmark",
    "query data-scaling",
    "query preflight",
    "replay mpb",
    "report",
    "report build",
    "report docs-audit",
    "report model-audit",
}


def _registered_commands(parser: argparse.ArgumentParser) -> set[str]:
    commands: set[str] = set()
    command = parser._defaults.get("command")
    if command is not None:
        commands.add(str(command))
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                commands.update(_registered_commands(child))
    return commands


def test_public_cli_entrypoint_is_a_thin_compatibility_facade() -> None:
    assert public_main is modular_main
    assert len(inspect.getsource(phc2d.cli).splitlines()) <= 30


def test_cli_command_contract_is_preserved_after_module_split() -> None:
    assert _registered_commands(build_parser()) == EXPECTED_COMMANDS


def test_cli_modules_stay_scoped_by_command_domain() -> None:
    cli_dir = Path(inspect.getsourcefile(phc2d.cli)).parent
    assert len((cli_dir / "main.py").read_text(encoding="utf-8").splitlines()) <= 100
    assert len((cli_dir / "parser.py").read_text(encoding="utf-8").splitlines()) <= 100
    command_sizes = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in (cli_dir / "commands").glob("*.py")
    }
    assert command_sizes
    assert max(command_sizes.values()) <= 250
