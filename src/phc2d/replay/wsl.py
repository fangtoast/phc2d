"""Command helpers for WSL Ubuntu conda meep_env validation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from shlex import quote
from typing import Sequence

WSL_MEEP_ENVIRONMENT = "wsl_ubuntu_conda_meep_env"


def build_wsl_meep_command(
    *,
    repo_path_wsl: str | PurePosixPath,
    python_args: Sequence[str],
    conda_sh: str | PurePosixPath = "/home/fangxiao/miniforge3/etc/profile.d/conda.sh",
    env_name: str = "meep_env",
) -> list[str]:
    """Build a Windows-side command that runs Python inside WSL ``meep_env``."""
    repo = quote(str(repo_path_wsl))
    args = " ".join(quote(str(arg)) for arg in python_args)
    script = (
        f"source {quote(str(conda_sh))} && "
        f"conda activate {quote(env_name)} && "
        f"cd {repo} && "
        f"python {args}"
    )
    return ["wsl", "-d", "Ubuntu", "-e", "bash", "-lc", script]
