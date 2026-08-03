"""Run inverse-design artifact diagnostics from a YAML config.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phc2d.inverse import run_inverse_design_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_inverse_design_diagnostic(args.config))


if __name__ == "__main__":
    main()
