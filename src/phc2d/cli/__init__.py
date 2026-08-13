"""Stable public entrypoints for the modular phc2d command line.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.cli.main import main
from phc2d.cli.parser import build_parser

__all__ = ["build_parser", "main"]
