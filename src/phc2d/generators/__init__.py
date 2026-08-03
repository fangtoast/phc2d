"""Structure generator registry for 2D photonic-crystal datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.generators.registry import (
    StructureSample,
    build_structure_generator,
    collect_structure_metadata,
)

__all__ = ["StructureSample", "build_structure_generator", "collect_structure_metadata"]
