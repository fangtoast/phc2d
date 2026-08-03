"""Reusable plotting API for datasets, bands, metrics, and fields.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.plot.api import (
    PlotRequest,
    PlotResult,
    plot_band_overlay,
    plot_field_map,
    plot_medium_map,
    plot_metric_report,
)
from phc2d.plot.training_curves import build_learning_curve_package

__all__ = [
    "PlotRequest",
    "PlotResult",
    "plot_band_overlay",
    "plot_field_map",
    "plot_medium_map",
    "plot_metric_report",
    "build_learning_curve_package",
]
