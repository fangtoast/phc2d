"""Compatibility namespace for the unified plotting API.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.plot import (
    PlotRequest,
    PlotResult,
    plot_band_overlay,
    plot_field_map,
    plot_medium_map,
    plot_metric_report,
)

__all__ = [
    "PlotRequest",
    "PlotResult",
    "plot_band_overlay",
    "plot_field_map",
    "plot_medium_map",
    "plot_metric_report",
]
