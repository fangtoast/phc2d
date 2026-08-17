"""Compatibility facade for reusable full-cell data interfaces.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.data.full_cell_fourier import (
    evaluate_real_fourier_field,
    independent_half_plane_modes,
    periodic_cell_centers,
    wrapped_periodic_delta,
)
from phc2d.data.full_cell_grid import build_full_bz_grid
from phc2d.data.full_cell_hashing import (
    CONTENT_HASH_VERSION,
    PARAMETER_HASH_VERSION,
    canonicalize_structure_payload,
    hash_raster_content,
    hash_structure_geometry,
    hash_structure_parameters,
)
from phc2d.data.full_cell_io import (
    FullCellDatasetBundle,
    load_full_cell_hdf5,
    write_full_cell_hdf5,
    write_full_cell_hdf5_streaming,
)
from phc2d.data.full_cell_raster import (
    DIRECT_CELL_CENTER_V1,
    MASTER_BLOCK_MEAN_V1,
    FullCellRasterPyramid,
    block_mean_downsample,
    build_full_cell_raster_pyramid,
    rasterize_full_cell,
)
from phc2d.data.full_cell_spec import (
    FULL_CELL_FOURIER_SCHEMA_VERSION,
    FULL_CELL_SCHEMA_VERSION,
    MATERIAL_EVALUATORS,
    SUPPORTED_FULL_CELL_FAMILIES,
    FullCellStructureSpec,
)


__all__ = [
    "CONTENT_HASH_VERSION",
    "DIRECT_CELL_CENTER_V1",
    "FULL_CELL_FOURIER_SCHEMA_VERSION",
    "FULL_CELL_SCHEMA_VERSION",
    "MASTER_BLOCK_MEAN_V1",
    "MATERIAL_EVALUATORS",
    "PARAMETER_HASH_VERSION",
    "SUPPORTED_FULL_CELL_FAMILIES",
    "FullCellDatasetBundle",
    "FullCellRasterPyramid",
    "FullCellStructureSpec",
    "block_mean_downsample",
    "build_full_bz_grid",
    "build_full_cell_raster_pyramid",
    "canonicalize_structure_payload",
    "evaluate_real_fourier_field",
    "hash_raster_content",
    "hash_structure_geometry",
    "hash_structure_parameters",
    "independent_half_plane_modes",
    "load_full_cell_hdf5",
    "periodic_cell_centers",
    "rasterize_full_cell",
    "wrapped_periodic_delta",
    "write_full_cell_hdf5",
    "write_full_cell_hdf5_streaming",
]
