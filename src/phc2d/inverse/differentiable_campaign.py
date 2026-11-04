"""Prepare paired initial geometries for surrogate inverse campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from typing import Any, Mapping

from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_geometry


def prepare_paired_initials(source: Any, *, base: Mapping[str, Any],
                            campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Choose each first eligible registered initial without consulting a surrogate or solver."""
    keys = tuple((int(i), str(component)) for i, component in base['coefficient_keys'])
    reference = [float(source.parameters['modes'][i][component]) for i, component in keys]
    reference.append(float(source.parameters['threshold']))
    scale, lo, hi = base['scales'], base['lower'], base['upper']
    if any(len(x) != len(reference) for x in (scale, lo, hi)):
        raise ValueError('parameter dimensions differ')
    records = []
    for index, options in enumerate(campaign['initial_options']):
        attempts = []
        selected = None
        for option in options:
            if len(option) != len(reference) or any(not lo[i] <= z <= hi[i] for i, z in enumerate(option)):
                raise ValueError('initial point outside parameter bounds')
            values = [reference[i] + scale[i] * z for i, z in enumerate(option)]
            structure = update_fourier_binary(source, sample_id=f'{source.sample_id}-initial-{index}',
                coefficients={key: float(values[i]) for i, key in enumerate(keys)},
                threshold=float(values[-1]), max_mode_radius=base['max_mode_radius'])
            geometry = screen_fourier_geometry(structure, raster_method='master_block_mean_v1',
                source_resolution=base['source_resolution'], input_resolution=base['input_resolution'],
                material_fraction_bounds=tuple(base['material_fraction_bounds']))
            attempts.append({'point': list(option), 'status': geometry.status,
                             'parameter_hash': structure.parameter_hash,
                             'material_fraction': geometry.material_fraction})
            if geometry.status == 'eligible':
                selected = list(option)
                break
        records.append({'initial_index': index, 'status': 'ready' if selected is not None else 'no_valid_initial',
                        'initial': selected, 'attempts': attempts})
    return records
