<!--
  Provenance manifest for BISON manuscript figures.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Figure provenance

Quantitative figures use values copied from accepted local run artifacts and
stored in the adjacent manuscript CSV files. Their plotting script reads only
those CSVs. Architecture schematics are instead traced to the live model code
and formal experiment configuration, as documented separately below.

## `bison_architecture_detailed.svg`, `.pdf`, `.png`, and `.tiff`

- Script: `scripts/generate_bison_architecture.py`
- Implementation source: `src/phc2d/models/baselines.py`
  (`BandSpectralConv2d`, `BandBlock`, and `Fno2DBand`)
- Configuration source: `configs/research/a1_forward_fair_v1.yaml`
- Encoded evaluated settings: width 32, depth 4, retained mode rectangles
  `8x8` and `5x5`, and low-rank width 8.
- Shape boundary: the network diagram keeps the output generic as
  `N_k x N_b`; `31 x 10` is identified only as the evaluated A1 data shape.
- Evidence boundary: this is an implementation-level architecture schematic,
  not performance evidence and not an MPB/Maxwell solver diagram. The query
  operator and inverse generator are intentionally outside the BISON network.
- Rendering: Python/Matplotlib only; SVG retains editable text, PDF uses
  TrueType fonts, and the PNG/TIFF exports are 600 dpi.
- SVG SHA-256:
  `976b88ffe50d58c456dc55d3d216cf97290293085c036cc00ce73f3afa518218`
- PDF SHA-256:
  `d476d3ccf1cdb8239234bd9a423679d9f3d3a70daf8ca87302f3415ec2b500df`
- PNG SHA-256:
  `0bec51d27a5ebe67f783bf1bbab7ce6c2c8e5c01ac56bc83bbefe49c69c48f38`
- TIFF SHA-256:
  `6fd761be1217f5595cb33642964e98658e5ceed0088cb4e7041637a4ec7d43aa`

## Retained aggregate summary: `results_summary.pdf` and `results_summary.png`

This reproducible aggregate visualization is retained with the manuscript
assets but is not included in the current compiled paper after Figure 3 was
replaced by the paired inverse-comparison figure.

- Script: `scripts/generate_results_figure.py`
- Forward source: `docs/results/forward/a1_forward_query_fair_v1_20260726.md`
  plus `runs/20260724/a1_forward_fair_v1_202607240020/per_band_metrics.csv`
- Query source: `docs/results/forward/a1_forward_query_fair_v1_20260726.md`
- Inverse source:
  `runs/20260728/a2_inverse_objective_bridge_seed0_test625_v1_202607280842/aggregate/results.json`
- Inverse aggregate SHA-256:
  `af97717abbd5bc200846aaef583e95d5dfee23cea0370d510e9faa93b593f9e9`
- Inverse per-target table SHA-256:
  `716bb6f426c4b581f0d14d639cf4b751c289bc189ea4695039c84f367149020b`

## `paired_main4.svg` (paired inverse-comparison figure)

- Canonical editable source:
  `docs/reports/assets/phc2d_project_full_introduction_20260724/inverse/paired_main4.svg`
- LaTeX companion export:
  `docs/reports/assets/phc2d_project_full_introduction_20260724/inverse/paired_main4.pdf`
- Preview export:
  `docs/reports/assets/phc2d_project_full_introduction_20260724/inverse/paired_main4.png`
- SVG SHA-256:
  `62830e47306e865476c7b87fffeb9a7a85ea80b259f5fed78090397881d96607`
- PDF SHA-256:
  `17babf05522551ba83b518c18ed8d793363b12c0f6dc09b58d2a426b9f343cc`
- PNG SHA-256:
  `29f5b546187ee7e9d3c508a2013355a5c3e92f21e7850fd04307476fbe0355eb`
- Content: joint best, joint median, largest-disagreement, and joint-worst
  paired comparisons of the single-mode method and four-mode baseline.
- Evidence boundary: qualitative seed-0 examples selected from the complete
  625-target replay; they do not establish cross-seed inverse stability.

## `architecture.tex`

The diagram is based on `src/phc2d/models/baselines.py` (`BandSpectralConv2d`,
`BandBlock`, and `Fno2DBand`), `src/phc2d/inverse/runner.py`
(`_MixtureConditionalGenerator` with `mode_count=1`), and
`configs/research/a2_inverse_objective_bridge_seed0_test625_v1.yaml`.
