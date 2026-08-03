<!--
  Fair inverse and data-scale governance preflight profile.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# Inverse / Scale Preflight

This profile freezes two future-execution contracts while the B4 dense-k admission gate remains unresolved. It is deliberately non-executing: it does not train an inverse model, sample candidates, rank candidates, run MPB/Meep replay, or generate any data.

## Inputs

- [Fair 100-target inverse preflight](../../configs/research/inverse_paired_100_preflight_v1.yaml) binds the existing A0 baseline protocol, the frozen test-target manifest, three seeds, `K=128`, `top-M=5`, and 500 real-solver calls per method per seed.
- [100k scale governance preflight](../../configs/research/scale_100k_governance_preflight_v1.yaml) binds OOD-pool rules, near-duplicate exclusion, metadata lineage, protected evaluation, and the cost-record format.
- [100-sample cost record template](../../configs/research/templates/scale_100_sample_cost_record_v1.csv) is a header-only schema. It is not a request to collect 100 samples.

## Fair inverse boundary

Every future method and seed must use the same test `target_index` set and solver budget. Each executable run must resolve checkpoint paths and SHA256 values explicitly; automatic checkpoint selection or missing lineage blocks the comparison. Surrogate ranking metrics, real-MPB replay metrics, and their gap remain separate outputs.

No result from this profile can establish fair inverse superiority, a physics-aware inverse gain, or physical validation. The inverse status remains `blocked` until B4 completes and a separate inverse execution authorization is granted.

## Scale governance boundary

The four governed pools are `reference_id`, `continuous`, `fragmented`, and `boundary_hard`. Their quantitative cutpoints are frozen as reference-population quantile rules, not guessed numeric values: the exact values and the hash of their V2 reference manifest must be recorded before any pool sampling.

Exact duplicates use the registered canonical structure SHA256. Near duplicates use normalized Hamming distance over the metadata-derived geometry degrees of freedom; pairs below `0.01` are in the same duplicate cluster and may not cross train, validation, or protected-test boundaries.

Protected validation, ID-test, and structured/OOD-test manifests must be materialized before pool sampling and then reused unchanged by every composition. A future 100k decision is restricted to `launch_uniform_100k`, `launch_stratified_100k`, `launch_active_sampling`, or `defer`; until learning-curve and cost evidence exist, the default is `defer`.

## Smallest future smoke test

Once a separately authorized calibration run exists, validate exactly 100 completed cost-record rows against the CSV header, required metadata schema, single solver provenance, k-path hash, band-order policy, and duplicate audit. This smoke test does not authorize a 100k launch.
