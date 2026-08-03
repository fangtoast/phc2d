<!--
  Fixed rules for Codex agents working in the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# AGENTS.md

This file defines fixed implementation rules for future Codex tasks in this repository.

## Research Plan

- Treat [`docs/planning/fno_generative_network_adjustment_plan_20260617.md`](docs/planning/fno_generative_network_adjustment_plan_20260617.md) as the research-plan authority.
- When implementation details are unclear, follow the gates, task families, metrics, replay boundary, and TODO discipline described in that plan and in `docs/planning/research_todo_20260617.md`.

## Repository Boundaries

- New shared/public capabilities must live under `src/phc2d`.
- Research experiment configs must live under `configs/research`.
- Research experiment descriptions must live under `experiments/research`.
- Do not restore `legacy/reproduction` or `research/ai4phc` as active runtime backends.
- Historical code may be consulted only through Git history, then adapted into the unified `src/phc2d` modules.

## Data And Shape Rules

- Do not hard-code `16x16`, `31`, or `10` in models, losses, metrics, plotting, or runners.
- Shapes must come from dataset metadata, schema inspection, config, `DataRepresentationSpec`, or `TargetSpec`.
- `rho_wedge -> Wh` is a baseline task, not the only project task.

## Dependencies

- `torch`, `h5py`, MPB, and Meep must be optional or lazy-imported.
- Basic CLI commands such as `phc2d --help` must not require heavy training or solver dependencies.
- External paper/project code must not be copied directly into `src/phc2d`.
- External references, if downloaded, must go under `external_references/` with `SOURCE.md`, license or license link, commit/hash, download time, and notes about referenced modules.

## Runs And Validation

- All run outputs must follow:

```text
runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/
```

- Standard run artifacts should include `run.log`, `run.meta.json`, `config.resolved.yaml`, `results.json`, `metrics.json`, `history.csv`, and `figures/` when applicable.
- Future tests and acceptance checks must run through WSL `Ubuntu` + conda `meep_env`.
- Windows `.venv` may be used only as an emergency editing/debugging convenience, and its output is not an acceptance-test result.
- Formal MPB/Meep validation must still be explicitly marked as WSL `Ubuntu` + conda `meep_env`.
- Never describe Windows-local surrogate or smoke results as accepted project validation.

## Metrics

- Metrics must include runtime when relevant, such as `wall_time_total_s`, `train_time_s`, `eval_time_s`, `inference_time_per_sample_ms`, `candidate_sampling_time_s`, `surrogate_ranking_time_s`, `mpb_replay_time_s`, and throughput keys.
- Keep surrogate metrics separate from replay metrics.
- Generative inverse reports must distinguish `best_of_k_surrogate_mre`, `best_of_k_replay_mre`, and `surrogate_replay_gap`.

## Model Inventory And Report Completeness

- Any document claiming project-wide or complete model coverage must reconcile
  `phc2d models list`, `docs/reports/phc2d_model_inventory_v1.yaml`, and the
  non-registry inverse methods implemented under `src/phc2d/inverse`.
- Keep registration, implementation, training, formal validation, historical
  evidence, placeholders, and blocked status in separate fields. A model name in
  code or config is not evidence of a completed result.
- Project-wide reports must explicitly check the MLP, ResMLP, CNN, FNO,
  DeepONet, U-Net, and inverse-generator families, or state why a family has no
  admissible result.
- Use exact code/config names such as `mlp`. Do not describe the mixture
  conditional generator as a cVAE or diffusion model, and do not describe
  adapted PINO as a separate faithful PINO architecture.
- Curated report figures belong under `docs/reports/assets/`; complete raw
  galleries remain under ignored `runs/` directories. Curated assets must retain
  source paths, SHA-256 hashes, and evidence boundaries.
- Before changing a project-wide report, run model-coverage, numeric-source, and
  local-link audits. Use `--require-source-artifacts` on the evidence-owning
  workstation.

## Testing

- Every implementation task must include tests.
- If a gate is not ready for tests, explicitly explain why and document the smallest future smoke test.
- At minimum, run the relevant subset of:

```powershell
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m compileall src tests"
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m pytest -q tests"
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m phc2d.cli --help"
wsl -d Ubuntu -e bash -lc "source /home/fangxiao/miniforge3/etc/profile.d/conda.sh && conda activate meep_env && cd '/mnt/d/project/article_reproduction/POD-DeepONET-2d(codex)' && PYTHONPATH=src python -m phc2d.cli plot --help"
```

## Git Discipline

- Commit each completed implementation/documentation task.
- Keep commits scoped to the task just completed.
- Do not stage unrelated user changes, generated data, checkpoints, run outputs, or external reference repositories unless the user explicitly asks.
