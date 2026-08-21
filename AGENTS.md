<!--
  Fixed rules for Codex agents working in the unified phc2d workspace.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

# AGENTS.md

This file is the project constitution for every AI Agent working in this repository.
It defines non-negotiable boundaries, not project background, tutorials, or task-specific knowledge.

## 1. Project Principles

- Separate human-readable knowledge from machine-readable evidence.
- Human-readable knowledge explains concepts, methods, decisions, and conclusions.
- Machine-readable evidence records configs, manifests, logs, metrics, checkpoints, hashes, and solver outputs.
- Never copy raw evidence into prose merely to make a document look complete.
- Never sacrifice repository architecture, evidence integrity, or compatibility to finish one task faster.
- Reuse an existing abstraction before creating a parallel implementation.
- Extend the nearest responsible module unless a new boundary is demonstrably necessary.
- Do not create a new top-level directory without an explicit migration or governance decision.
- Temporary task output, chat reasoning, smoke results, and Agent notes are not permanent project knowledge.
- AI-created knowledge is Draft until a human owner reviews and accepts it.
- For changes, declare one primary layer: L0 research goal, L1 scientific argument, L2 method architecture, L3 experiment/evidence, or L4 implementation.
- Cross-layer reading is allowed; unscoped cross-layer mutation is not.
- Current artifacts and live repository behavior outrank memory, TODOs, plans, and narrative summaries.
- Evidence authority is:

```text
run / solver / metrics / checkpoint
  > current code / config / tests / CLI
  > curated result document
  > TODO / plan
  > synthesis report / chat summary
```

## 2. Repository Responsibilities

| Path | Responsibility |
|---|---|
| `src/phc2d/` | Reusable implementation, public APIs, domain logic, workflow services, and infrastructure adapters |
| `experiments/` | Stable experiment specifications: question, inputs, configs, gates, baselines, and expected evidence |
| `configs/` | Machine-readable parameters; `research/` for research and `smoke/` for non-scientific smoke checks |
| `runs/` | Raw machine-generated execution evidence, organized by Experiment ID and timestamp |
| `docs/` | Human-readable project knowledge, methods, decisions, guides, curated results, and archives |
| `scripts/` | Thin user-facing wrappers or one-time migration tools; not a reusable implementation layer |
| `jobs/` | Scheduler launchers and platform templates; not scientific logic or result authority |
| `tests/` | Unit, contract, regression, CLI, and formal validation tests |
| `.agent/` | Ignored local Agent workspace; never canonical knowledge or formal evidence |
| `data/` | Local datasets, schemas, manifests, and data provenance; large datasets remain untracked |
| `checkpoints/` | Local model artifacts; never documentation and normally untracked |

The following distinctions are mandatory:

- `runs/` is not `docs/results/`: runs are raw evidence; results are curated human interpretation.
- `experiments/` is not `src/phc2d/workflows/`: experiments define what and why; workflows implement how.
- `models/` is not task orchestration: model code defines architectures and model contracts only.
- `scripts/` is not reusable implementation: reusable logic belongs under `src/phc2d/`.
- `docs/` is not an execution log: ordinary command traces and raw metadata belong in `runs/`.
- `docs/results/` is not `docs/reports/`: results answer a bounded evidence question; reports synthesize multiple results.

## 3. Coding Rules

Before writing code, an Agent must:

1. Search for existing implementations, callers, configs, tests, and compatibility paths.
2. Identify the nearest reusable abstraction.
3. State the intended module responsibility and dependency direction.
4. Decide whether to extend, split, or create; creation requires a clear missing boundary.
5. Identify the smallest tests that prove the changed contract.

Implementation rules:

- New shared or public capabilities belong under `src/phc2d/`.
- Shapes must come from metadata, schema inspection, config, `DataRepresentationSpec`, or `TargetSpec`.
- Do not hard-code research-specific dimensions such as `16x16`, `31`, or `10` in reusable code.
- `torch`, `h5py`, MPB, and Meep must remain optional or lazy-imported where basic CLI use does not require them.
- `phc2d --help` and lightweight inspection commands must not import heavy training or solver stacks.
- Model classes must not own filesystem paths, run directories, plotting, or report generation.
- A single function must not combine data loading, model construction, training, evaluation, saving, and plotting.
- Do not copy a model implementation for one experiment; express variation through config, composition, registry, or a named reusable component.
- Do not place substantial reusable logic in `scripts/` or scheduler files.
- Do not create generic garbage-drawer modules such as unbounded `utils.py` or `helpers.py`.
- Name modules by stable responsibility; names such as `final_v2_new_fix.py` are forbidden.
- Avoid importing private functions across domain modules; promote a stable interface when reuse is real.
- Preserve model identity, factory identity, architecture fingerprint, state-dict schema, and checkpoint compatibility when refactoring models.
- External project code must not be copied directly into `src/phc2d/`; record sources and licenses under `external_references/`.

## 4. Refactor Gate

At the end of every medium or larger code task, inspect the changed surface for:

- duplicated logic;
- oversized functions;
- oversized modules;
- responsibility leakage;
- missing reusable interfaces;
- stale implementations;
- unnecessary wrappers.

Apply these gates:

- A function over 300 lines or a Python module over 800 lines is an architectural warning, not a normal endpoint.
- Do not increase an oversized unit without either splitting the touched responsibility or recording why compatibility blocks the split.
- Prefer extracting a tested domain operation over adding another conditional branch to a monolithic runner.
- Remove stale code only after checking imports, CLI routes, configs, tests, reports, and checkpoint compatibility.
- Keep compatibility facades thin and explicitly marked; do not add new business logic to them.
- Do not perform unrelated large-scale refactoring merely for elegance.
- If the necessary refactor exceeds current scope, document the smallest boundary problem and stop at a safe seam.

## 5. Experiment Rules

Every new formal research experiment must have a stable Experiment ID:

- `EXP-FWD-xxx` for forward surrogate experiments;
- `EXP-QRY-xxx` for continuous-query experiments;
- `EXP-INV-xxx` for inverse-design experiments;
- `EXP-PHY-xxx` for MPB, Meep, field, or physical-validation experiments;
- `EXP-DAT-xxx` for dataset generation, representation, or provenance experiments.

The Experiment ID must connect the full chain:

```text
experiment spec -> config -> run artifacts -> curated result
```

- Dates are secondary execution metadata, not the primary scientific identity.
- Existing historical experiment names remain valid until an explicit migration maps them to stable IDs.
- Formal experiment specifications belong under `experiments/research/`; configs belong under `configs/research/`.
- A spec defines the question, baseline, dataset identity, split, metrics, gates, failure conditions, and expected evidence.
- A resolved config snapshot must be stored with every run.
- Run outputs must follow `runs/{yyyymmdd}/{experiment_id}_{YYYYMMDDHHmm}/`.
- Standard artifacts include `run.log`, `run.meta.json`, `config.resolved.yaml`, `results.json`, `metrics.json`, `history.csv`, and `figures/` when applicable.
- Runtime and throughput metrics are required when relevant.
- Keep surrogate metrics, proxy metrics, solver replay, and physical-validation metrics in separate fields.
- Generative inverse evidence must distinguish surrogate ranking, real-MPB replay, and the surrogate-replay gap.
- Smoke, preflight, dry-run, config readiness, and single-seed pilots must not be promoted into formal completion claims.
- MPB/Meep and formal acceptance must run in WSL `Ubuntu` with conda `meep_env`.
- Windows `.venv` output may support editing or local diagnostics but is not project acceptance evidence.
- Remote or expensive experiments require explicit authorization; monitoring is read-only by default, with no automatic retry, cancellation, resubmission, or resource change.

## 6. Documentation Rules

Human-readable documentation has these roles:

| Role | Canonical location |
|---|---|
| Project knowledge | `README.md`, `PROJECT_MAP.md`, `docs/index.md`, `docs/README.md` |
| Research concepts | `docs/概念/` |
| Methods | `docs/方法/` |
| Experiment results | `docs/results/` |
| Research decisions and long-term plans | `PROJECT_MAP.md`, `docs/planning/` |
| Operational guides | `docs/开始/`, `docs/指南/` |
| Reference material | `docs/参考/` |
| Historical archive | `docs/历史档案.md` and explicitly archived pages |
| Dated synthesis reports | `docs/reports/` |

- Keep the root README stable and free of volatile benchmark numbers and today-only status.
- Put current state, blockers, and next gates in `PROJECT_MAP.md`.
- `docs/` must not contain ordinary execution logs, scheduler output, raw metric dumps, or copied manifests.
- Ordinary runs do not generate long Markdown reports.
- Create or update a result document only when evidence changes a scientific conclusion, comparison, gate, or decision.
- A result document should answer only:
  1. What is the result?
  2. How does it compare with the baseline?
  3. What did we learn?
  4. What is the next decision?
  5. Where is the raw evidence?
- Registration, implementation, training, validation, historical evidence, placeholder, and blocked status must remain distinct.
- A model name in code or config is not evidence of a completed result.
- Project-wide model claims must reconcile the live model registry, the model inventory, and non-registry inverse methods.
- New or renamed canonical Markdown must be reachable from `docs/README.md` or a linked index.
- New human-facing filenames under `docs/` should be concise Chinese unless the name is a fixed identifier or compatibility path.
- AI-created or substantially rewritten documents remain Draft until human acceptance.
- Documentation completion never upgrades experimental completion.

## 7. Planning Lifecycle

- Short task plans remain in the Agent's temporary working state and are not written to the repository.
- TODO lists, chat summaries, and implementation scratchpads are not permanent planning documents.
- Only these plans may be preserved long term:
  - research-stage plans;
  - architecture adjustments;
  - experiment campaigns;
  - multi-task roadmaps.
- Long-term plans belong under `docs/planning/` and must be reachable from its index.
- Every preserved plan must declare scope, owner, status, effective date, gates, and related results.
- Use explicit lifecycle states such as `active`, `completed`, `superseded`, or `reference`.
- A plan becoming `completed` requires a result or evidence link; a checked TODO alone is insufficient.
- Superseded plans remain historical records and must not silently govern current execution.
- Do not turn daily Agent activity into dated planning Markdown.

## 8. Documentation Writing Style

All human-facing research documentation must:

- lead with the conclusion;
- prefer tables for comparisons, states, mappings, and gates;
- link evidence near the claim it supports;
- distinguish fact, interpretation, decision, and unknown;
- avoid chronological work logs and implementation diaries;
- avoid describing what an Agent did unless the Agent behavior itself is under study;
- avoid repeating raw metadata already present in run artifacts;
- use exact project identifiers, tensor shapes, units, and evidence levels;
- keep one page focused on one reader question;
- link downward instead of duplicating long explanations across entry pages.

## 9. Change Discipline

Before changing the project:

- Read the nearest authority files; project-wide work also requires `PROJECT_MAP.md`, `docs/README.md`, and `docs/collaboration.md`.
- Inspect `git status` and preserve unrelated user changes.
- Confirm the authorized files, primary layer, scientific boundary, and non-goals.
- Prefer the smallest coherent change that leaves a reusable seam.

If a task changes directory structure or canonical paths:

1. Propose a migration plan before moving files.
2. Inventory callers, imports, configs, tests, links, artifacts, and historical identities.
3. Migrate in small batches with compatibility handling.
4. Validate each batch before continuing.
5. Never perform a one-shot mass relocation of historical files.

Validation and closeout rules:

- Every implementation task requires tests for changed behavior.
- Run the narrowest relevant checks first, then the required broader regression.
- Formal acceptance uses WSL `Ubuntu` with conda `meep_env`; report local and formal validation separately.
- Documentation changes must pass `phc2d report docs-audit --root README.md --strict` when applicable.
- Project-wide reports must pass the current model/source audit when applicable.
- Do not claim a gate passed when required data, artifacts, tests, or physical replay are missing.
- Do not stage datasets, checkpoints, run outputs, caches, secrets, local paths, or unrelated dirty files.
- Commit completed implementation or documentation work as a scoped commit unless the user explicitly forbids committing.
- End with: changed files, validation performed, evidence boundary, commit status, and the next human decision or gate.

When this file conflicts with a project skill, cached memory, plan, or historical note, follow this file and stronger current repository evidence.
