# Cyberwheel — RL cyber-defense simulation environment

A Gymnasium RL environment for training and evaluating autonomous red and blue cyber
agents on simulated networks. Config-driven; networks are `networkx` graphs. Based on
ORNL/cyberwheel - treat this file as ground truth and update it whenever structure changes.

## Stack & tooling
- Python 3.10. Dependency manager: **Poetry** (`pyproject.toml`, `poetry.lock`).
- RL: Gymnasium env + PyTorch, PPO-style trainer. Tracking: Weights &
  Biases. Network graphs: `networkx`. (graphviz/pygraphviz and Dash/Plotly are retired —
  do not reintroduce them.)
- Web UI: FastAPI + uvicorn (`cyberwheel/server/`) serving a JSON API and the built React
  SPA from `cyberwheel/server/static/` (source in `frontend/`, Vite build). Graph rendering is
  a custom Canvas 2D renderer over a deterministic pure-Python layout (no graphviz).
  **Rebuild rule:** any change under `frontend/src` requires `cd frontend && npm run build`
  (emits into `cyberwheel/server/static/`) committed in the SAME commit — the committed bundle
  is what `python -m cyberwheel frontend <port>` serves, and the frontend suite's static-serving
  case fails loudly if it is missing.
- Env note: this sandbox has no system Python 3.10 — it's provisioned via `uv python install 3.10`
  and Poetry (`uv tool install poetry`; `poetry env use $(uv python find 3.10)`).

## How to run  — `python3 -m cyberwheel <mode> <config>.yaml`
- `train`             — trains agent(s); saves to `cyberwheel/data/models/<experiment_name>/`; logs to W&B.
- `evaluate`          — evaluates a trained run; writes `cyberwheel/data/action_logs/<name>.csv`
                        plus `<name>.summary.json` (per-episode/per-seed/overall reward stats with
                        95% Student-t CIs); if `visualize: true`, also writes `cyberwheel/data/graphs/<name>/`.
                        Optional `seeds:` list (or `--seeds 1,2,3`) runs a multi-seed batch in one
                        process — one CSV/viz dir with a global episode index + `seed` column; each
                        seed reseeds its episode block regardless of `deterministic`. Served by
                        `GET /api/runs/<id>/summary` and a summary panel in the UI.
- `emulate`           — like `evaluate`, but on the FIREWHEEL emulation backend.
- `run`               — steps the env with inactive agents (no RL); scaffolding / sanity only.
- `frontend <port>`   — experimentation web UI + API (e.g. `frontend 8080`; binds 0.0.0.0).
  Launches train/evaluate runs as CLI subprocesses driven by generated env configs in
  `data/configs/environment/generated/<run_id>.yaml` (never CLI overrides: `agents.*`,
  per-optimizer LRs, reward selection, and the string `anneal_lr` are unreachable/corrupted
  via flags). Run registry (config snapshot + status + stdout) lives in
  `cyberwheel/data/frontend/runs/<run_id>/`; both locations are git-ignored.
  Dashboard multi-select (up to 6 runs) → `/compare` view: overlaid training curves
  (episodic return + SPS), evaluation-summary comparison (mean ± 95% CI), config diff.
  Frontend-only feature — it composes existing endpoints plus `GET /api/runs/<id>/summary`.
- CLI args override YAML values. Dispatch needs `<mode> <config>` (2+ argv); a bare `-h` prints
  nothing. To list all parameters use an unknown mode, e.g. `python3 -m cyberwheel help x`.

## Testing — `python3 -m cyberwheel.tests`
- Custom framework (no pytest) in `cyberwheel/tests/framework/`; suites: `config` (every YAML
  loads + references resolve), `smoke` (train/evaluate end-to-end at tiny scale via the real
  CLI, incl. batch evaluate + summary stats, viz artifacts + layout determinism), `frontend`
  (boots the experimentation server and drives it over HTTP: options, SPA, train/evaluate +
  batch-evaluate e2e via the API, stop/orphan, validation, delete), `perf` (benchmarks gated
  against the parent commit's results).
- Common invocations: `--suite {config,smoke,perf,all}`, `--quick`, `--list`, `--json PATH`,
  `--record-baseline`, `--compare-rev REV` (same-machine dual measurement), `--tolerance F`.
  Exit codes: 0 ok · 1 config/smoke failure · 2 perf regression · 3 framework error.
- **Baseline convention:** `cyberwheel/tests/baselines/baseline.json` is committed; re-record it
  (`--record-baseline`) in the SAME commit as any intentional perf change. The working-tree
  baseline therefore always holds the parent commit's results; CI re-measures the parent on the
  same runner via `--compare-rev` (`.github/workflows/tests.yml`, final `perf-gate` job).
- Known pre-existing bugs are encoded as xfail ("known-issue") cases, not fixed: `run` mode
  (baseline_runner service_mapping/agent_config), base-env `reset()` (InactiveRedAgent),
  and two env configs referencing the missing `network/emulator_15_host.yaml`.
- Perf benchmarks: `bench_network_build` (200-host), `bench_sim_step` (inactive base env),
  `bench_rl_step` (full RL env step: masks, blue/red actions, detector, obs, reward),
  `bench_train_sps` (real train CLI). In `--compare-rev` mode a benchmark the compared
  revision cannot run is skipped there and its metric reported as NEW (not gated).

## Profiling — `python3 -m cyberwheel.profiler`
- Reusable env profiler in `cyberwheel/profiler/`: per-phase wall-time attribution
  (blue/red act, detector, observations, reward, action mask, reset) via non-invasive method
  wrapping + cProfile hotspot tables, per scenario: `network-build`, `sim-step`, `rl-step`
  (deterministic defaults), `train` (opt-in; cProfiles a tiny real train run, never gated).
- Options: `--scenario X` (repeatable, or `all`), `--network`, `--env-config`, `--network-size`
  (profile networks bigger than the env config's compat size), `--quick`, `--json PATH`,
  `--top N`, `--no-hotspots`, `--seed N`. Exit codes: 0 ok · 2 regression (`--check`) · 3 error.
- **Profile baseline:** `cyberwheel/profiler/baselines/profile_baseline.json` is committed and
  follows the same parent-commit convention as the test baseline: `--record-baseline` in the
  SAME commit as intentional perf changes; `--check` gates against it (exit 2), same-machine
  only. Sub-floor metrics are reported but never gated (timer noise): phases < 0.02 ms/step,
  one-shot `ms` metrics < 1 ms (per-step `ms/step` metrics are gated at any magnitude).

## Architecture (conceptual — stable)
- **Env loop (per step):** the red agent acts on a host → the action emits **Alerts** → the
  blue agent's **detector** layer filters/perturbs those Alerts → Alerts become the blue
  agent's **observation** → **reward** is computed → repeat.
- **Network:** routers → subnets → hosts, as `networkx` nodes. Hosts carry services (ports,
  CVEs, OS). Built from a network config.
- **Red agents:** (a) *RL red* — observation is a limited network view that expands as it
  explores; (b) *ART agent* — heuristic; maps MITRE ATT&CK killchain phases → Atomic Red Team
  techniques, validated against a target host's OS / phase / CVEs; (c) *ART campaign* — a fixed
  killchain of specific techniques (used for emulation).
- **Blue agent:** deploys decoys to slow/stop red; observation = full network + detector alerts.
- **Detectors / Alerts:** a pluggable detector stack that can drop alerts, add noise, or emit
  false positives before they reach the observation.
- Supports **multi-agent** (train RL red vs RL blue simultaneously) and **multi-network**
  training (fixed obs sizes: small=100 hosts / medium=1000 / large=10000).

## Where things live
- `cyberwheel/data/configs/` — all YAML configs, organized by concern: `environment/` (training params + which agents), `red_agent/`, `blue_agent/`,
  plus network, services, host-types, decoys, detectors, and campaign definitions.
- `cyberwheel/data/models/`, `.../action_logs/`, `.../graphs/` — run artifacts (git-ignored).
- `cyberwheel/emulator/` — FIREWHEEL emulation setup + its own README.
- `cyberwheel/__main__.py` — CLI entry point / mode dispatch (`train`/`evaluate`/`emulate`/`run`).
- **Env / step loop:** `cyberwheel/cyberwheel_envs/` — `cyberwheel.py` (base env + per-step loop,
  `step()` returns Red/BlueAgentResult), `cyberwheel_rl.py` (Gymnasium RL wrapper; `step()` returns
  the gym `(obs, reward, term, trunc, info)` tuple), `cyberwheel_emulator.py` (FIREWHEEL backend).
- **Network:** `cyberwheel/network/` — `network_base.py` (the `networkx` graph), plus
  `router.py`, `subnet.py`, `host.py`, `service.py`, `process.py`, `command.py`, `network_object.py`;
  generation in `network/network_generation/` (`network_generator.py`, `example.py`).
- **Red agents:** `cyberwheel/red_agents/` — `red_agent_base.py`, `rl_red_agent.py` (RL red),
  `art_agent.py` (heuristic ART), `art_campaign.py` (fixed killchain), `rl_red_campaign.py`,
  `emulator_rl_red_campaign.py`, `inactive_red_agent.py`; `action_space/red_discrete.py`;
  `strategies/` (bfs/dfs exfiltration, server downtime, brute force, impact — `red_strategy.py` base).
- **Red actions:** `cyberwheel/red_actions/` — `red_base.py`, `technique.py`, `art_techniques.py`,
  `atomic_test.py`; concrete killchain phases in `actions/` (discovery, port scan, ping sweep,
  privilege escalation, lateral movement, impact, `art_killchain_phase.py`, `nothing.py`).
  Exploit success is binary by default; opt-in CVSS-weighted probabilistic success (env flag
  `probabilistic_exploits` / `--probabilistic-exploits`, floor/ceiling knobs, optional
  `exploit_severity_config` CVE→[0,1] map in `data/configs/exploit_severity/`) lives in
  `utils/exploit_model.py` and gates `ARTKillChainPhase.sim_execute` — RNG-neutral when off.
- **Blue agent:** `cyberwheel/blue_agents/` — `blue_agent.py`, `rl_blue_agent.py`,
  `random_blue_agent.py`, `inactive_blue_agent.py`; `action_space/` (`action_space.py`, `discrete.py`).
- **Blue actions:** `cyberwheel/blue_actions/` — `blue_action.py` base (Standalone/Host/Subnet/Range);
  `actions/` (DeployDecoyHost, RemoveDecoyHost, IsolateDecoy, Nothing; active-defense on real hosts:
  QuarantineHost, RestoreHost, PatchHost); `shared_data/`. Active-defense actions ship in the
  `active_defense_blue_agent.yaml` config (default `rl_blue_agent.yaml` unchanged so old models load);
  quarantine is enforced red-side via `host.isolated` at the killchain chokepoint, restore/patch signal
  red through `network.pending_restores`/`pending_patches` (drained each red step).
- **Detectors / alerts:** `cyberwheel/detectors/` — `detector_base.py`, `alert.py`, `handler.py`;
  `detectors/` (`probability_detector.py`, `isolate_detector.py`, `example_detectors.py`).
- **Observation:** `cyberwheel/observation/` — `observation.py`, `blue_observation.py`,
  `red_observation.py`, `observation_attributes.py`.
- **Reward:** `cyberwheel/reward/` — `reward_base.py`, `rl_reward.py`, `rl_split_reward.py`,
  `decoy_reward.py`, `isolate_reward.py`, `blue_reward_functions.py`, `red_reward_functions.py`,
  `step_detected_reward.py`, `step_accomplished.py`.
- **RL trainer / runners:** `cyberwheel/runners/` — `rl_trainer.py` (trainer), `train_cyberwheel.py`,
  `evaluate_cyberwheel.py`, `rl_evaluator.py`, `rl_handler.py`, `baseline_runner.py`,
  `run_baseline_cyberwheel.py`.
- **Visualization artifacts:** `cyberwheel/visualization/` — `layout.py` (deterministic
  radial-cluster layout, graphviz-free), `states.py` (killchain state codes), `writer.py`
  (`VizWriter`: per-step JSON deltas written during evaluate when `visualize: true` →
  `data/graphs/<graph_name>/{meta,layout,episode_<n>}.json`).
- **Experimentation server:** `cyberwheel/server/` — `app.py`/`serve.py` (FastAPI factory +
  uvicorn entry), `routes/` (options/runs/eval endpoints), `jobs.py` (subprocess launch/stop/
  reap), `registry.py` (run.json store + orphan detection + external model dirs), `options.py`
  (config enumeration), `metrics.py` (TensorBoard EventAccumulator reader), `actions_log.py`,
  `paths.py`, `validation.py`; built SPA committed in `static/`. Endpoints take/return plain
  dicts — keep pydantic out of handler signatures (repo pins pydantic v1).
- **Utils:** `cyberwheel/utils/` — arg parsing (`parse_override_args.py`), `yaml_config.py`,
  `rl_policy.py`, `get_service_map.py`, `host_types.py`, `step_metrics.py` (evaluation summary
  stats: t-table, mean/std/95% CI, summary builder+writer; stdlib-only), `set_seed.py`.
- `cyberwheel/legacy/` — old multiagent / pyattck / scripts; not part of the active path.
- **Tests:** `cyberwheel/tests/framework/` — `core.py` (registry/runner), `cli.py`,
  `baseline.py` (perf baseline + comparison), `gitio.py`, `artifacts.py`,
  `suites/{config,smoke,perf}_suite.py`, `benchmarks/bench_*.py` (standalone scripts);
  committed baseline in `cyberwheel/tests/baselines/`. The old `cyberwheel/tests/*.py` files
  are stale (pre-reorg imports) and are not imported by the framework.
- **Profiler:** `cyberwheel/profiler/` — `cli.py`/`__main__.py`, `scenarios.py` (workloads +
  drivers), `phase_timer.py` (PhaseAccumulator/MethodInstrumenter), `hotspots.py` (cProfile),
  `baseline_io.py` (reuses the test framework's baseline machinery), `report.py`;
  committed baseline in `cyberwheel/profiler/baselines/profile_baseline.json`.

## Conventions & red lines
- **Never `git push`. Never add or restore a git remote.** (A hook enforces this; do not
  attempt to disable it.)
- Dependencies: **Poetry only** — `poetry add <pkg>`; never pip or requirements.txt. Commit
  the updated `poetry.lock`.
- Follow SOLID. Add new agents / actions / detectors / rewards via config + subclassing, not
  by editing core control flow.
- One commit per numbered feature; never bundle unrelated changes. Commit locally; never push.
- **Definition of done** (every feature): tests pass **and** the profiler shows no regression
  vs the parent commit **and** you have shown the evidence (test output + profiler numbers).
