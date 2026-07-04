# Cyberwheel — RL cyber-defense simulation environment

A Gymnasium RL environment for training and evaluating autonomous red and blue cyber
agents on simulated networks. Config-driven; networks are `networkx` graphs. Based on
ORNL/cyberwheel - treat this file as ground truth and update it whenever structure changes.

## Stack & tooling
- Python 3.10. Dependency manager: **Poetry** (`pyproject.toml`, `poetry.lock`).
- RL: Gymnasium env + PyTorch, PPO-style trainer. Tracking: Weights &
  Biases. Visualization: Dash / Plotly. Network graphs: `networkx`.
- Env note: this sandbox has no system Python 3.10 — it's provisioned via `uv python install 3.10`
  and Poetry (`uv tool install poetry`; `poetry env use $(uv python find 3.10)`). Building
  `pygraphviz` against the system graphviz 14.x needs `CFLAGS="-Wno-incompatible-pointer-types"`
  during `poetry install` (GCC 14 hard-errors on graphviz's pointer-type mismatch otherwise).

## How to run  — `python3 -m cyberwheel <mode> <config>.yaml`
- `train`             — trains agent(s); saves to `cyberwheel/data/models/<experiment_name>/`; logs to W&B.
- `evaluate`          — evaluates a trained run; writes `cyberwheel/data/action_logs/<name>.csv`;
                        if `visualize: true` in the config, also writes `cyberwheel/data/graphs/<name>/`.
- `visualizer <port>` — Dash server to replay evaluations in the browser (e.g. `visualizer 8050`).
- `emulate`           — like `evaluate`, but on the FIREWHEEL emulation backend.
- `run`               — steps the env with inactive agents (no RL); scaffolding / sanity only.
- CLI args override YAML values. Dispatch needs `<mode> <config>` (2+ argv); a bare `-h` prints
  nothing. To list all parameters use an unknown mode, e.g. `python3 -m cyberwheel help x`.

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
- `cyberwheel/__main__.py` — CLI entry point / mode dispatch (`train`/`evaluate`/`emulate`/`run`/`visualizer`).
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
- **Blue agent:** `cyberwheel/blue_agents/` — `blue_agent.py`, `rl_blue_agent.py`,
  `random_blue_agent.py`, `inactive_blue_agent.py`; `action_space/` (`action_space.py`, `discrete.py`).
- **Blue actions:** `cyberwheel/blue_actions/` — `blue_action.py` base; `actions/` (DeployDecoyHost,
  RemoveDecoyHost, IsolateDecoy, Nothing); `shared_data/`.
- **Detectors / alerts:** `cyberwheel/detectors/` — `detector_base.py`, `alert.py`, `handler.py`;
  `detectors/` (`probability_detector.py`, `isolate_detector.py`, `example_detectors.py`).
- **Observation:** `cyberwheel/observation/` — `observation.py`, `blue_observation.py`,
  `red_observation.py`, `observation_attributes.py`.
- **Reward:** `cyberwheel/reward/` — `reward_base.py`, `rl_reward.py`, `rl_split_reward.py`,
  `decoy_reward.py`, `isolate_reward.py`, `blue_reward_functions.py`, `red_reward_functions.py`,
  `step_detected_reward.py`, `step_accomplished.py`.
- **RL trainer / runners:** `cyberwheel/runners/` — `rl_trainer.py` (trainer), `train_cyberwheel.py`,
  `evaluate_cyberwheel.py`, `rl_evaluator.py`, `rl_handler.py`, `baseline_runner.py`,
  `run_baseline_cyberwheel.py`, `run_visualization_server.py`, `visualizer.py`.
- **Utils:** `cyberwheel/utils/` — arg parsing (`parse_override_args.py`), `yaml_config.py`,
  `rl_policy.py`, `get_service_map.py`, `host_types.py`, `step_metrics.py`, `set_seed.py`.
- `cyberwheel/legacy/` — old multiagent / pyattck / scripts; not part of the active path.

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
