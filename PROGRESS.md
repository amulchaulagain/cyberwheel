# Cyberwheel — Progress Log

Running log so work survives context resets. One entry per numbered feature.

## Task 0 — environment + structure check (done)

**Status:** complete. Committed as `docs: verify structure and complete CLAUDE.md architecture map`.

### Toolchain
- Sandbox has **no system Python 3.10** (system is 3.14) and **no Poetry** by default.
  Provisioned both via `uv`:
  - `uv python install 3.10` → cpython-3.10.19
  - `uv tool install poetry` (Poetry 2.4.1); `~/.local/bin` added to `/etc/sandbox-persistent.sh`
  - `poetry env use $(uv python find 3.10)`
- `poetry install` needs system build tooling: `build-essential python3-dev graphviz graphviz-dev pkg-config`.
- **pygraphviz 1.13 won't compile against system graphviz 14.x** unless you relax a GCC-14 error:
  `CFLAGS="-Wno-incompatible-pointer-types" poetry install`. (Recorded in CLAUDE.md.)
- After that, `poetry install` succeeds and `poetry run python -m cyberwheel ...` imports cleanly.

### Git / hooks
- Removed the `origin` remote (pointed at ORNL/cyberwheel) per the working agreement. No remotes now.
- `.claude/hooks/block-push.sh` exists and is executable; blocks `git push` via exit code 2.

### Structure
- Mapped all core modules and filled in the "paths TBD" line in CLAUDE.md with real paths
  (env/step loop, network, red/blue agents+actions, detectors/alerts, observation, reward,
  runners/trainer, utils). Also fixed a typo and corrected the `-h` note (bare `-h` prints
  nothing; help needs `<mode> <config>`, e.g. `python3 -m cyberwheel help x`).

### Known issues observed (not fixed — out of Task 0 scope)
- `pytest` is not installed in the Poetry env (not in dev deps). Add via `poetry add` when a
  feature needs it.
- **`run` mode crashes** on `environment/cyberwheel.yaml`: `baseline_runner.configure()` calls
  `files(...).joinpath(self.args.network_config)` where `network_config` is a **list** at runtime
  → `TypeError: unsupported operand type(s) for /: 'PosixPath' and 'list'`. The YAML value is a
  single string (`15-host-network.yaml`), so something upstream wraps it in a list. Pre-existing;
  revisit if a future task touches run/baseline paths.

## Feature 1 — testing framework (done)

**Status:** complete. Committed as `feat: add custom test framework with config/smoke/perf
suites, committed perf baseline, and CI gate`.

### What was built
- Custom no-pytest framework: `cyberwheel/tests/framework/` + `python -m cyberwheel.tests` CLI.
  55 cases: 48 config (one per YAML, auto-discovered), 5 smoke, 3 perf benchmarks.
- Perf baseline committed at `cyberwheel/tests/baselines/baseline.json`; convention: re-record
  in the same commit as any intentional perf change, so the working-tree baseline == parent
  commit's results. `--compare-rev REV` measures a parent worktree on the same machine
  (benchmarks ship from HEAD via PYTHONPATH, so the parent needn't contain the framework).
- CI: `.github/workflows/tests.yml` — `functional` job (config --quick + smoke), then
  `perf-gate` (final gate) using `--compare-rev <parent> --tolerance 0.30`. Won't trigger in
  this sandbox (never pushes) — expected.
- Baseline numbers (this sandbox, aarch64/8cpu): network_build_200host ≈ 0.68 s,
  sim_step_sps_15host ≈ 394 k steps/s (inactive-agent base env), train_sps_15host ≈ 63 steps/s.

### Key decisions
- Smoke train/eval run as subprocesses through the real CLI (`train
  train_rl_red_agent_vs_rl_blue.yaml` with tiny overrides: total_timesteps 16, num_steps 8,
  num_envs 1 → 2 PPO updates, ~4 s; eval loads the saved TEST_ checkpoint, 1×10 steps).
- Import-order gotcha: `cyberwheel.utils` must be imported before network/detector modules
  (cycle: network.host → utils.host_types → utils/__init__ → red_actions → detectors.alert →
  network.host). `__main__.py` masks this; test fns/benchmarks import cyberwheel.utils first.
- W&B/network isolation: track=false + download_model=false + env (WANDB_MODE=disabled,
  TQDM_DISABLE=1) set for all framework processes and job-wide in CI.

### Pre-existing bugs found (encoded as xfail known-issue cases, NOT fixed)
1. `run` mode crashes: `baseline_runner.py:23` host-keyed service_mapping + missing
   `agent_config` vs `art_agent.py:101,119` (smoke:run_mode_known_issue).
2. Base-env `reset()` broken: `inactive_red_agent.py:31-32` calls `ARTAgent.reset()` without
   its required `(network, service_mapping)` args (smoke:base_env_reset_known_issue).
   Also forced bench_sim_step to measure stepping without resets.
3. `environment/art_campaign_vs_rl_blue.yaml` and `environment/rl_red_campaign_vs_rl_blue.yaml`
   reference `network/emulator_15_host.yaml`, which does not exist (config xfails).
4. `environment/cyberwheel.yaml` references `red_agent: inactive_red_agent.yaml` (file absent) —
   harmless, never consumed by code (INFO).
If any get fixed, the runner reports XPASS_WARN so the known-issue entry gets promoted/removed.

## Feature 2 — performance + profiler (done)

**Status:** complete. Committed as `feat: add reusable env profiler with committed baselines;
fix profiler-found bottlenecks and latent crashes`.

### What was built
- `cyberwheel/profiler/` + `python3 -m cyberwheel.profiler` CLI. Scenarios: `network-build`,
  `sim-step` (base env), `rl-step` (full RL step: masks, actions, detector, obs, reward),
  `train` (opt-in cProfile of a tiny real train run). Three measurement passes per scenario so
  observers don't skew gated numbers: plain timed loop (gated metrics), instrumented pass
  (per-phase table via `MethodInstrumenter` — non-invasive method wrapping with
  inclusive/exclusive nesting), cProfile pass (hotspot tables). `--network-size` overrides the
  env config's compat size to profile big networks.
- Committed profiler baseline `cyberwheel/profiler/baselines/profile_baseline.json`
  (`--record-baseline` / `--check`, exit 2 on regression; sub-floor metrics reported but not
  gated: phases < 0.02 ms/step, one-shot `ms` metrics < 1 ms — a post-commit `--check` flaked
  on `network_build/service_map_ms` (~50 µs, 10x sample spread on identical code), which is
  what set the one-shot floor). Same parent-commit convention as the test baseline;
  same-machine only.
- New perf-suite benchmark `bench_rl_step.py` (metric `rl_step_sps_15host`) gating the full
  RL step path in CI; self-contained so `--compare-rev` can measure a parent without the
  profiler. `measure_metrics(skip_failing=True)` in compare-rev mode: a benchmark the parent
  cannot run is reported as NEW instead of failing the gate (the parent has crashes this
  benchmark trips; see below).

### Bottlenecks found (profiler evidence) and fixes
Pre-fix rl-step @15-host: 0.107 ms/step (RL red vs RL blue), 0.164 (ART vs RL blue);
@200-host medium: 0.822 ms/step. Phase attribution + cProfile pinned, in cost order:
1. `RedObservation.obs_vec` kept as a Python list, converted `np.array(list)` twice per step
   (49% of step @15-host; 82% @200-host) → obs_vec is now a persistent int64 ndarray updated
   in place; `get_observation_space` returns it directly (consumers copy into tensors).
2. `DeployDecoyHost.execute` re-validated a pydantic `HostType` (deep `Service` set) on every
   deploy (~24% of loop) → prototype built once in `__init__`, shared read-only.
3. `create_host_type_from_yaml` re-parsed `windows_exploitable_services.yaml` and rebuilt the
   HostType per host at network build (~93% of 1000-host build) → module-level caches
   (`_WINDOWS_SERVICES_CACHE`, `_HOST_TYPE_CACHE`); 1000-host build 4113 → 302 ms.
4. Technique-validity scans (`get_service_map`, ART `get_valid_techniques_by_host`,
   `ARTAgent.__init__`) recomputed per host though hosts share (os, cve_list) profiles →
   one cached helper `cyberwheel.utils.get_valid_techniques_by_host`; 1000-host service map
   46 → 3 ms; also collapses the per-decoy cost in `handle_network_change`.
5. `ARTAgent.handle_network_change` diffed host sets O(hosts) every step (15% of ART step) →
   `Network.topology_version` counter (bumped on host add/remove/reset) + early-out when
   unchanged. Note: the removed-hosts branch now cleans *all* removed hosts per pass (old code
   popped one per step; identical under the ≤1-removal-per-step reality, safer otherwise).
6. `RLReward.get_valid_targets` rebuilt the target set every step → cached per
   (network id, topology_version); `reset()` invalidates (covers leader re-rolls).
7. `ARTKillChainPhase/PingSweep/PortScan.sim_execute` refiltered each technique's atomic
   tests per action → `get_valid_tests` memoized per (mitre_id, os). Same list order, so the
   `random.choice` sequence is unchanged.
8. `BlueObservation` zeroed the alert region with a per-element Python loop → numpy slice.
9. `DetectorHandler.obs` rebuilt a NodeDataView per edge and re-added nodes → hoisted node
   accessor, in-place list mutation (semantics identical; all nodes own `detector_output`).
10. Trainer: `RLHandler.reset()` called `envs.reset()` once *per agent* — redundant work AND
    each agent got obs from a different reset (only the last described live state). Now resets
    once and slices per agent. **Behavior change** (training trajectories shift vs old code);
    correctness fix, documented here.

### Latent crash bugs found by the profiler's randomized rl-step driver (both fixed)
- RL red action space keeps hosts removed from the network (decoy torn down): selecting one
  crashed `RLARTAgent.act` with KeyError. Now a failed no-op (`target_host = "invalid"`
  sentinel, matching the eval path / reward functions).
- A pingswept decoy entered the RL red observation/action space (`handle_action`) but never
  `service_mapping` (only `handle_network_change` populated it, and RLARTAgent never calls
  it) → Discovery/LateralMovement on it crashed. `handle_action` now backfills the mapping
  (cached helper); `RLARTAgent.from_yaml` gained the missing `all_kcps`.
  Both were reachable in real training (same masks); the parent commit crashes bench_rl_step
  within a few hundred steps.

### Results (this sandbox, aarch64/8cpu; same-machine measurements)
- rl-step @15-host: 0.107 → 0.029 ms/step (3.7x); ART vs RL blue: 0.164 → 0.055 (3.0x).
- rl-step @200-host medium: 0.822 → 0.048 ms/step (17x); reset 0.79 → 0.25 ms.
- network build @1000-host: 4113 → 302 ms (13.6x); service map 46 → 3 ms.
- Perf gate vs parent (`--compare-rev HEAD`, same machine): network_build_200host_s -91%,
  sim_step ~level, train_sps ~level (torch policy forward dominates at bench scale — see
  `train` scenario hotspots: `torch._C._nn.linear` is ~64% of internal time).
- Baselines re-recorded in this commit: `cyberwheel/tests/baselines/baseline.json` (now with
  `rl_step_sps_15host`) and `cyberwheel/profiler/baselines/profile_baseline.json`.

### Notes / conventions
- Profiler baseline convention documented in CLAUDE.md (re-record with intentional perf
  changes, same commit). The profiler reuses the test framework's baseline machinery
  (`cyberwheel.tests.framework.baseline`) — one schema, one compare.
- The `train` scenario is never gated (cProfile skews absolute numbers); gated training SPS
  remains the perf suite's `train_sps_15host`.

## Next
- Awaiting next numbered feature. Candidates surfaced by feature 1: fix run mode, fix
  base-env reset, fix/retire the two broken campaign env configs. Surfaced by feature 2:
  trainer-side wins (batch policy forwards are dominated by per-call torch overhead at
  num_envs=1; eval agents re-run orthogonal init before load_state_dict), and
  `RLReward.network` is never updated on multi-network reset (pre-existing; reward computed
  against the initial network's decoys).
