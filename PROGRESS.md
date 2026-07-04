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

## Next
- Awaiting next numbered feature. Candidates surfaced by feature 1: fix run mode, fix
  base-env reset, fix/retire the two broken campaign env configs.
