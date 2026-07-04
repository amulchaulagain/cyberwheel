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

## Next
- Awaiting Task 1.
