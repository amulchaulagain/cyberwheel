"""Frontend suite: the experimentation server works end-to-end.

Boots ``python -m cyberwheel frontend <port>`` once (lazily, on the first
case that needs it) and drives it over HTTP with stdlib urllib: options
enumeration, SPA serving, a real tiny training run launched through the
API, an evaluation with visualization artifacts, job stop/orphan handling,
and input validation. The final case deletes every run the suite created
through the DELETE endpoint (which is itself under test); an atexit hook
backstops both the server process and stray registry entries.

All runs the suite launches use the ``tfs-`` display-name prefix so cleanup
can target exactly them.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import yaml

from cyberwheel.tests.framework.core import (
    CONFIG_ROOT,
    DATA_ROOT,
    REPO_ROOT,
    SAFE_ENV,
    Context,
    Outcome,
    Registry,
    Status,
    TestCase,
    check,
    check_file,
)

SUITE = "frontend"
PREFIX = "tfs"

_NETWORK = "15-host-network.yaml"
_TINY_TRAIN_PARAMS = {
    "network_config": [_NETWORK],
    "total_timesteps": 16,
    "num_steps": 8,
    "num_envs": 1,
    "num_saves": 1,
    "num_minibatches": 2,
    "update_epochs": 2,
    "eval_episodes": 1,
    "async_env": False,
    "track": False,
    "device": "cpu",
    "seed": 1,
    "deterministic": True,
}

_STATE: dict = {}


class _Server:
    def __init__(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        env = dict(os.environ)
        env.update(SAFE_ENV)
        self.log_path = DATA_ROOT / "frontend" / f"{PREFIX}-server.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cyberwheel", "frontend", str(self.port)],
            cwd=REPO_ROOT,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"frontend server exited {self.proc.returncode} during startup; "
                    f"see {self.log_path}"
                )
            try:
                self.request("GET", "/api/health")
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.25)
        raise AssertionError("frontend server did not become healthy within 60s")

    def request(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def api(self, method: str, path: str, body: dict | None = None, expect: int = 200) -> dict:
        status, _, raw = self.request(method, path, body)
        check(
            status == expect,
            f"{method} {path} -> {status} (expected {expect}): {raw[:300]!r}",
        )
        return json.loads(raw) if raw else {}

    def shutdown(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        self._log.close()
        self.log_path.unlink(missing_ok=True)


def _server() -> _Server:
    if "server" not in _STATE:
        _STATE["server"] = _Server()
    return _STATE["server"]


def _fs_cleanup() -> None:
    """Backstop: remove suite runs directly from disk (normally the final
    case already deleted them through the API)."""
    registry_dir = DATA_ROOT / "frontend" / "runs"
    if registry_dir.is_dir():
        for entry in registry_dir.glob(f"{PREFIX}-*"):
            shutil.rmtree(entry, ignore_errors=True)
    for sub in ("models", "runs", "graphs"):
        base = DATA_ROOT / sub
        if base.is_dir():
            for entry in base.glob(f"{PREFIX}-*"):
                shutil.rmtree(entry, ignore_errors=True)
    for pattern in (
        DATA_ROOT.glob(f"action_logs/{PREFIX}-*.csv"),
        (CONFIG_ROOT / "environment" / "generated").glob(f"{PREFIX}-*.yaml"),
    ):
        for entry in pattern:
            entry.unlink(missing_ok=True)


def _atexit() -> None:
    server = _STATE.pop("server", None)
    if server is not None:
        server.shutdown()
    _fs_cleanup()


atexit.register(_atexit)


def _poll_until_done(server: _Server, run_id: str, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        record = server.api("GET", f"/api/runs/{run_id}")
        if record["status"] not in ("queued", "running"):
            return record
        time.sleep(1.0)
    raise AssertionError(f"run {run_id} still {record['status']} after {timeout_s}s")


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def _case_options() -> Outcome:
    server = _server()
    options = server.api("GET", "/api/options")

    on_disk = sorted(p.name for p in (CONFIG_ROOT / "environment").glob("*.yaml"))
    listed = sorted(
        name for names in options["env_configs"].values() for name in names
    )
    check(listed == on_disk, f"env configs mismatch: api={listed} disk={on_disk}")
    for field, subdir in (
        ("red_agents", "red_agent"),
        ("blue_agents", "blue_agent"),
        ("detector_configs", "detector"),
        ("decoy_configs", "decoy_hosts"),
        ("host_configs", "host_definitions"),
    ):
        disk = sorted(p.name for p in (CONFIG_ROOT / subdir).glob("*.yaml"))
        check(options[field] == disk, f"{field} mismatch: {options[field]} != {disk}")
    network_15 = next(
        n for n in options["network_configs"] if n["file"] == _NETWORK
    )
    check(network_15["hosts"] == 15, f"15-host network reports {network_15['hosts']} hosts")
    check("reward_red_delay" in options["blue_reward_functions"], "blue reward fns missing")
    check("reward_decoy_hits" in options["red_reward_functions"], "red reward fns missing")
    check(
        "active_defense_blue_agent.yaml" in options["blue_agents"],
        "active-defense blue agent not enumerated",
    )

    defaults = server.api(
        "GET", "/api/options/env-config/train_rl_red_agent_vs_rl_blue.yaml"
    )["params"]
    with open(CONFIG_ROOT / "environment" / "train_rl_red_agent_vs_rl_blue.yaml") as f:
        expected = yaml.safe_load(f)
    check(defaults == expected, "env-config defaults do not round-trip the YAML")
    server.api("GET", "/api/options/env-config/nope.yaml", expect=404)
    return Outcome(Status.PASS, f"{len(listed)} env configs enumerated + defaults round-trip")


def _case_static() -> Outcome:
    server = _server()
    status, headers, raw = server.request("GET", "/")
    check(status == 200, f"GET / -> {status}")
    check(b"<!doctype html>" in raw.lower(), "GET / did not return HTML")
    status, _, spa_raw = server.request("GET", "/runs/some/client/route")
    check(status == 200 and spa_raw == raw, "SPA fallback did not serve index.html")
    body = server.api("GET", "/api/nope", expect=404)
    check("error" in body, f"API 404 lacks error envelope: {body}")
    return Outcome(Status.PASS, "index + SPA fallback + JSON 404 OK")


def _case_train() -> Outcome:
    server = _server()
    created = server.api(
        "POST",
        "/api/runs/train",
        {
            "display_name": f"{PREFIX} train e2e",
            "base_config": "train_rl_red_agent_vs_rl_blue.yaml",
            "params": dict(_TINY_TRAIN_PARAMS),
        },
        expect=201,
    )
    run_id = created["id"]
    _STATE["train_id"] = run_id
    record = _poll_until_done(server, run_id, 600)
    check(record["status"] == "succeeded", f"train ended {record['status']}")

    generated = CONFIG_ROOT / "environment" / "generated" / f"{run_id}.yaml"
    check_file(generated)
    with open(generated) as f:
        config = yaml.safe_load(f)
    check(config["experiment_name"] == run_id, "generated config experiment_name mismatch")
    check(config["anneal_lr"] == "cosine_restarts", "anneal_lr string not preserved")
    check(config["network_config"] == [_NETWORK], "network_config list not preserved")
    check_file(DATA_ROOT / "models" / run_id / "blue_agent.pt", min_size=1024)

    metrics = server.api("GET", f"/api/runs/{run_id}/metrics")
    check("charts/SPS" in metrics["tags"].get("charts", []), "charts/SPS tag missing")
    check(metrics["last_step"] == 16, f"last_step {metrics['last_step']} != 16")
    scalars = server.api(
        "GET", f"/api/runs/{run_id}/metrics/scalars?tags=charts/SPS"
    )["series"]["charts/SPS"]
    check(len(scalars) == 2, f"expected 2 SPS points, got {len(scalars)}")
    logs = server.api("GET", f"/api/runs/{run_id}/logs")
    check("SPS:" in logs["content"], "stdout log missing SPS lines")
    checkpoints = server.api("GET", f"/api/runs/{run_id}/checkpoints")
    check(checkpoints["checkpoints"][:1] == ["agent"], f"checkpoints: {checkpoints}")
    check(sorted(checkpoints["agents"]) == ["blue", "red"], f"agents: {checkpoints}")
    return Outcome(Status.PASS, f"trained via API ({run_id}); metrics/logs/checkpoints OK")


def _case_evaluate() -> Outcome:
    server = _server()
    num_steps = 10
    created = server.api(
        "POST",
        "/api/runs/evaluate",
        {
            "display_name": f"{PREFIX} eval e2e",
            "base_config": "evaluate_rl_red_vs_rl_blue.yaml",
            "source": {"run_id": _STATE["train_id"]},
            "checkpoint": "agent",
            "params": {
                "network_config": _NETWORK,
                "num_episodes": 1,
                "num_steps": num_steps,
                "seed": 1,
                "deterministic": True,
            },
        },
        expect=201,
    )
    run_id = created["id"]
    _STATE["eval_id"] = run_id
    check(created["visualize"] is True, "visualize should default to true")
    record = _poll_until_done(server, run_id, 600)
    check(record["status"] == "succeeded", f"evaluate ended {record['status']}")

    actions = server.api("GET", f"/api/runs/{run_id}/actions?episode=0")
    check(len(actions["rows"]) == num_steps, f"{len(actions['rows'])} action rows")
    check(actions["episodes"] == [0], f"episodes: {actions['episodes']}")

    summary = server.api("GET", f"/api/runs/{run_id}/summary")
    check(
        len(summary["seeds"]) == 1 and summary["overall"]["total_reward"]["n"] == 1,
        f"single-seed summary wrong: seeds {summary['seeds']}, "
        f"overall {summary['overall'].get('total_reward')}",
    )

    meta = server.api("GET", f"/api/runs/{run_id}/viz/meta")
    check(meta["episodes_written"] == [0], f"viz episodes_written: {meta}")
    layout = server.api("GET", f"/api/runs/{run_id}/viz/layout")
    node_count = len(layout["nodes"])
    check(node_count == meta["counts"]["nodes"], "layout/meta node count mismatch")
    episode = server.api("GET", f"/api/runs/{run_id}/viz/episodes/0")
    check(len(episode["steps"]) == num_steps, f"{len(episode['steps'])} viz frames")
    for frame in episode["steps"]:
        for added in frame.get("add", []):
            check(added["id"] >= node_count, "dynamic decoy id collides with static ids")
    server.api("GET", f"/api/runs/{run_id}/viz/episodes/99", expect=404)
    return Outcome(Status.PASS, f"evaluated via API; {node_count} layout nodes, viz + actions OK")


def _case_evaluate_batch() -> Outcome:
    server = _server()
    num_steps = 10

    # Validation: bad seeds params are rejected before anything launches.
    base_body = {
        "base_config": "evaluate_rl_red_vs_rl_blue.yaml",
        "source": {"run_id": _STATE["train_id"]},
        "checkpoint": "agent",
    }
    for label, bad_seeds in (
        ("string", "1,2"),
        ("duplicates", [1, 1]),
        ("too many", list(range(21))),
    ):
        body = server.api(
            "POST",
            "/api/runs/evaluate",
            {
                **base_body,
                "display_name": f"{PREFIX} bad batch {label}",
                "params": {"seeds": bad_seeds},
            },
            expect=400,
        )
        check("seeds" in body["error"]["message"], f"unhelpful seeds error: {body}")
    leftovers = list(
        (CONFIG_ROOT / "environment" / "generated").glob(f"{PREFIX}-bad-batch*")
    )
    check(not leftovers, f"rejected batch launch left configs behind: {leftovers}")

    created = server.api(
        "POST",
        "/api/runs/evaluate",
        {
            **base_body,
            "display_name": f"{PREFIX} eval batch",
            "params": {
                "network_config": _NETWORK,
                "num_episodes": 1,
                "num_steps": num_steps,
                "seeds": [1, 2],
                "seed": 1,
                "deterministic": False,
            },
        },
        expect=201,
    )
    run_id = created["id"]
    _STATE["eval_batch_id"] = run_id
    # Summary is written only at the end of the run — a run created one HTTP
    # roundtrip ago cannot have finished, so this exercises the 404 branch.
    server.api("GET", f"/api/runs/{run_id}/summary", expect=404)

    record = _poll_until_done(server, run_id, 600)
    check(record["status"] == "succeeded", f"batch evaluate ended {record['status']}")
    check(record["progress"] == 1.0, f"progress {record.get('progress')} != 1.0")

    detail = server.api("GET", f"/api/runs/{run_id}")
    check(detail["artifacts"].get("summary") is True, f"artifacts: {detail['artifacts']}")

    summary = server.api("GET", f"/api/runs/{run_id}/summary")
    check(summary["seeds"] == [1, 2], f"summary seeds: {summary['seeds']}")
    check(
        summary["overall"]["total_reward"]["n"] == 2,
        f"overall n: {summary['overall']['total_reward']}",
    )
    check(len(summary["per_seed"]) == 2, f"per_seed: {summary['per_seed']}")

    actions = server.api("GET", f"/api/runs/{run_id}/actions")
    check(actions["episodes"] == [0, 1], f"episodes: {actions['episodes']}")
    meta = server.api("GET", f"/api/runs/{run_id}/viz/meta")
    check(meta["episodes_written"] == [0, 1], f"viz episodes_written: {meta}")

    # Summaries exist only for evaluation runs.
    server.api("GET", f"/api/runs/{_STATE['train_id']}/summary", expect=400)
    return Outcome(
        Status.PASS,
        "2-seed batch via API: validation, progress, summary, actions, viz OK",
    )


def _case_compare_contract() -> Outcome:
    """The exact server contract the /compare page consumes stays intact."""
    server = _server()
    train_id = _STATE["train_id"]

    record = server.api("GET", f"/api/runs/{train_id}")
    check(isinstance(record.get("params"), dict), "run record missing params dict")
    check(bool(record.get("display_name")), "run record missing display_name")

    metrics = server.api("GET", f"/api/runs/{train_id}/metrics")
    all_tags = [tag for tags in metrics["tags"].values() for tag in tags]
    return_tags = [
        tag
        for tag in all_tags
        if tag.startswith("charts/") and tag.endswith("_episodic_return")
    ]
    check(bool(return_tags), f"no charts/*_episodic_return tags: {all_tags}")

    wanted = ",".join(return_tags)
    series = server.api(
        "GET", f"/api/runs/{train_id}/metrics/scalars?tags={wanted}"
    )["series"]
    for tag in return_tags:
        points = series.get(tag)
        check(bool(points), f"scalars for {tag!r} are empty")
        check(
            all(len(point) == 3 for point in points),
            f"scalar points for {tag!r} are not [step, wall_time, value] triples",
        )

    for key in ("eval_id", "eval_batch_id"):
        run_id = _STATE.get(key)
        check(bool(run_id), f"prerequisite run {key} missing (earlier case failed?)")
        eval_record = server.api("GET", f"/api/runs/{run_id}")
        check(isinstance(eval_record.get("params"), dict), f"{key} missing params")
        summary = server.api("GET", f"/api/runs/{run_id}/summary")
        check(
            isinstance(summary["metrics"], list)
            and summary["overall"]["total_reward"]["mean"] is not None,
            f"{key} summary lacks comparable stats: {summary.get('overall')}",
        )
    return Outcome(
        Status.PASS,
        f"compare-view contract OK ({len(return_tags)} return tags, 2 eval summaries)",
    )


def _case_stop_and_orphan() -> Outcome:
    server = _server()
    created = server.api(
        "POST",
        "/api/runs/train",
        {
            "display_name": f"{PREFIX} stop target",
            "base_config": "train_rl_red_agent_vs_rl_blue.yaml",
            "params": {
                **_TINY_TRAIN_PARAMS,
                "total_timesteps": 1_000_000,
                "num_steps": 32,
                "num_saves": 2,
            },
        },
        expect=201,
    )
    run_id = created["id"]
    _STATE["stop_id"] = run_id
    time.sleep(3.0)
    stopped = server.api("POST", f"/api/runs/{run_id}/stop")
    check(stopped["status"] == "stopped", f"stop -> {stopped['status']}")
    pid = stopped["pid"]
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            break
        # A zombie still answers kill(0); only a live process has a cmdline.
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if not f.read():
                    break
        except OSError:
            break
        time.sleep(0.5)
    else:
        raise AssertionError(f"pid {pid} still alive 15s after stop")

    # Orphan detection: fabricate a 'running' record whose pid is dead.
    orphan_id = f"{PREFIX}-orphan-{os.getpid()}"
    _STATE["orphan_id"] = orphan_id
    orphan_dir = DATA_ROOT / "frontend" / "runs" / orphan_id
    orphan_dir.mkdir(parents=True, exist_ok=True)
    with open(orphan_dir / "run.json", "w") as f:
        json.dump(
            {
                "id": orphan_id,
                "kind": "train",
                "display_name": orphan_id,
                "status": "running",
                "pid": 2**22 + 7,  # beyond default pid_max: never a live pid
                "created_at": "2026-01-01T00:00:00+00:00",
                "experiment_name": orphan_id,
                "params": {},
            },
            f,
        )
    record = server.api("GET", f"/api/runs/{orphan_id}")
    check(record["status"] == "orphaned", f"dead-pid run reported {record['status']}")
    return Outcome(Status.PASS, "stop reaps the process group; dead runs report orphaned")


def _case_validation() -> Outcome:
    server = _server()
    body = server.api(
        "POST",
        "/api/runs/train",
        {
            "display_name": f"{PREFIX} bad",
            "base_config": "train_rl_red_agent_vs_rl_blue.yaml",
            "params": {**_TINY_TRAIN_PARAMS, "network_config": ["no-such-net.yaml"]},
        },
        expect=400,
    )
    check("no-such-net" in body["error"]["message"], f"unhelpful error: {body}")
    server.api(
        "POST",
        "/api/runs/evaluate",
        {
            "display_name": f"{PREFIX} bad eval",
            "base_config": "evaluate_rl_red_vs_rl_blue.yaml",
            "source": {"experiment_name": "no-such-experiment"},
            "params": {},
        },
        expect=404,
    )
    server.api("GET", "/api/runs/no-such-run", expect=404)
    leftovers = list((CONFIG_ROOT / "environment" / "generated").glob(f"{PREFIX}-bad*"))
    check(not leftovers, f"rejected launch left generated configs behind: {leftovers}")
    return Outcome(Status.PASS, "launch validation rejects bad refs with 4xx")


def _case_cleanup() -> Outcome:
    server = _server()
    deleted = []
    for key in ("eval_id", "eval_batch_id", "stop_id", "orphan_id", "train_id"):
        run_id = _STATE.get(key)
        if not run_id:
            continue
        server.api("DELETE", f"/api/runs/{run_id}?artifacts=true")
        server.api("GET", f"/api/runs/{run_id}", expect=404)
        deleted.append(run_id)
    train_id = _STATE.get("train_id")
    if train_id:
        check(
            not (DATA_ROOT / "models" / train_id).exists(),
            "DELETE ?artifacts=true left the models dir behind",
        )
    eval_batch_id = _STATE.get("eval_batch_id")
    if eval_batch_id:
        check(
            not (DATA_ROOT / "action_logs" / f"{eval_batch_id}.summary.json").exists(),
            "DELETE ?artifacts=true left the summary JSON behind",
        )
    remaining = [
        r["id"]
        for r in server.api("GET", "/api/runs")["runs"]
        if r["id"].startswith(f"{PREFIX}-")
    ]
    check(not remaining, f"suite runs left in registry: {remaining}")
    return Outcome(Status.PASS, f"deleted {len(deleted)} runs + artifacts via API")


def register(registry: Registry, ctx: Context) -> None:
    cases = [
        ("frontend:options_api", _case_options, 120.0, None),
        ("frontend:static_serving", _case_static, 120.0, None),
        ("frontend:train_e2e_via_api", _case_train, 900.0, None),
        ("frontend:evaluate_e2e_via_api", _case_evaluate, 600.0, "frontend:train_e2e_via_api"),
        ("frontend:evaluate_batch_via_api", _case_evaluate_batch, 600.0, "frontend:train_e2e_via_api"),
        ("frontend:compare_view_contract", _case_compare_contract, 120.0, "frontend:evaluate_batch_via_api"),
        ("frontend:job_stop_and_orphan", _case_stop_and_orphan, 300.0, None),
        ("frontend:launch_validation", _case_validation, 120.0, None),
        ("frontend:cleanup_via_api", _case_cleanup, 300.0, None),
    ]
    for name, fn, timeout_s, depends_on in cases:
        registry.add(
            TestCase(
                name=name,
                suite=SUITE,
                fn=fn,
                timeout_s=timeout_s,
                depends_on=depends_on,
            )
        )
