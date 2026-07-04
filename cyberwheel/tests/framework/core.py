"""Core primitives for the Cyberwheel test framework.

A small, dependency-free (no pytest) harness with three suites:

- ``config``: every YAML under ``cyberwheel/data/configs`` loads and its
  code-consumed references resolve.
- ``smoke``:  the environment, training, and evaluation work end-to-end at
  tiny scale, through the real CLI.
- ``perf``:   fixed benchmarks whose results gate against the parent
  commit's recorded results (see ``baseline.py`` for the convention).

Run with ``python -m cyberwheel.tests`` (see ``cli.py`` for flags).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "cyberwheel" / "data" / "configs"
DATA_ROOT = REPO_ROOT / "cyberwheel" / "data"
BASELINE_RELPATH = "cyberwheel/tests/baselines/baseline.json"

# Keep every test hermetic and quiet: no W&B traffic, no display backends,
# no tqdm progress bars.
SAFE_ENV = {
    "WANDB_MODE": "disabled",
    "WANDB_SILENT": "true",
    "MPLBACKEND": "Agg",
    "TQDM_DISABLE": "1",
}


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"
    XFAIL = "XFAIL"  # known issue failed as documented (never gates)
    XPASS_WARN = "XPASS_WARN"  # known issue unexpectedly passed (never gates)
    INFO = "INFO"  # passed, with notes worth reading (never gates)


GATING_STATUSES = frozenset({Status.FAIL, Status.ERROR})


class TestFailure(Exception):
    """An assertion made by a test case failed."""


class TestSkip(Exception):
    """A test case declined to run."""


@dataclass
class Outcome:
    """Optional rich return value for a test function (None means PASS)."""

    status: Status
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class TestCase:
    name: str
    suite: str
    fn: Callable[[], Optional[Outcome]]
    timeout_s: float = 300.0
    depends_on: Optional[str] = None
    quick_skip: bool = False
    known_issue: Optional[str] = None
    # If set, the case is skipped with this reason unless --filter matches it.
    default_skip_reason: Optional[str] = None


@dataclass
class TestResult:
    name: str
    suite: str
    status: Status
    duration_s: float = 0.0
    message: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "suite": self.suite,
            "status": self.status.value,
            "duration_s": round(self.duration_s, 3),
            "message": self.message,
            "details": self.details,
        }


class Registry:
    def __init__(self) -> None:
        self.cases: list[TestCase] = []

    def add(self, case: TestCase) -> None:
        if any(c.name == case.name for c in self.cases):
            raise ValueError(f"duplicate test case name: {case.name}")
        self.cases.append(case)


@dataclass
class Context:
    """Options and shared state passed to every suite at registration time."""

    run_id: str
    quick: bool = False
    filter: Optional[str] = None
    verbose: bool = False
    keep_artifacts: bool = False
    # Filled in by perf cases; consumed by the baseline comparison in cli.py.
    perf_metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------


def check(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


def check_eq(actual, expected, label: str) -> None:
    if actual != expected:
        raise TestFailure(f"{label}: expected {expected!r}, got {actual!r}")


def check_file(path: Path, min_size: int = 1) -> None:
    if not path.is_file():
        raise TestFailure(f"expected file does not exist: {path}")
    size = path.stat().st_size
    if size < min_size:
        raise TestFailure(f"file too small ({size} < {min_size} bytes): {path}")


# ---------------------------------------------------------------------------
# subprocess helper
# ---------------------------------------------------------------------------


def run_cli(
    argv: list[str],
    timeout: float,
    env_extra: Optional[dict] = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run ``python <argv...>`` with W&B/network-safe environment vars."""
    env = os.environ.copy()
    env.update(SAFE_ENV)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(cwd or REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_cases(
    cases: list[TestCase],
    quick: bool = False,
    fltr: Optional[str] = None,
    on_result: Optional[Callable[[TestResult], None]] = None,
) -> list[TestResult]:
    results: list[TestResult] = []
    status_by_name: dict[str, Status] = {}

    for case in cases:
        if fltr and fltr not in case.name:
            continue

        if case.default_skip_reason and not (fltr and fltr in case.name):
            result = TestResult(
                case.name, case.suite, Status.SKIP, 0.0, case.default_skip_reason
            )
        elif quick and case.quick_skip:
            result = TestResult(
                case.name, case.suite, Status.SKIP, 0.0, "skipped (--quick)"
            )
        elif case.depends_on and status_by_name.get(case.depends_on) not in (
            Status.PASS,
            Status.INFO,
        ):
            result = TestResult(
                case.name,
                case.suite,
                Status.SKIP,
                0.0,
                f"skipped (dependency {case.depends_on!r} did not pass)",
            )
        else:
            start = time.perf_counter()
            try:
                outcome = case.fn()
                duration = time.perf_counter() - start
                if outcome is None:
                    result = TestResult(case.name, case.suite, Status.PASS, duration)
                else:
                    result = TestResult(
                        case.name,
                        case.suite,
                        outcome.status,
                        duration,
                        outcome.message,
                        outcome.details,
                    )
            except TestFailure as exc:
                duration = time.perf_counter() - start
                if case.known_issue:
                    result = TestResult(
                        case.name,
                        case.suite,
                        Status.XFAIL,
                        duration,
                        f"failed as documented: {case.known_issue} ({exc})",
                    )
                else:
                    result = TestResult(
                        case.name, case.suite, Status.FAIL, duration, str(exc)
                    )
            except TestSkip as exc:
                duration = time.perf_counter() - start
                result = TestResult(
                    case.name, case.suite, Status.SKIP, duration, str(exc)
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.perf_counter() - start
                result = TestResult(
                    case.name,
                    case.suite,
                    Status.FAIL,
                    duration,
                    f"timed out after {exc.timeout:.0f}s",
                )
            except Exception:
                duration = time.perf_counter() - start
                tb_tail = "".join(
                    traceback.format_exc().splitlines(keepends=True)[-12:]
                )
                if case.known_issue:
                    result = TestResult(
                        case.name,
                        case.suite,
                        Status.XFAIL,
                        duration,
                        f"failed as documented: {case.known_issue}",
                        {"traceback_tail": tb_tail},
                    )
                else:
                    result = TestResult(
                        case.name,
                        case.suite,
                        Status.ERROR,
                        duration,
                        (
                            tb_tail.strip().splitlines()[-1]
                            if tb_tail.strip()
                            else "unknown error"
                        ),
                        {"traceback_tail": tb_tail},
                    )

        # A known-issue case that PASSes is a warning: the issue may be fixed.
        if case.known_issue and result.status is Status.PASS:
            result.status = Status.XPASS_WARN
            result.message = (
                f"known issue did NOT reproduce ({case.known_issue}) — "
                "consider promoting this to a real test"
            )

        status_by_name[case.name] = result.status
        results.append(result)
        if on_result:
            on_result(result)

    return results
