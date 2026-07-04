"""Phase-level wall-clock accounting with nesting support.

``PhaseAccumulator`` keeps a stack so that a phase's *exclusive* time excludes
any nested phases (e.g. ``red.observation`` timed inside ``red.act`` is not
double-counted). ``MethodInstrumenter`` wraps bound methods (or class
functions) so existing env objects can be timed without editing core code;
``restore()`` puts every original back.
"""

from __future__ import annotations

import time
from typing import Any


class PhaseAccumulator:
    def __init__(self) -> None:
        self._stack: list[list] = []  # [name, start_ns, child_ns]
        self.phases: dict[str, dict[str, int]] = {}

    def push(self, name: str) -> None:
        self._stack.append([name, time.perf_counter_ns(), 0])

    def pop(self) -> int:
        name, start, child_ns = self._stack.pop()
        elapsed = time.perf_counter_ns() - start
        record = self.phases.setdefault(
            name, {"inclusive_ns": 0, "exclusive_ns": 0, "calls": 0}
        )
        record["inclusive_ns"] += elapsed
        record["exclusive_ns"] += elapsed - child_ns
        record["calls"] += 1
        if self._stack:
            self._stack[-1][2] += elapsed
        return elapsed

    def rows(self, denominator: int, anchor: str) -> list[dict]:
        """Per-phase stats normalized by ``denominator`` (e.g. steps).

        ``anchor`` names the phase whose inclusive time defines 100%%
        (typically the driver's whole-step phase).
        """
        anchor_ns = self.phases.get(anchor, {}).get("inclusive_ns", 0)
        rows = []
        for name in sorted(self.phases):
            record = self.phases[name]
            rows.append(
                {
                    "phase": name,
                    "inclusive_ms_per_unit": record["inclusive_ns"] / denominator / 1e6,
                    "exclusive_ms_per_unit": record["exclusive_ns"] / denominator / 1e6,
                    "calls_per_unit": record["calls"] / denominator,
                    "pct_of_anchor": (
                        100.0 * record["inclusive_ns"] / anchor_ns
                        if anchor_ns
                        else None
                    ),
                }
            )
        return rows


class MethodInstrumenter:
    """Wraps methods so each call is timed as a phase in an accumulator."""

    def __init__(self, accumulator: PhaseAccumulator) -> None:
        self.accumulator = accumulator
        self._originals: list[tuple[Any, str, Any]] = []
        self._wrapped_functions: set[int] = set()

    def wrap(self, obj: Any, attr: str, phase: str) -> None:
        """Time ``obj.<attr>`` as ``phase``. Works on instances and classes.

        For a class, wraps the method on the class in its MRO that actually
        defines it, and never wraps the same underlying function twice (so
        subclasses sharing an inherited method are instrumented once).
        """
        if isinstance(obj, type):
            owner = next(cls for cls in obj.__mro__ if attr in cls.__dict__)
            original = owner.__dict__[attr]
            if id(original) in self._wrapped_functions:
                return
            self._wrapped_functions.add(id(original))
            obj = owner
            had_own_attr = True
        else:
            original = getattr(obj, attr)
            had_own_attr = attr in obj.__dict__
        accumulator = self.accumulator

        def timed(*args, **kwargs):
            accumulator.push(phase)
            try:
                return original(*args, **kwargs)
            finally:
                accumulator.pop()

        timed.__profiler_original__ = original
        self._originals.append((obj, attr, original, had_own_attr))
        setattr(obj, attr, timed)

    def wrap_if_present(self, obj: Any, attr: str, phase: str) -> None:
        if obj is not None and hasattr(obj, attr):
            self.wrap(obj, attr, phase)

    def restore(self) -> None:
        for obj, attr, original, had_own_attr in reversed(self._originals):
            if had_own_attr:
                setattr(obj, attr, original)
            else:
                # The wrap shadowed a class attribute on an instance; removing
                # the shadow restores normal method lookup.
                delattr(obj, attr)
        self._originals.clear()
        self._wrapped_functions.clear()
