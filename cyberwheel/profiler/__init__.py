"""Reusable performance profiler for the Cyberwheel environment.

Run with ``python3 -m cyberwheel.profiler``. See ``cli.py`` for options and
``baselines/`` for the committed per-phase baseline that ``--check`` gates
against (same-machine convention; re-record with ``--record-baseline`` in the
same commit as any intentional perf change, mirroring the test framework's
``cyberwheel/tests/baselines/baseline.json``).
"""

from cyberwheel.profiler.phase_timer import PhaseAccumulator, MethodInstrumenter

__all__ = ["PhaseAccumulator", "MethodInstrumenter"]
