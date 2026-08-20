"""Accumulating wall-time profiler for the per-molecule loops.

The run scripts (`QM7_test/run*.bash`) already time each *stage* of the pipeline. This module
gives the missing resolution one level down: which operation *inside* a stage spends the time.
It is opt-in (`--profile`) and, when disabled, costs one attribute lookup per step.
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger('__main__')


class StepTimer:
    """Accumulate wall time per named step and log a sorted breakdown.

    Steps are entered through the `step()` context manager and may be re-entered any number of
    times (once per molecule, typically); the report gives the accumulated total, the per-call
    average and the share of the profiled total.
    """

    def __init__(self, *, enabled=True):
        """Initialize the timer.

        Args:
            enabled (bool): Whether to actually measure. A disabled timer is a no-op, so callers
                can be instrumented unconditionally.
        """
        self.enabled = enabled
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    @contextmanager
    def step(self, name):
        """Time the enclosed block and accumulate it under `name`.

        Args:
            name (str): Step label, shown verbatim in the report.

        Yields:
            None: The enclosed block runs inside the timed section.
        """
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.total[name] += time.perf_counter() - t0
            self.count[name] += 1

    def report(self, header='', *, nunits=None, unit='molecule'):
        """Log the accumulated breakdown, slowest step first.

        Args:
            header (str): Line printed above the table.
            nunits (int | None): Number of processed units (molecules); when given, an extra
                per-unit column is printed.
            unit (str): Name of the unit for the per-unit column header.
        """
        if not self.enabled or not self.total:
            return
        grand = sum(self.total.values())
        width = max(len(name) for name in self.total) + 2
        lines = [f'[profile] {header}' if header else '[profile] wall-time breakdown',
                 f'[profile] {"step":<{width}s} {"calls":>7s} {"total_s":>10s} {"s/call":>10s} '
                 f'{f"s/{unit}":>12s} {"%":>7s}']
        for name, dt in sorted(self.total.items(), key=lambda kv: -kv[1]):
            n = self.count[name]
            # a per-unit average is only meaningful for steps that run once per unit: a one-off
            # setup step amortised over the units would read as if it were part of the loop.
            per_unit = f'{dt/nunits:12.4f}' if nunits and n==nunits else f'{"":12s}'
            lines.append(f'[profile] {name:<{width}s} {n:7d} {dt:10.3f} {dt/n:10.4f} {per_unit} '
                         f'{100*dt/grand:7.1f}')
        lines.append(f'[profile] {"TOTAL profiled":<{width}s} {"":7s} {grand:10.3f}')
        logger.info('\n'.join(lines))
