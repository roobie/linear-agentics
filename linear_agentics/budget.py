"""Budget tracking — step and time limits for agent execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


class BudgetExhaustedError(Exception):
    pass


class BudgetTimeoutError(Exception):
    pass


class Budget:
    """Tracks step count and wall-clock time for an agent run."""

    def __init__(self, max_steps: int, timeout_minutes: float) -> None:
        self.max_steps = max_steps
        self.timeout_minutes = timeout_minutes
        self._steps_used = 0
        self._start: datetime | None = None

    def start(self) -> None:
        self._start = datetime.now(timezone.utc)

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self._steps_used)

    @property
    def steps_used(self) -> int:
        return self._steps_used

    @property
    def elapsed(self) -> timedelta:
        if self._start is None:
            return timedelta(0)
        return datetime.now(timezone.utc) - self._start

    def spend(self, cost: int = 1) -> None:
        self._steps_used += cost
        if self._steps_used > self.max_steps:
            raise BudgetExhaustedError(f"Budget exhausted: {self._steps_used}/{self.max_steps} steps used")

    def check_timeout(self) -> None:
        if self._start is None:
            return
        if self.elapsed > timedelta(minutes=self.timeout_minutes):
            raise BudgetTimeoutError(
                f"Budget timeout: {self.elapsed.total_seconds():.0f}s elapsed, limit is {self.timeout_minutes * 60:.0f}s"
            )

    def check(self) -> None:
        """Check both step budget and timeout."""
        if self._steps_used >= self.max_steps:
            raise BudgetExhaustedError(f"Budget exhausted: {self._steps_used}/{self.max_steps} steps used")
        self.check_timeout()
