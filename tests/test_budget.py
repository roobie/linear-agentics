"""Tests for budget tracking."""

import time

import pytest

from linear_agentics.budget import Budget, BudgetExhaustedError, BudgetTimeoutError


class TestBudget:
    def test_initial_state(self):
        b = Budget(max_steps=10, timeout_minutes=5)
        assert b.remaining == 10
        assert b.steps_used == 0

    def test_spend_reduces_remaining(self):
        b = Budget(max_steps=5, timeout_minutes=5)
        b.spend(1)
        assert b.remaining == 4
        assert b.steps_used == 1

    def test_spend_multiple(self):
        b = Budget(max_steps=10, timeout_minutes=5)
        b.spend(3)
        assert b.remaining == 7

    def test_exhaust_raises(self):
        b = Budget(max_steps=2, timeout_minutes=5)
        b.spend(1)
        b.spend(1)
        with pytest.raises(BudgetExhaustedError):
            b.spend(1)

    def test_check_raises_when_exhausted(self):
        b = Budget(max_steps=1, timeout_minutes=5)
        b.spend(1)
        with pytest.raises(BudgetExhaustedError):
            b.check()

    def test_timeout_check(self):
        b = Budget(max_steps=100, timeout_minutes=0.0001)  # ~6ms
        b.start()
        time.sleep(0.01)
        with pytest.raises(BudgetTimeoutError):
            b.check_timeout()

    def test_no_timeout_before_start(self):
        b = Budget(max_steps=100, timeout_minutes=0.0001)
        # Should not raise before start()
        b.check_timeout()

    def test_elapsed_before_start(self):
        b = Budget(max_steps=10, timeout_minutes=5)
        assert b.elapsed.total_seconds() == 0
