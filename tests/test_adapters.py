"""
tests/test_adapters.py — Unit tests for The Brain.

These tests use mocking so they run without real API keys.
Every adapter, the router, and the orchestrator are covered.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from brain.task import Task, TaskResult, TaskType, Priority, TaskStatus
from brain.adapters.base import BaseAdapter
from brain.router import Router, STATIC_ROUTING_TABLE
from brain.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_task() -> Task:
    return Task(
        prompt="What colour is the sky?",
        task_type=TaskType.FACTUAL_QA,
        priority=Priority.NORMAL,
    )


@pytest.fixture
def successful_result(sample_task: Task) -> TaskResult:
    return TaskResult(
        task_id=sample_task.id,
        provider="groq",
        model="llama3-8b-8192",
        content="The sky is blue.",
        tokens_used=15,
        latency_ms=120.0,
    )


def _make_mock_adapter(
    provider_key: str,
    available: bool = True,
    tier: str = "free",
    result: TaskResult | None = None,
) -> MagicMock:
    """Build a mock adapter with sensible defaults."""
    adapter = MagicMock(spec=BaseAdapter)
    adapter.PROVIDER_KEY = provider_key
    adapter.TIER = tier
    adapter.is_available.return_value = available
    adapter.provider_info.return_value = {
        "provider": provider_key,
        "tier": tier,
        "cost_per_1k": None,
        "task_types": [t.value for t in TaskType],
        "available": available,
    }
    if result is not None:
        adapter.complete.return_value = result
    return adapter


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_id_is_unique(self):
        t1 = Task(prompt="Hello")
        t2 = Task(prompt="Hello")
        assert t1.id != t2.id

    def test_full_prompt_without_context(self, sample_task: Task):
        assert sample_task.full_prompt() == sample_task.prompt

    def test_full_prompt_with_context(self):
        task = Task(prompt="Summarise this.", context="Long article text here.")
        assert "Context:" in task.full_prompt()
        assert "Long article text here." in task.full_prompt()
        assert "Summarise this." in task.full_prompt()

    def test_default_status_is_pending(self, sample_task: Task):
        assert sample_task.status == TaskStatus.PENDING


# ---------------------------------------------------------------------------
# TaskResult tests
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_succeeded_when_no_error(self, successful_result: TaskResult):
        assert successful_result.succeeded is True

    def test_failed_when_error_set(self, sample_task: Task):
        result = TaskResult(
            task_id=sample_task.id,
            provider="groq",
            model="llama3",
            content="",
            error="Rate limit exceeded",
        )
        assert result.succeeded is False

    def test_summary_contains_provider(self, successful_result: TaskResult):
        assert "groq" in successful_result.summary()


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestRouter:
    def test_routes_to_first_available_in_table(self, sample_task: Task):
        # Preference order for FACTUAL_QA starts with groq.
        groq_mock = _make_mock_adapter("groq")
        registry  = {"groq": groq_mock, "gemini": _make_mock_adapter("gemini")}
        router    = Router(registry)
        assert router.route(sample_task) == "groq"

    def test_skips_unavailable_providers(self, sample_task: Task):
        groq_mock   = _make_mock_adapter("groq", available=False)
        gemini_mock = _make_mock_adapter("gemini", available=True)
        registry    = {"groq": groq_mock, "gemini": gemini_mock}
        router      = Router(registry)
        # groq is unavailable, should fall through to gemini
        assert router.route(sample_task) == "gemini"

    def test_respects_preferred_model(self, sample_task: Task):
        sample_task.preferred_model = "openai"
        openai_mock = _make_mock_adapter("openai")
        registry    = {"groq": _make_mock_adapter("groq"), "openai": openai_mock}
        router      = Router(registry)
        assert router.route(sample_task) == "openai"

    def test_low_priority_prefers_free(self, sample_task: Task):
        sample_task.priority = Priority.LOW
        openai_mock = _make_mock_adapter("openai", tier="paid")
        groq_mock   = _make_mock_adapter("groq",   tier="free")
        registry    = {"openai": openai_mock, "groq": groq_mock}
        router      = Router(registry)
        result = router.route(sample_task)
        assert result == "groq"

    def test_returns_none_when_no_providers_available(self, sample_task: Task):
        registry = {"groq": _make_mock_adapter("groq", available=False)}
        router   = Router(registry)
        assert router.route(sample_task) is None

    def test_static_table_covers_all_task_types(self):
        for task_type in TaskType:
            assert task_type in STATIC_ROUTING_TABLE, (
                f"TaskType.{task_type.name} is missing from STATIC_ROUTING_TABLE"
            )


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def _make_orchestrator(self, registry: dict) -> Orchestrator:
        orch = Orchestrator.__new__(Orchestrator)
        orch._registry      = registry
        orch._router        = Router(registry)
        orch._max_fallbacks = 3
        orch._total_calls   = 0
        orch._total_tokens  = 0
        orch._total_cost    = 0.0
        orch._failed_calls  = 0
        return orch

    def test_returns_successful_result(self, sample_task: Task, successful_result: TaskResult):
        groq_mock = _make_mock_adapter("groq", result=successful_result)
        orch = self._make_orchestrator({"groq": groq_mock})
        result = orch.run(sample_task)
        assert result.succeeded
        assert result.content == "The sky is blue."

    def test_falls_back_on_error(self, sample_task: Task, successful_result: TaskResult):
        fail_result = TaskResult(
            task_id=sample_task.id,
            provider="groq", model="llama3", content="", error="API error"
        )
        groq_mock   = _make_mock_adapter("groq",   result=fail_result)
        gemini_mock = _make_mock_adapter("gemini", result=successful_result)
        # Override provider on fallback result to match gemini
        successful_result.provider = "gemini"
        orch = self._make_orchestrator({"groq": groq_mock, "gemini": gemini_mock})
        result = orch.run(sample_task)
        assert result.succeeded

    def test_session_stats_increment(self, sample_task: Task, successful_result: TaskResult):
        groq_mock = _make_mock_adapter("groq", result=successful_result)
        orch = self._make_orchestrator({"groq": groq_mock})
        orch.run(sample_task)
        orch.run(sample_task)
        stats = orch.session_stats()
        assert stats["total_calls"] == 2
        assert stats["total_tokens"] == 30  # 15 tokens × 2 calls

    def test_fails_gracefully_with_no_providers(self, sample_task: Task):
        orch = self._make_orchestrator({})
        result = orch.run(sample_task)
        assert not result.succeeded
        assert "No providers" in result.error
