"""Fast lane: safety-net alerts are raised before the first LLM call, then enriched in place."""

from __future__ import annotations

import os
from typing import Any

import pytest

from config import Settings
from core.llm import LLMClient, LLMResult
from nightwatch.classify import fast_lane_classification
from nightwatch.dataset import generate_events
from nightwatch.pipeline import NightWatchRun


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_BASE_URL", "mock")
    monkeypatch.setenv("VLLM_METRICS_URL", "mock")
    monkeypatch.setenv("PHI_MODE", "true")
    monkeypatch.setenv("MOCK_FAIL_RUN", "0")
    return Settings.from_env(os.environ)


class ProbeClient(LLMClient):
    """Records the alerts visible at the first LLM call and the call's offset."""

    def __init__(self) -> None:
        super().__init__()
        self.run: NightWatchRun | None = None
        self.first_call_alerts: list[dict[str, Any]] | None = None

    def complete(self, messages: Any, **kwargs: Any) -> LLMResult:
        if self.first_call_alerts is None and self.run is not None:
            self.first_call_alerts = self.run.alerts()
        return super().complete(messages, **kwargs)


def _run(monkeypatch: pytest.MonkeyPatch) -> tuple[NightWatchRun, ProbeClient]:
    client = ProbeClient()
    run = NightWatchRun(client=client, settings=_settings(monkeypatch))
    client.run = run
    run.execute()
    return run, client


def test_fall_alert_first_without_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = _run(monkeypatch)
    alerts = run.alerts()
    ids = [alert["id"] for alert in alerts]
    assert len(ids) == len(set(ids))
    assert alerts[0]["time"] == "03:12"
    assert "fall" in (alerts[0]["kind"] + alerts[0]["text"]).lower()
    assert [alert["seq"] for alert in alerts] == list(range(len(alerts)))


def test_alerts_exist_before_first_llm_call_and_are_enriched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, client = _run(monkeypatch)
    early = client.first_call_alerts
    assert early, "fast-lane alerts must be published before the first LLM call"
    assert early[0]["time"] == "03:12"
    assert early[0]["rationale"].startswith("fast lane: safety net matched")
    assert early[0]["summary"] == early[0]["text"]

    final = {alert["id"]: alert for alert in run.alerts()}
    fall = run.alerts()[0]
    assert fall["id"] == early[0]["id"]
    assert fall["fast_lane"] is True
    assert isinstance(fall["raised_ms"], int) and fall["raised_ms"] >= 0
    assert not fall["rationale"].startswith("fast lane:")  # enriched by the model pass
    assert set(alert["id"] for alert in early) <= set(final)
    assert all(alert["fast_lane"] for alert in early)


def test_fast_lane_matches_safety_net_only() -> None:
    events = generate_events()
    flagged = [event for event in events if fast_lane_classification(event) is not None]
    assert flagged
    assert all(fast_lane_classification(event).urgency == "urgent" for event in flagged)  # type: ignore[union-attr]
