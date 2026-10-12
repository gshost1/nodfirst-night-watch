"""Judge-visible dashboard features: transitions, model chip, egress memory, economics cards."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from app.app import create_app
from inference.metrics import MockMetricsSource
from tests.test_app import FakeClock, _client, _settings

DASHBOARD = Path(__file__).resolve().parents[1] / "app" / "dashboard.html"


def test_metrics_transitions_are_newest_first_with_reasons() -> None:
    clock = FakeClock()
    app = create_app(settings=_settings(), metrics_source=MockMetricsSource(clock=clock))

    async def scenario() -> dict:
        async with _client(app) as client:
            await client.post("/api/simulate-load", json={"duration_s": 12, "intensity": 1.5})
            payload: dict = {}
            for _ in range(60):
                clock.advance(0.5)
                payload = (await client.get("/api/metrics")).json()
            return payload

    payload = asyncio.run(scenario())
    transitions = payload["transitions"]
    assert 1 <= len(transitions) <= 8
    for t in transitions:
        assert set(t) == {"at", "from", "to", "reason"}
        assert t["from"] != t["to"] and t["reason"]
    assert [t["at"] for t in transitions] == sorted((t["at"] for t in transitions), reverse=True)


def test_status_has_model_and_egress_remembers_last_blocked_host() -> None:
    app = create_app(settings=_settings())

    async def scenario() -> tuple[dict, dict]:
        async with _client(app) as client:
            before = (await client.get("/api/status")).json()
            await client.post("/api/egress-demo")
            after = (await client.get("/api/status")).json()
            return before, after

    before, after = asyncio.run(scenario())
    assert before["model"]
    assert before["privacy"]["last_blocked_host"] is None
    assert after["privacy"]["blocked_attempts"] == 1
    assert after["privacy"]["last_blocked_host"]


def test_dashboard_has_new_cards_and_relative_fetches() -> None:
    html = DASHBOARD.read_text()
    for marker in ("timeline-card", "econ-card", "model-chip", "$0 per night"):
        assert marker in html
    assert not re.search(r"""getJSON\(["']/""", html)
