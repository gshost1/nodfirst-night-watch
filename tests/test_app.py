"""Integration tests for the Night Watch HTTP API (`app/app.py`).

Everything runs in-process over `httpx.ASGITransport` in mock mode — no ports,
no sockets, no outbound inference. Covers VAL-API-001..007 and VAL-LOAD-001:

- VAL-API-001 the status payload's documented shape + mock flags
- VAL-API-002 the metrics endpoint's latest snapshot plus growing history
- VAL-API-003 the exclusive `since` cursor on `/api/events`
- VAL-API-004 a second start while a run is active returns 409
- VAL-API-005 a new run resets the prior run's events/alerts/calls/digest
- VAL-API-006 the egress demo blocks pre-I/O (wrapped client never reached)
- VAL-API-007 a background failure is reported without killing the app
- VAL-LOAD-001 `POST /api/simulate-load` walks the policy NORMAL→CONSERVE→SURGE→NORMAL
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import socket
import time
from typing import Any

import httpx
import pytest

from app.app import create_app
from config import Settings
from core.llm import LLMClient
from inference.metrics import MockMetricsSource

#: sha256 of the vendored Chart.js 4.5.1 UMD build (see architecture.md §app/dashboard.html).
CHART_UMD_SHA256 = "ecc3cd1eeb8c34d2178e3f59fd63ec5a3d84358c11730af0b9958dc886d7652a"

#: Every element the dashboard JavaScript binds to (and the browser flow asserts on).
DASHBOARD_ELEMENT_IDS = (
    "btn-run",
    "btn-load",
    "btn-egress",
    "chart-kv",
    "chart-waiting",
    "chart-tokens",
    "mode-badge",
    "mode-reason",
    "mode-params",
    "privacy-badge",
    "failure-banner",
    "alerts-feed",
    "events-feed",
    "digest-panel",
    "call-log",
    "egress-result",
)


def _settings(**overrides: str) -> Settings:
    """Mock-mode settings built from an explicit environment mapping (no `.env` reads)."""
    env = {
        "LLM_BASE_URL": "mock",
        "VLLM_METRICS_URL": "mock",
        "PHI_MODE": "true",
        "MOCK_FAIL_RUN": "0",
        "EGRESS_ALLOWLIST": "localhost,127.0.0.1",
    }
    env.update(overrides)
    return Settings.from_env(env)


class FakeClock:
    """Deterministic stand-in for `time.time`, so the mock curve can be advanced instantly."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def _wait_for_run_state(
    client: httpx.AsyncClient, wanted: str, timeout: float = 15.0
) -> dict:
    """Poll `/api/status` until the run reaches `wanted` (or fail loudly)."""
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = (await client.get("/api/status")).json()
        if payload["run_state"] == wanted:
            return payload
        await asyncio.sleep(0.02)
    raise AssertionError(f"run never reached {wanted!r}; last payload: {payload}")


def test_status_exposes_the_documented_operating_state() -> None:
    app = create_app(settings=_settings())

    async def scenario() -> dict:
        async with _client(app) as client:
            return (await client.get("/api/status")).json()

    payload = asyncio.run(scenario())

    assert set(payload) == {
        "run_state",
        "running",
        "mode",
        "reason",
        "params",
        "egress",
        "flags",
        "privacy",
        "events_total",
        "events_processed",
        "alerts_count",
        "digest_ready",
    }
    assert payload["run_state"] == "idle"
    assert payload["running"] is False
    assert payload["mode"] in {"NORMAL", "CONSERVE", "SURGE"}
    assert payload["reason"]
    assert set(payload["params"]) == {
        "max_tokens",
        "context_items",
        "batch_size",
        "enable_thinking",
    }
    assert payload["params"]["max_tokens"] in {512, 256, 128}
    assert payload["flags"] == {"mock_llm": True, "mock_metrics": True, "phi_mode": True}
    assert payload["egress"] == {
        "endpoint_host": "mock",
        "is_mock": True,
        "allowlist": ["127.0.0.1", "localhost"],
        "phi_mode": True,
        "owned": False,
    }
    assert payload["privacy"] == {
        "phi_calls_on_host": 0,
        "blocked_attempts": 0,
        "bytes_egressed": 0,
        "host": "mock",
    }
    assert payload["events_total"] == 0
    assert payload["events_processed"] == 0
    assert payload["alerts_count"] == 0
    assert payload["digest_ready"] is False


def test_status_tracks_a_completed_run_and_flags_an_error_only_on_failure() -> None:
    app = create_app(settings=_settings())

    async def scenario() -> tuple[dict, list, list]:
        async with _client(app) as client:
            assert (await client.post("/api/run")).status_code == 200
            done = await _wait_for_run_state(client, "done")
            alerts = (await client.get("/api/alerts")).json()
            calls = (await client.get("/api/calls")).json()
            return done, alerts, calls

    done, alerts, calls = asyncio.run(scenario())

    assert done["run_state"] == "done"
    assert done["running"] is False
    assert "error" not in done
    assert done["events_total"] == 40
    assert done["events_processed"] == 40
    assert done["alerts_count"] == 3
    assert done["digest_ready"] is True
    assert done["params"]["max_tokens"] in {512, 256, 128}
    # Alerts are the emitted urgent ones, severity-first: the 03:12 possible fall leads.
    assert [alert["time"] for alert in alerts] == ["03:12", "23:39", "21:15"]
    assert "thud" in alerts[0]["text"].lower()
    assert calls and all(
        {"step", "mode", "latency_ms", "prompt_tokens", "completion_tokens", "ok"} <= set(call)
        for call in calls
    )


def test_metrics_endpoint_returns_latest_snapshot_and_growing_history() -> None:
    app = create_app(settings=_settings())

    async def scenario() -> tuple[dict, dict]:
        async with _client(app) as client:
            first = (await client.get("/api/metrics")).json()
            await asyncio.sleep(0.3)  # the mock curve samples at most every 0.25s
            second = (await client.get("/api/metrics")).json()
            return first, second

    first, second = asyncio.run(scenario())

    assert first["latest"]["source"] == "mock"
    assert first["latest"]["degraded"] is False
    assert first["latest"]["kv_cache_pct"] is not None
    assert first["latest"]["waiting"] is not None
    assert first["latest"]["tokens_per_s"] is not None
    assert len(first["history"]) >= 1
    assert len(second["history"]) > len(first["history"])
    assert all(
        {"waiting", "running", "kv_cache_pct", "ttft_p50", "tokens_per_s", "mode"} <= set(sample)
        for sample in second["history"]
    )
    assert all(sample["mode"] in {"NORMAL", "CONSERVE", "SURGE"} for sample in second["history"])


def test_events_endpoint_honors_the_exclusive_since_cursor() -> None:
    app = create_app(settings=_settings(), step_delay_s=0.02)

    async def scenario() -> tuple[list[dict], dict, dict, dict]:
        async with _client(app) as client:
            assert (await client.post("/api/run")).status_code == 200
            seen: list[dict] = []
            cursor = 0
            while True:
                page = (await client.get("/api/events", params={"since": cursor})).json()
                seen += page
                if page:
                    cursor = page[-1]["seq"]
                status = (await client.get("/api/status")).json()
                if status["run_state"] == "done":
                    break
                await asyncio.sleep(0.02)
            tail = (await client.get("/api/events", params={"since": cursor})).json()
            from_zero = (await client.get("/api/events", params={"since": 0})).json()
            last_seq = from_zero[-1]["seq"]
            from_last = (await client.get("/api/events", params={"since": last_seq})).json()
            return seen, tail, from_zero, from_last

    seen, tail, from_zero, from_last = asyncio.run(scenario())

    assert [event["seq"] for event in from_zero] == list(range(1, 41))
    assert len(from_zero) == 40
    # Incremental polling: every page only carries events past the cursor, no repeats.
    assert [event["seq"] for event in seen] == list(range(1, 41))
    assert all(event["seq"] > 0 for event in seen)
    assert tail == []
    assert from_last == []
    assert from_zero[0]["urgency"] == "urgent"  # the fall is processed (and emitted) first
    assert "thud" in from_zero[0]["text"].lower()


def test_double_run_returns_409_without_disturbing_the_active_run() -> None:
    app = create_app(settings=_settings(), step_delay_s=0.03)

    async def scenario() -> tuple[dict, httpx.Response, dict, dict]:
        async with _client(app) as client:
            first = (await client.post("/api/run")).json()
            await asyncio.sleep(0.1)
            second = await client.post("/api/run")
            active = (await client.get("/api/status")).json()
            done = await _wait_for_run_state(client, "done")
            return first, second, active, done

    first, second, active, done = asyncio.run(scenario())

    assert first["started"] is True
    assert first["run_id"]
    assert second.status_code == 409
    assert "active" in second.json()["detail"].lower()
    assert first["run_id"] in second.json()["detail"]
    assert active["run_state"] == "running"
    assert active["running"] is True
    assert done["run_state"] == "done"
    assert done["events_processed"] == 40
    assert done["digest_ready"] is True


def test_a_new_run_resets_the_prior_run_state() -> None:
    app = create_app(settings=_settings(), step_delay_s=0.03)

    async def scenario() -> dict:
        async with _client(app) as client:
            first = (await client.post("/api/run")).json()
            await _wait_for_run_state(client, "done")
            first_digest = (await client.get("/api/digest")).json()
            first_calls = (await client.get("/api/calls")).json()

            second = (await client.post("/api/run")).json()
            early_digest = (await client.get("/api/digest")).json()
            early_events = (await client.get("/api/events", params={"since": 0})).json()
            await _wait_for_run_state(client, "done")
            final_digest = (await client.get("/api/digest")).json()
            return {
                "first": first,
                "first_digest": first_digest,
                "first_calls": first_calls,
                "second": second,
                "early_digest": early_digest,
                "early_events": early_events,
                "final_digest": final_digest,
            }

    result = asyncio.run(scenario())

    assert result["first_digest"]["ready"] is True
    assert result["first_calls"]
    assert result["second"]["run_id"] != result["first"]["run_id"]
    # Reset, not leakage: the digest is not ready and the event sequence restarts at 1.
    assert result["early_digest"] == {"ready": False}
    assert len(result["early_events"]) < 40
    assert [event["seq"] for event in result["early_events"]] == list(
        range(1, len(result["early_events"]) + 1)
    )
    assert result["final_digest"]["ready"] is True
    assert result["final_digest"]["header"] == result["first_digest"]["header"]


def test_egress_demo_blocks_before_any_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    reached: list[str] = []
    connects: list[Any] = []

    def never_reach_wrapped_client(*_args: Any, **_kwargs: Any) -> None:
        reached.append("complete")
        raise AssertionError("the wrapped client must never be reached for blocked PHI")

    def block_connect(_sock: Any, address: Any) -> None:
        connects.append(address)
        raise AssertionError(f"network connect attempted: {address}")

    monkeypatch.setattr(LLMClient, "complete", never_reach_wrapped_client)
    monkeypatch.setattr(socket.socket, "connect", block_connect)

    app = create_app(settings=_settings())

    async def scenario() -> httpx.Response:
        async with _client(app) as client:
            return await client.post("/api/egress-demo")

    response = asyncio.run(scenario())
    body = response.json()

    assert response.status_code == 200
    assert body["blocked"] is True
    assert body["host"] == "api.openai.com"
    assert body["attempted_url"] == "https://api.openai.com/v1"
    assert body["allowlist"] == ["127.0.0.1", "localhost"]
    assert body["bytes_egressed"] == 0
    assert "EgressBlocked" in body["message"]
    assert reached == []
    assert connects == []


def test_mock_fail_run_reports_failure_and_the_app_keeps_serving() -> None:
    app = create_app(settings=_settings(MOCK_FAIL_RUN="1"))

    async def scenario() -> dict:
        async with _client(app) as client:
            assert (await client.post("/api/run")).status_code == 200
            failed = await _wait_for_run_state(client, "failed")
            metrics = await client.get("/api/metrics")
            events = await client.get("/api/events", params={"since": 0})
            digest = await client.get("/api/digest")
            return {
                "failed": failed,
                "status": (await client.get("/api/status")).json(),
                "metrics_code": metrics.status_code,
                "events_code": events.status_code,
                "events": events.json(),
                "digest": digest.json(),
            }

    result = asyncio.run(scenario())

    assert result["failed"]["run_state"] == "failed"
    assert result["failed"]["running"] is False
    assert "MOCK_FAIL_RUN" in result["failed"]["error"]
    assert result["status"]["run_state"] == "failed"
    assert result["status"]["error"] == result["failed"]["error"]
    assert result["metrics_code"] == 200
    assert result["events_code"] == 200
    assert result["events"] == []
    assert result["digest"] == {"ready": False}


def test_simulate_load_drives_the_curve_through_every_policy_mode() -> None:
    clock = FakeClock()
    source = MockMetricsSource(clock=clock)
    app = create_app(settings=_settings(), metrics_source=source)

    async def scenario() -> dict:
        async with _client(app) as client:
            baseline = (await client.get("/api/status")).json()
            start = await client.post(
                "/api/simulate-load", json={"duration_s": 12, "intensity": 1.5}
            )
            modes: list[tuple[str, str]] = []
            for _ in range(60):
                clock.advance(0.5)
                status = (await client.get("/api/status")).json()
                history = (await client.get("/api/metrics")).json()["history"]
                modes.append((status["mode"], status["reason"]))
                if len(modes) > 4 and modes[-1][0] == "NORMAL":
                    break
            return {
                "baseline": baseline,
                "start": start.json(),
                "modes": modes,
                "history": history,
            }

    result = asyncio.run(scenario())
    observed = [mode for mode, _ in result["modes"]]

    assert result["start"] == {
        "started": True,
        "mode": "mock",
        "duration_s": 12.0,
        "intensity": 1.5,
        "concurrency": 0,
    }
    assert result["baseline"]["mode"] == "NORMAL"
    assert observed[0] == "CONSERVE"
    assert "SURGE" in observed
    assert observed[-1] == "NORMAL"
    assert observed.index("CONSERVE") < observed.index("SURGE")
    assert observed.index("SURGE") < len(observed) - 1

    reasons = dict(result["modes"])
    assert "batch" in reasons["CONSERVE"]
    assert "deferring" in reasons["SURGE"]
    samples = result["history"]
    waitings = [sample["waiting"] for sample in samples if sample["waiting"] is not None]
    kvs = [sample["kv_cache_pct"] for sample in samples if sample["kv_cache_pct"] is not None]
    assert max(waitings) >= 8
    assert max(kvs) >= 70


def test_status_badge_recovers_to_normal_after_the_load_window() -> None:
    clock = FakeClock()
    source = MockMetricsSource(clock=clock)
    app = create_app(settings=_settings(), metrics_source=source)

    async def scenario() -> tuple[str, str]:
        async with _client(app) as client:
            await client.post("/api/simulate-load", json={"duration_s": 4, "intensity": 1.5})
            for _ in range(8):
                clock.advance(0.5)
                await client.get("/api/metrics")
            surged = (await client.get("/api/status")).json()
            assert (await client.post("/api/run")).status_code == 200
            await _wait_for_run_state(client, "done")
            # Past the load window: the badge must follow live telemetry, not the
            # finished run's last decision.
            for _ in range(40):
                clock.advance(0.5)
                await client.get("/api/metrics")
            recovered = (await client.get("/api/status")).json()
            return surged["mode"], recovered["mode"]

    surged, recovered = asyncio.run(scenario())

    assert surged == "SURGE"
    assert recovered == "NORMAL"


def test_metrics_poller_starts_and_stops_with_the_app() -> None:
    source = MockMetricsSource()
    app = create_app(settings=_settings(), metrics_source=source, poll_interval_s=0.05)
    state = app.state.nightwatch

    async def scenario() -> tuple[bool, list[str]]:
        async with app.router.lifespan_context(app):
            running = [
                task
                for task in asyncio.all_tasks()
                if task.get_name() == "nightwatch-metrics-poller"
            ]
            await asyncio.sleep(0.15)
            sampled = len(source.history()) >= 1
            polling = state.polling
        leaked = [
            task.get_name()
            for task in asyncio.all_tasks()
            if task.get_name() == "nightwatch-metrics-poller" and not task.done()
        ]
        return sampled and bool(running) and polling, leaked

    sampled_and_running, leaked = asyncio.run(scenario())

    assert sampled_and_running is True
    assert leaked == []


def test_dashboard_serves_the_controls_with_local_assets_only() -> None:
    """VAL-DASH-001/002: `GET /` is the real dashboard — controls bound, zero external URLs."""
    app = create_app(settings=_settings())

    async def scenario() -> tuple[int, str, str]:
        async with _client(app) as client:
            page = await client.get("/")
            asset = await client.get("/static/chart.umd.js")
            return page.status_code, page.text, asset.headers.get("content-type", "")

    status, html, asset_type = asyncio.run(scenario())

    assert status == 200
    assert "<title>Night Watch" in html
    for label in ("Run Night Watch", "Simulate Load (30s)", "Attempt PHI Egress"):
        assert label in html, f"missing control label: {label}"
    for element_id in DASHBOARD_ELEMENT_IDS:
        assert f'id="{element_id}"' in html, f"missing dashboard element: #{element_id}"

    # The polling layer drives all three documented endpoints, plus the panel feeds.
    for path in ("static/chart.umd.js", "api/status", "api/metrics", "api/events?since="):
        assert path in html, f"dashboard does not reference {path}"

    # Offline guarantee: every asset reference is local or a data: URI, no absolute URL anywhere.
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs, "expected the vendored chart asset reference"
    for ref in refs:
        assert ref.startswith("/") or ref.startswith("data:"), f"non-local asset reference: {ref}"
    assert "http://" not in html
    assert "https://" not in html
    assert "//cdn" not in html
    assert "javascript:load" not in asset_type  # the chart asset endpoint is static, not a stub


def test_vendored_chartjs_serves_at_the_verified_hash() -> None:
    """VAL-DASH-002: `/static/chart.umd.js` is the vendored Chart.js 4.5.1 build, byte-exact."""
    app = create_app(settings=_settings())

    async def scenario() -> tuple[int, int, str, str]:
        async with _client(app) as client:
            response = await client.get("/static/chart.umd.js")
            body = response.content
            return (
                response.status_code,
                len(body),
                hashlib.sha256(body).hexdigest(),
                response.headers.get("content-type", ""),
            )

    status, size, digest, content_type = asyncio.run(scenario())

    assert status == 200
    assert size == 208_518
    assert digest == CHART_UMD_SHA256
    assert "javascript" in content_type
