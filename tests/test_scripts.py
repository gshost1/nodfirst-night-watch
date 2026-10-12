"""Tests for the operational scripts (`scripts/load.py`, `scripts/demo.py`).

Covers VAL-LOAD-002 (the load CLI uses the mock app route) and VAL-CROSS-002
(the console demo prints a completed run's results) — all offline, mock mode:

- `load.py` in mock mode POSTs to the app's `/api/simulate-load`, reports
  success, exits 0, and fires zero completions at any inference endpoint.
- `load.py`'s real-vLLM branch (N concurrent non-PHI completions) is implemented
  (exercised here with a fake client, no network) but unreachable while the
  LLM endpoint is mock.
- `demo.py` runs a full mock watch end to end and prints the alerts, the digest,
  and the call-log table, exiting 0 — via the app API when it is reachable,
  in-process otherwise, and adopting an already-active run on 409.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx
import pytest

from config import Settings
from scripts.demo import main as demo_main
from scripts.load import (
    DEFAULT_CONCURRENCY,
    DEFAULT_DURATION_S,
    DEFAULT_INTENSITY,
    decide_branch,
    fire_concurrent_completions,
)
from scripts.load import (
    build_parser as build_load_parser,
)
from scripts.load import (
    main as load_main,
)


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Mock-mode settings from the monkeypatched process env (mirrors test_pipeline)."""
    env = {
        "LLM_BASE_URL": "mock",
        "VLLM_METRICS_URL": "mock",
        "PHI_MODE": "true",
        "MOCK_FAIL_RUN": "0",
        "EGRESS_ALLOWLIST": "localhost,127.0.0.1",
        "PORT": "8765",
    }
    env.update(overrides)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env(os.environ)


def _simulate_load_ok(request: httpx.Request) -> httpx.Response:
    """The app's happy `/api/simulate-load` response (mock branch)."""
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "started": True,
            "mode": "mock",
            "duration_s": body["duration_s"],
            "intensity": body["intensity"],
            "concurrency": 0,
        },
    )


def _refuse(request: httpx.Request) -> httpx.Response:
    """Nothing is listening on the port."""
    raise httpx.ConnectError("connection refused", request=request)


class RecordingTransport(httpx.MockTransport):
    """A MockTransport that remembers every request it served."""

    def __init__(self, handler: Any) -> None:
        self.requests: list[httpx.Request] = []
        super().__init__(self._recording(handler))

    def _recording(self, handler: Any) -> Any:
        def wrapped(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        return wrapped


class CompletionSpy:
    """Stand-in completion client: records calls, never touches the network."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Any, *, phi: bool, max_tokens: int) -> str:
        self.calls.append({"phi": phi, "max_tokens": max_tokens, "n_messages": len(messages)})
        return "ok"


# --------------------------------------------------------------------- load.py


def test_load_cli_flags_have_the_documented_defaults() -> None:
    args = build_load_parser().parse_args([])
    assert args.duration == DEFAULT_DURATION_S == 30.0
    assert args.intensity == DEFAULT_INTENSITY == 1.0
    assert args.n == DEFAULT_CONCURRENCY == 8


def test_load_in_mock_mode_posts_to_the_app_simulate_route(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)
    transport = RecordingTransport(_simulate_load_ok)

    code = load_main(
        ["--duration", "12", "--intensity", "1.5", "--n", "4"],
        settings=settings,
        http=httpx.Client(transport=transport),
    )
    out = capsys.readouterr().out

    assert code == 0
    assert len(transport.requests) == 1, "mock mode must make exactly one request"
    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url.host == "127.0.0.1"
    assert request.url.port == 8765
    assert request.url.path == "/api/simulate-load"
    assert json.loads(request.content) == {"duration_s": 12.0, "intensity": 1.5}
    assert "accepted" in out.lower()
    assert "mock" in out.lower()


def test_load_in_mock_mode_never_fires_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    spy = CompletionSpy()
    transport = RecordingTransport(_simulate_load_ok)

    code = load_main(
        ["--duration", "5", "--n", "16"],
        settings=settings,
        http=httpx.Client(transport=transport),
        completion_client=spy,
    )

    assert code == 0
    assert spy.calls == [], "the mock branch must not issue any completions"
    assert all(request.url.path == "/api/simulate-load" for request in transport.requests)


def test_load_reports_unreachable_app_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)
    code = load_main(
        ["--duration", "1"],
        settings=settings,
        http=httpx.Client(transport=httpx.MockTransport(_refuse)),
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "uvicorn" in (captured.out + captured.err).lower()


def test_load_rejects_invalid_arguments() -> None:
    with pytest.raises(SystemExit) as duration_exit:
        load_main(["--duration", "0"])
    assert duration_exit.value.code == 2

    with pytest.raises(SystemExit) as intensity_exit:
        load_main(["--intensity", "-1"])
    assert intensity_exit.value.code == 2

    with pytest.raises(SystemExit) as concurrency_exit:
        load_main(["--n", "0"])
    assert concurrency_exit.value.code == 2


def test_decide_branch_gates_on_the_mock_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    assert decide_branch(_settings(monkeypatch)) == "app"
    assert (
        decide_branch(_settings(monkeypatch, LLM_BASE_URL="http://vllm-sandbox.local:8000/v1"))
        == "endpoint"
    )


def test_real_branch_fires_concurrent_non_phi_completions() -> None:
    spy = CompletionSpy()

    asyncio.run(fire_concurrent_completions(spy, n=3, duration_s=0.25))

    assert len(spy.calls) >= 3, "every worker must issue at least one completion"
    assert all(call["phi"] is False for call in spy.calls), "load probes are non-PHI"
    assert all(call["max_tokens"] > 0 for call in spy.calls)


# --------------------------------------------------------------------- demo.py


class ScriptedApp:
    """A canned app API for the demo's HTTP mode: probe, run, poll, sections."""

    def __init__(self, *, conflict: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.status_polls = 0
        self.conflict = conflict

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/run" and request.method == "POST":
            if self.conflict:
                return httpx.Response(
                    409, json={"detail": "night-watch run xyz is already active"}
                )
            return httpx.Response(200, json={"run_id": "demo42", "started": True})
        if path == "/api/status":
            self.status_polls += 1
            running = self.status_polls <= 2
            return httpx.Response(
                200,
                json={
                    "run_state": "running" if running else "done",
                    "running": running,
                    "mode": "NORMAL",
                    "reason": "baseline telemetry",
                    "events_total": 40,
                    "events_processed": 12 if running else 40,
                    "alerts_count": 3,
                    "digest_ready": not running,
                    "egress": {
                        "endpoint_host": "mock",
                        "is_mock": True,
                        "allowlist": ["127.0.0.1", "localhost"],
                        "phi_mode": True,
                        "owned": False,
                    },
                    "flags": {"mock_llm": True, "mock_metrics": True, "phi_mode": True},
                },
            )
        if path == "/api/alerts":
            return httpx.Response(
                200,
                json=[
                    {
                        "seq": 0,
                        "time": "03:12",
                        "kind": "possible_fall",
                        "source": "smart_speaker",
                        "text": "Loud thud, no verbal response",
                        "summary": "Possible fall — check on Mom now.",
                    }
                ],
            )
        if path == "/api/digest":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "date": "2026-10-12",
                    "header": "Mom (Ludmila, 81) — overnight watch 21:00–06:30",
                    "markdown": (
                        "# Morning digest\n\n"
                        "- Anna: call Mom's doctor about the missed metformin\n"
                        "- Pavel: check the bathroom night light"
                    ),
                    "urgent_alerts": [],
                    "tasks": [],
                    "attention_items": [],
                    "stats": {},
                },
            )
        if path == "/api/calls":
            return httpx.Response(
                200,
                json=[
                    {
                        "ts": 1.0,
                        "step": "classify",
                        "mode": "NORMAL",
                        "latency_ms": 3,
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "ok": True,
                        "error": None,
                        "params": {"max_tokens": 256},
                        "events": [],
                    }
                ],
            )
        return httpx.Response(404, json={"detail": f"no route for {path}"})


def _assert_sections(out: str) -> None:
    assert "URGENT ALERTS" in out
    assert "MORNING DIGEST" in out
    assert "CALL LOG" in out
    assert "03:12" in out
    assert "Anna" in out and "Pavel" in out
    assert "0 outbound inference calls" in out


def test_demo_in_process_prints_a_complete_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)

    started = time.monotonic()
    code = demo_main(
        ["--in-process", "--step-delay", "0", "--poll-interval", "0.02"],
        settings=settings,
    )
    elapsed = time.monotonic() - started
    out = capsys.readouterr().out

    assert code == 0
    _assert_sections(out)
    # The alerts section lists the 03:12 possible fall first (severity-first order).
    alerts_section = out.split("URGENT ALERTS", 1)[1].split("MORNING DIGEST", 1)[0]
    assert alerts_section.index("03:12") < alerts_section.index("23:39")
    assert alerts_section.index("23:39") < alerts_section.index("21:15")
    assert elapsed < 20, "a zero-delay mock run must not take tens of seconds"


def test_demo_app_mode_drives_the_app_end_to_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)
    app = ScriptedApp()

    code = demo_main(
        ["--poll-interval", "0.02"],
        settings=settings,
        http=httpx.Client(transport=httpx.MockTransport(app.handler)),
    )
    out = capsys.readouterr().out

    assert code == 0
    _assert_sections(out)
    paths = {(request.method, request.url.path) for request in app.requests}
    assert ("POST", "/api/run") in paths
    assert ("GET", "/api/alerts") in paths
    assert ("GET", "/api/digest") in paths
    assert ("GET", "/api/calls") in paths
    assert all(request.url.host == "127.0.0.1" for request in app.requests), "localhost only"


def test_demo_app_mode_adopts_an_already_active_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)
    app = ScriptedApp(conflict=True)

    code = demo_main(
        ["--poll-interval", "0.02"],
        settings=settings,
        http=httpx.Client(transport=httpx.MockTransport(app.handler)),
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "already active" in out
    _assert_sections(out)


def test_demo_falls_back_in_process_when_the_app_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)

    code = demo_main(
        ["--step-delay", "0", "--poll-interval", "0.02"],
        settings=settings,
        http=httpx.Client(transport=httpx.MockTransport(_refuse)),
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "in-process" in out
    _assert_sections(out)


def test_demo_with_explicit_url_fails_when_the_app_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch)

    code = demo_main(
        ["--url", "http://127.0.0.1:8765", "--poll-interval", "0.02"],
        settings=settings,
        http=httpx.Client(transport=httpx.MockTransport(_refuse)),
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "uvicorn" in (captured.out + captured.err).lower()


def test_demo_reports_a_failed_run_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(monkeypatch, MOCK_FAIL_RUN="1")

    code = demo_main(
        ["--in-process", "--step-delay", "0", "--poll-interval", "0.02"],
        settings=settings,
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "failed" in (captured.out + captured.err).lower()
