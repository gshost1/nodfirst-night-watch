"""Offline contract tests for the Night Watch pipeline.

Covers VAL-PIPE-001 (severity-first urgent ordering, stable by timestamp within
each severity, the 03:12 fall alert emitted first), VAL-PIPE-002 (mid-run mode
changes alter the actual call params), VAL-PIPE-003 (a failed LLM step is logged,
gets a canned fallback, and the run continues), VAL-PIPE-004 (exactly one call-log
entry per LLM call, with step/mode/latency/tokens/ok), and the MOCK_FAIL_RUN knob.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any

import pytest

from config import CONSERVE_PARAMS, SURGE_PARAMS, URGENT_PARAMS, Settings
from core.egress_guard import EgressGuard
from core.llm import LLMClient, LLMResult
from inference.metrics import MetricsSnapshot
from nightwatch.classify import CLASSIFY_CONTEXT_ITEMS, CLASSIFY_MAX_TOKENS, Classification
from nightwatch.dataset import Event, generate_events
from nightwatch.digest import HEADER_LINE
from nightwatch.pipeline import (
    CANNED_FALLBACK_SUMMARY,
    CONFUSION_SEVERITY_KEYWORDS,
    FALL_SEVERITY_KEYWORDS,
    MISSED_DOSE_SEVERITY_KEYWORDS,
    MockFailRunError,
    NightWatchRun,
    RunAlreadyActiveError,
    order_for_processing,
    severity_rank,
)


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Mock-mode settings built from the (monkeypatched) process environment."""
    monkeypatch.setenv("LLM_BASE_URL", "mock")
    monkeypatch.setenv("VLLM_METRICS_URL", "mock")
    monkeypatch.setenv("PHI_MODE", "true")
    monkeypatch.setenv("MOCK_FAIL_RUN", "0")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env(os.environ)


def _snapshot(
    *, waiting: float = 1.0, kv: float = 30.0, ttft: float = 0.5
) -> MetricsSnapshot:
    return MetricsSnapshot(
        waiting=waiting,
        running=2.0,
        kv_cache_pct=kv,
        ttft_p50=ttft,
        tokens_per_s=55.0,
        collected_at=0.0,
        source="mock",
    )


class ScriptedSource:
    """Snapshot provider that walks a scripted sequence, then repeats the last sample."""

    def __init__(self, snapshots: list[MetricsSnapshot]) -> None:
        self.snapshots = list(snapshots)
        self.requests = 0

    def snapshot(self) -> MetricsSnapshot | None:
        if not self.snapshots:
            return None
        index = min(self.requests, len(self.snapshots) - 1)
        self.requests += 1
        return self.snapshots[index]


class ObservingClient:
    """Wraps a real client, recording every complete/complete_json invocation."""

    def __init__(
        self,
        inner: Any,
        fail_when: Any = None,
    ) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []
        self.fail_when = fail_when

    def _maybe_fail(self, kind: str, messages: list[dict[str, Any]]) -> None:
        if self.fail_when is None:
            return
        text = "\n".join(str(message.get("content")) for message in messages)
        if self.fail_when(kind, text):
            raise RuntimeError("simulated LLM outage")

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> LLMResult:
        self.calls.append(
            {
                "kind": "complete",
                "messages": messages,
                "phi": phi,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "mode_tag": mode_tag,
            }
        )
        self._maybe_fail("complete", messages)
        return self.inner.complete(
            messages,
            phi=phi,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            mode_tag=mode_tag,
        )

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema_hint: Any,
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "kind": "complete_json",
                "messages": messages,
                "phi": phi,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "mode_tag": mode_tag,
            }
        )
        self._maybe_fail("complete_json", messages)
        return self.inner.complete_json(
            messages,
            schema_hint,
            phi=phi,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            mode_tag=mode_tag,
        )


class StubClient:
    """Offline stand-in client with canned replies (used where no mock env is loaded)."""

    def __init__(self, base_url: str = "mock") -> None:
        self.base_url = base_url
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> LLMResult:
        self.calls.append({"phi": phi, "max_tokens": max_tokens, "mode_tag": mode_tag})
        return LLMResult(
            content="A quiet overnight note for the family.",
            latency_ms=3,
            prompt_tokens=10,
            completion_tokens=8,
            model="stub",
            mode_tag=mode_tag,
        )

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema_hint: Any,
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"phi": phi, "max_tokens": max_tokens, "mode_tag": mode_tag})
        return {"urgency": "routine", "rationale": "A quiet overnight check-in."}


def make_run(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[NightWatchRun, ObservingClient]:
    """Build a NightWatchRun with an observing client and mock-mode settings."""
    settings = kwargs.pop("settings", None) or _settings(monkeypatch)
    fail_when = kwargs.pop("fail_when", None)
    source = kwargs.pop("source", None)
    if source is not None:
        kwargs["metrics_source"] = source
    inner = kwargs.pop("client", None)
    observing = (
        inner
        if isinstance(inner, ObservingClient)
        else ObservingClient(inner if inner is not None else LLMClient(), fail_when=fail_when)
    )
    run = NightWatchRun(client=observing, settings=settings, **kwargs)
    return run, observing


def _mini_events() -> list[Event]:
    """Seven events: a possible fall, a missed dose, five routine check-ins."""
    def event(
        event_id: str, hour: int, minute: int, kind: str, text: str, source: str = "smart_speaker"
    ) -> Event:
        return Event(
            id=event_id,
            timestamp=datetime(2026, 10, 11, hour, minute),
            source=source,
            kind=kind,
            text=text,
        )

    return [
        event("e-fall", 23, 0, "possible_fall",
              'A loud thud, then no response to "Mom, are you okay?"'),
        event("e-missed", 23, 10, "missed_dose", "Evening metformin dose appears missed.",
              source="med_box"),
        event("e-r1", 23, 20, "mood_check", "Mom says she is comfortable."),
        event("e-r2", 23, 30, "check_in", "Mom says she is doing fine."),
        event("e-r3", 23, 40, "mood_check", "Mom says she is sleepy."),
        event("e-r4", 23, 50, "routine_chatter", "Mom asks for a quiet song."),
        event("e-r5", 23, 55, "check_in", "Mom says good night."),
    ]


def _count_context_lines(messages: list[dict[str, Any]]) -> int:
    """Number of recent-notes lines a summarization prompt carries."""
    text = str(messages[-1]["content"])
    marker = "Recent notes from tonight:"
    if marker not in text:
        return 0
    tail = text.split(marker, 1)[1]
    return sum(1 for line in tail.splitlines() if line.startswith("- "))


# ---------------------------------------------------------------- ordering (VAL-PIPE-001)


def test_severity_sets_partition_the_safety_net() -> None:
    from nightwatch.classify import SAFETY_KEYWORDS

    severity_union = (
        FALL_SEVERITY_KEYWORDS | CONFUSION_SEVERITY_KEYWORDS | MISSED_DOSE_SEVERITY_KEYWORDS
    )
    assert severity_union | {"unsteady"} == set(SAFETY_KEYWORDS)
    assert not FALL_SEVERITY_KEYWORDS & CONFUSION_SEVERITY_KEYWORDS
    assert not FALL_SEVERITY_KEYWORDS & MISSED_DOSE_SEVERITY_KEYWORDS
    assert not CONFUSION_SEVERITY_KEYWORDS & MISSED_DOSE_SEVERITY_KEYWORDS


def test_severity_rank_orders_fall_above_confusion_above_missed_dose() -> None:
    def event(event_id: str, kind: str, text: str) -> Event:
        return Event(
            id=event_id,
            timestamp=datetime(2026, 10, 12, 3, 12),
            source="smart_speaker",
            kind=kind,
            text=text,
        )

    fall = event("fall", "possible_fall", 'A loud thud, no response to "Mom, are you okay?"')
    confusion = event("conf", "confusion_episode", "Mom sounds confused about being at home.")
    missed = event("missed", "missed_dose", "Evening metformin dose appears missed.")
    unsteady = event("unsteady", "gait", "Mom seemed unsteady on her feet.")
    routine = event("ok", "mood_check", "Mom says she is comfortable.")

    assert severity_rank(fall, "urgent") == 0
    assert severity_rank(confusion, "urgent") == 1
    assert severity_rank(missed, "urgent") == 2
    assert severity_rank(unsteady, "urgent") == 3  # other urgent: still before non-urgent
    assert severity_rank(routine, "routine") > severity_rank(unsteady, "urgent")
    assert severity_rank(routine, "attention") < severity_rank(routine, "routine")


def test_order_for_processing_is_severity_first_then_timestamp() -> None:
    events = _mini_events()
    classifications = [
        Classification(urgency="urgent", rationale="r"),  # e-fall
        Classification(urgency="urgent", rationale="r"),  # e-missed
        Classification(urgency="routine", rationale="r"),  # e-r1
        Classification(urgency="attention", rationale="r"),  # e-r2
        Classification(urgency="routine", rationale="r"),  # e-r3
        Classification(urgency="routine", rationale="r"),  # e-r4
        Classification(urgency="routine", rationale="r"),  # e-r5
    ]
    ordered = order_for_processing(events, classifications)

    # Fall (severity 0), missed dose (severity 2), then the attention event, then
    # the routine events in timestamp order.
    assert [event.id for event, _ in ordered] == [
        "e-fall", "e-missed", "e-r2", "e-r1", "e-r3", "e-r4", "e-r5",
    ]


# ------------------------------------------------------------------ default mock run


def test_default_mock_run_completes_with_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    run, observing = make_run(monkeypatch)

    digest = run.execute()

    status = run.status()
    assert status["run_state"] == "done"
    assert status["digest_ready"] is True
    assert status["events_total"] == 40
    assert status["events_processed"] == 40
    assert digest.header == HEADER_LINE
    assert len(run.processed_events()) == 40
    assert len(run.calls()) == len(observing.calls)


def test_fall_alert_is_emitted_first_in_default_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = make_run(monkeypatch)

    run.execute()

    alerts = run.alerts()
    assert [alert["id"] for alert in alerts] == ["event-27", "event-14", "event-03"]
    assert alerts[0]["kind"] == "possible_fall"
    assert alerts[0]["time"] == "03:12"
    assert "fall" in alerts[0]["summary"].casefold()


def test_urgent_events_process_before_non_urgent_stable_by_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = make_run(monkeypatch)

    run.execute()

    feed = run.processed_events()
    assert [item["seq"] for item in feed] == list(range(40))
    ids = [item["id"] for item in feed]
    # Severity ladder first: possible fall, then confusion, then the missed dose.
    assert ids[:3] == ["event-27", "event-14", "event-03"]
    assert all(item["urgency"] == "urgent" for item in feed[:3])
    # Everything after the urgent block is non-urgent and chronological.
    non_urgent = feed[3:]
    assert all(item["urgency"] != "urgent" for item in non_urgent)
    timestamps = [item["timestamp"] for item in non_urgent]
    assert timestamps == sorted(timestamps)


@pytest.mark.parametrize("mode", ["CONSERVE", "SURGE"])
def test_urgent_first_in_every_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    run, _ = make_run(monkeypatch, forced_mode=mode)

    run.execute()

    feed = run.processed_events()
    assert [item["id"] for item in feed[:3]] == ["event-27", "event-14", "event-03"]
    assert all(item["urgency"] == "urgent" for item in feed[:3])
    assert all(item["urgency"] != "urgent" for item in feed[3:])
    assert [alert["id"] for alert in run.alerts()] == ["event-27", "event-14", "event-03"]


# ------------------------------------------------------------------- mode behaviours


def test_normal_mode_makes_one_call_per_event(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = make_run(monkeypatch)

    run.execute()

    calls = run.calls()
    classify = [call for call in calls if call["step"].startswith("classify:")]
    summarize = [call for call in calls if call["step"].startswith("summarize:")]
    digest_call = [call for call in calls if call["step"] == "digest"]
    assert len(classify) == 40
    assert len(summarize) == 40
    assert len(digest_call) == 1
    assert all(len(call["events"]) == 1 for call in summarize)
    assert {call["mode"] for call in summarize} == {"NORMAL"}
    assert {call["params"]["max_tokens"] for call in summarize} == {512, URGENT_PARAMS.max_tokens}


def test_conserve_batches_non_urgent_with_context_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    run, observing = make_run(monkeypatch, forced_mode="CONSERVE")

    run.execute()

    summarize = [call for call in run.calls() if call["step"].startswith("summarize:")]
    urgent_calls = [
        call for call in summarize if call["params"]["max_tokens"] == URGENT_PARAMS.max_tokens
    ]
    batch_calls = [
        call for call in summarize if call["params"]["max_tokens"] == CONSERVE_PARAMS.max_tokens
    ]
    assert len(urgent_calls) == 3
    assert [len(call["events"]) for call in batch_calls] == [4] * 9 + [1]
    assert all(call["mode"] == "CONSERVE" for call in batch_calls)

    batch_messages = [
        call["messages"]
        for call in observing.calls
        if call["kind"] == "complete" and call["max_tokens"] == CONSERVE_PARAMS.max_tokens
    ]
    assert batch_messages
    assert max(_count_context_lines(messages) for messages in batch_messages) == (
        CONSERVE_PARAMS.context_items
    )
    for messages in batch_messages:
        assert _count_context_lines(messages) <= CONSERVE_PARAMS.context_items


def test_surge_defers_non_urgent_into_one_final_batch_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = make_run(monkeypatch, forced_mode="SURGE")

    run.execute()

    summarize = [call for call in run.calls() if call["step"].startswith("summarize:")]
    urgent_calls = [
        call for call in summarize if call["params"]["max_tokens"] == URGENT_PARAMS.max_tokens
    ]
    deferred_calls = [call for call in summarize if call["step"] == "summarize:deferred"]
    assert len(urgent_calls) == 3
    assert len(deferred_calls) == 1
    assert len(deferred_calls[0]["events"]) == 37
    assert deferred_calls[0]["params"]["max_tokens"] == SURGE_PARAMS.max_tokens
    assert deferred_calls[0]["mode"] == "SURGE"
    assert set(run.deferred_event_ids) == set(deferred_calls[0]["events"])

    digest = run.digest
    assert digest is not None
    assert digest.data["deferred_items"] is not None
    assert digest.data["deferred_items"]["count"] == 37

    feed = run.processed_events()
    deferred_feed = [item for item in feed if item["id"] in set(run.deferred_event_ids)]
    assert all(item["deferred"] for item in deferred_feed)


@pytest.mark.parametrize("mode", ["NORMAL", "CONSERVE", "SURGE"])
def test_urgent_calls_use_urgent_params_in_every_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    run, _ = make_run(monkeypatch, forced_mode=mode)

    run.execute()

    urgent_steps = [
        call
        for call in run.calls()
        if call["step"]
        in {"summarize:event-27", "summarize:event-14", "summarize:event-03"}
    ]
    assert len(urgent_steps) == 3
    for call in urgent_steps:
        assert call["params"]["max_tokens"] == URGENT_PARAMS.max_tokens
        assert call["params"]["context_items"] == URGENT_PARAMS.context_items
        assert call["params"]["batch_size"] == URGENT_PARAMS.batch_size


def test_mid_run_mode_change_alters_actual_call_params(monkeypatch: pytest.MonkeyPatch) -> None:
    source = ScriptedSource(
        [
            _snapshot(),
            _snapshot(),
            _snapshot(),
            _snapshot(waiting=4.0),
            _snapshot(waiting=4.0),
            _snapshot(waiting=9.0),
        ]
    )
    run, observing = make_run(monkeypatch, events=_mini_events(), source=source)

    run.execute()

    summarize = [call for call in run.calls() if call["step"].startswith("summarize:")]
    assert [call["mode"] for call in summarize] == ["NORMAL", "NORMAL", "CONSERVE", "CONSERVE"]
    assert [call["params"]["max_tokens"] for call in summarize] == [160, 160, 256, 256]
    assert [len(call["events"]) for call in summarize] == [1, 1, 4, 1]

    digest_call = next(call for call in run.calls() if call["step"] == "digest")
    assert digest_call["mode"] == "SURGE"
    assert digest_call["params"]["max_tokens"] == 128

    observed_modes = [
        call["mode_tag"] for call in observing.calls if call["kind"] == "complete"
    ]
    assert observed_modes == ["NORMAL", "NORMAL", "CONSERVE", "CONSERVE", "SURGE"]


# ------------------------------------------------------------------- failure paths


def test_failed_call_is_logged_and_run_continues_with_canned_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, observing = make_run(
        monkeypatch, fail_when=lambda kind, text: kind == "complete" and "event-05" in text
    )

    run.execute()

    feed_by_id = {item["id"]: item for item in run.processed_events()}
    assert feed_by_id["event-05"]["summary"] == CANNED_FALLBACK_SUMMARY
    assert feed_by_id["event-05"]["ok"] is False

    failed = [call for call in run.calls() if not call["ok"]]
    assert len(failed) == 1
    assert failed[0]["step"] == "summarize:event-05"
    assert "simulated LLM outage" in (failed[0]["error"] or "")
    assert failed[0]["completion_tokens"] == 0

    assert run.status()["run_state"] == "done"
    after = [
        item
        for item in run.processed_events()
        if item["seq"] > feed_by_id["event-05"]["seq"]
    ]
    assert after
    assert all(item["ok"] for item in after)
    assert len(run.calls()) == len(observing.calls)


def test_every_llm_call_is_logged_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    run, observing = make_run(monkeypatch)

    run.execute()

    entries = run.calls()
    assert entries and len(entries) == len(observing.calls)
    required = {"ts", "step", "mode", "latency_ms", "prompt_tokens", "completion_tokens", "ok"}
    for entry in entries:
        assert required <= set(entry)
        assert entry["mode"] in {"NORMAL", "CONSERVE", "SURGE"}
        assert entry["prompt_tokens"] > 0
        if entry["ok"]:
            assert entry["error"] is None
            assert entry["completion_tokens"] > 0
        else:
            assert entry["error"]


@pytest.mark.parametrize("phi_env,expected", [("true", True), ("false", False)])
def test_all_calls_are_tagged_with_phi_mode(
    monkeypatch: pytest.MonkeyPatch, phi_env: str, expected: bool
) -> None:
    run, observing = make_run(
        monkeypatch, settings=_settings(monkeypatch, PHI_MODE=phi_env), events=_mini_events()
    )

    run.execute()

    assert observing.calls
    assert all(call["phi"] is expected for call in observing.calls)


def test_mock_fail_run_raises_controlled_exception_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, observing = make_run(
        monkeypatch, settings=_settings(monkeypatch, MOCK_FAIL_RUN="1")
    )

    with pytest.raises(MockFailRunError):
        run.execute()

    status = run.status()
    assert status["run_state"] == "failed"
    assert "MOCK_FAIL_RUN" in (status["error"] or "")
    assert observing.calls == []
    assert run.processed_events() == []
    assert run.alerts() == []
    assert run.calls() == []


def test_mock_fail_run_knob_is_mock_only(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch, LLM_BASE_URL="https://api.openai.com/v1", MOCK_FAIL_RUN="1"
    )
    run, _ = make_run(
        monkeypatch, client=StubClient(), settings=settings, events=_mini_events()[:2]
    )

    run.execute()

    assert run.status()["run_state"] == "done"


def test_second_start_while_running_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    class GatedClient(ObservingClient):
        def complete(self, messages: Any, **kwargs: Any) -> LLMResult:
            result = super().complete(messages, **kwargs)
            release.wait(timeout=5)
            return result

    observing = GatedClient(LLMClient())
    run = NightWatchRun(client=observing, settings=_settings(monkeypatch))
    thread = threading.Thread(target=run.execute, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while run.status()["run_state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert run.status()["run_state"] == "running"

    with pytest.raises(RunAlreadyActiveError):
        run.execute()

    release.set()
    thread.join(timeout=5)
    assert run.status()["run_state"] == "done"


def test_urgent_alert_is_visible_the_moment_it_is_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class ProbingClient(ObservingClient):
        def __init__(self, inner: Any) -> None:
            super().__init__(inner)
            self.complete_count = 0

        def complete(self, messages: Any, **kwargs: Any) -> LLMResult:
            self.complete_count += 1
            if self.complete_count == 2:
                seen["alert_ids"] = [alert["id"] for alert in run.alerts()]
            return super().complete(messages, **kwargs)

    observing = ProbingClient(LLMClient())
    run = NightWatchRun(client=observing, settings=_settings(monkeypatch))
    run.execute()

    # The second summarization call happens while the first (the 03:12 fall) is
    # already done: its alert must be readable at that exact moment. The fast lane
    # publishes every safety-net alert before the model, so the fall leads the list.
    assert seen["alert_ids"][0] == "event-27"
    assert len(seen["alert_ids"]) == len(set(seen["alert_ids"]))


# ----------------------------------------------------------------------- pacing etc.


def test_step_delay_sleeps_once_per_processing_step(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    run, _ = make_run(
        monkeypatch, events=_mini_events(), step_delay_s=0.02, sleeper=sleeps.append
    )

    run.execute()

    assert sleeps == [0.02] * 7  # one sleep per processing step; classify/digest add none

    run_two, _ = make_run(monkeypatch, events=_mini_events(), sleeper=sleeps.append)

    run_two.execute()

    assert sleeps == [0.02] * 7  # zero delay (the default) never sleeps


def test_run_is_deterministic_across_executions(monkeypatch: pytest.MonkeyPatch) -> None:
    run_one, _ = make_run(monkeypatch)
    run_two, _ = make_run(monkeypatch)

    run_one.execute()
    run_two.execute()

    def without_ts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value for key, value in entry.items() if key != "ts"} for entry in entries]

    assert without_ts(run_one.calls()) == without_ts(run_two.calls())
    feed_one = run_one.processed_events()
    feed_two = run_two.processed_events()
    assert [item["id"] for item in feed_one] == [item["id"] for item in feed_two]
    assert [item["summary"] for item in feed_one] == [item["summary"] for item in feed_two]
    digest_one = run_one.digest
    digest_two = run_two.digest
    assert digest_one is not None and digest_two is not None
    assert digest_one.markdown == digest_two.markdown


def test_run_completes_when_every_call_is_egress_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubClient(base_url="https://api.openai.com/v1")
    settings = _settings(
        monkeypatch, LLM_BASE_URL="https://api.openai.com/v1", EGRESS_ALLOWLIST="localhost"
    )
    guard = EgressGuard(stub, settings.egress_allowlist, base_url="https://api.openai.com/v1")
    run = NightWatchRun(client=guard, settings=settings, events=_mini_events())

    run.execute()

    assert run.status()["run_state"] == "done"
    assert stub.calls == []  # the guard raised before the wrapped client was touched
    entries = run.calls()
    assert entries and all(entry["ok"] is False for entry in entries)
    assert all("EgressBlocked" in (entry["error"] or "") for entry in entries)
    # The keyword safety net keeps urgent recall alive even with no model at all.
    assert {alert["id"] for alert in run.alerts()} == {"e-fall", "e-missed"}
    assert all(
        item["summary"] == CANNED_FALLBACK_SUMMARY for item in run.processed_events()
    )
    digest = run.digest
    assert digest is not None
    assert "metformin" in digest.markdown


# ------------------------------------------------------------------- digest contents


def test_full_mock_run_digest_contents(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = make_run(monkeypatch)

    digest = run.execute()

    data = digest.data
    assert data["header"] == HEADER_LINE
    assert data["date"] == "2026-10-11"
    assert data["stats"]["events_total"] == 40
    assert data["stats"]["urgent"] == 3
    assert data["deferred_items"] is None  # the NORMAL default run defers nothing

    anna = next(task for task in data["tasks"] if task["sibling"] == "Anna")
    assert any(
        "doctor" in item.casefold() and "metformin" in item.casefold() for item in anna["tasks"]
    )
    pavel = next(task for task in data["tasks"] if task["sibling"] == "Pavel")
    assert any(
        "night light" in item.casefold() and "trip hazard" in item.casefold()
        for item in pavel["tasks"]
    )
    assert any("6 bathroom trips" in item for item in pavel["tasks"])

    # The overview line is grounded in the night's notable events, alerts first.
    assert "possible_fall" in data["overview"]

    markdown = digest.markdown
    assert markdown.index("## Urgent alerts") < markdown.index("## Tasks for Anna")
    assert "metformin" in markdown
    assert "night light" in markdown


def test_classification_params_are_fixed_and_mode_independent() -> None:
    """Classification uses the documented fixed budget; the dataset stays 40 events."""
    assert CLASSIFY_MAX_TOKENS == 256
    assert CLASSIFY_CONTEXT_ITEMS == 6
    assert len(generate_events()) == 40
