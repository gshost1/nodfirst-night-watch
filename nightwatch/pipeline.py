"""Night Watch run: classify, order urgent-first, summarize per policy mode, digest.

One `NightWatchRun` walks the overnight events end to end — offline in mock mode
or against the owned vLLM on site, same code path.

Ordering — urgent events jump the queue in every mode. Within the urgent class the
run keeps a severity ladder (the owner's decision): a possible fall (and the rest
of the persona's call-911 patterns: thud, no response, chest pain, trouble
breathing, stroke, bleeding, 911) processes before a confusion episode, which
processes before a missed dose; any other urgent event follows; attention and
routine events come last. Every severity tier is stable-sorted by timestamp. This
makes the 03:12 possible-fall alert the first alert of the default flow (the
demo moment) while every tier stays chronological and all urgent work still
precedes everything else.

Per processing step the run consults `inference.policy.choose_mode` with the
latest telemetry snapshot, so mid-run mode changes are real — the next step uses
the new params and the call log shows the mixed modes:
- NORMAL — one summarization call per event.
- CONSERVE — non-urgent events batched `batch_size` at a time, with the recent
  context capped at `context_items` notes.
- SURGE — urgent events still processed immediately with the fixed urgent-alert
  params; all remaining non-urgent work defers into ONE final batch call, which
  the digest records as deferred items.
Urgent alerts are appended the moment their step finishes, so a dashboard polling
the run sees them immediately.

Every LLM call (classification, summarization, digest overview) gets exactly one
call-log entry with ts, step, mode, latency, token counts, ok/error and the params
used. A call that fails after the client's retries is logged with its error, the
step gets a canned fallback summary, and the run continues — a bad step never
kills the night.

`MOCK_FAIL_RUN=1` (mock mode only) raises a controlled `MockFailRunError`
immediately after start so the failed-run path is demonstrable offline.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from pydantic import BaseModel, Field

from config import Config, ModeParams, Settings
from core.egress_guard import EgressGuard
from core.llm import LLMClient, _estimate_tokens
from inference.metrics import MetricsSnapshot, build_metrics_source
from inference.policy import ModeDecision, choose_mode, params_for_mode
from nightwatch.classify import (
    CLASSIFY_CONTEXT_ITEMS,
    CLASSIFY_MAX_TOKENS,
    Classification,
    EventClassifier,
    fast_lane_classification,
    matched_safety_keywords,
)
from nightwatch.dataset import Event, generate_events
from nightwatch.digest import DEFAULT_OVERVIEW, Digest, ProcessedEvent, build_digest

__all__ = [
    "CANNED_FALLBACK_SUMMARY",
    "CONFUSION_SEVERITY_KEYWORDS",
    "FALL_SEVERITY_KEYWORDS",
    "MISSED_DOSE_SEVERITY_KEYWORDS",
    "MOCK_FAIL_RUN_MESSAGE",
    "CallLogEntry",
    "MockFailRunError",
    "NightWatchRun",
    "RunAlreadyActiveError",
    "SUMMARIZER_SYSTEM_PROMPT",
    "SnapshotProvider",
    "digest_messages",
    "order_for_processing",
    "severity_rank",
    "summarize_messages",
]

#: Error recorded (and raised) when MOCK_FAIL_RUN=1 trips the controlled failure.
MOCK_FAIL_RUN_MESSAGE = "MOCK_FAIL_RUN=1: controlled night-watch failure requested (mock mode only)"

#: Canned summary for a step whose LLM call failed after retries + repair.
CANNED_FALLBACK_SUMMARY = (
    "The assistant could not be reached just now, so this reading is kept as-is "
    "for the family to review."
)

#: Severity tier 0 — the persona's call-911 ladder; a possible fall lives here.
FALL_SEVERITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "fall",
        "fell",
        "thud",
        "no response",
        "unresponsive",
        "chest pain",
        "can't breathe",
        "trouble breathing",
        "stroke",
        "bleeding",
        "911",
    }
)
#: Severity tier 1 — an acute cognitive episode (still urgent, after falls).
CONFUSION_SEVERITY_KEYWORDS: frozenset[str] = frozenset({"confusion", "confused"})
#: Severity tier 2 — a medication concern for the morning.
MISSED_DOSE_SEVERITY_KEYWORDS: frozenset[str] = frozenset({"missed dose", "didn't take"})

#: Urgent severity tiers in processing order; an event matching several takes the
#: most acute tier.
SEVERITY_TIERS: tuple[frozenset[str], ...] = (
    FALL_SEVERITY_KEYWORDS,
    CONFUSION_SEVERITY_KEYWORDS,
    MISSED_DOSE_SEVERITY_KEYWORDS,
)

#: Non-urgent ranks follow the residual urgent tier (see `severity_rank`).
_OTHER_URGENT_RANK = len(SEVERITY_TIERS)
_ATTENTION_RANK = _OTHER_URGENT_RANK + 1
_ROUTINE_RANK = _ATTENTION_RANK + 1

SUMMARIZER_SYSTEM_PROMPT = (
    "You are Night Watch, the private caregiver assistant for one family: "
    "Ludmila — \"Mom\", 81 — and the adult children who share her care, siblings "
    "Anna and Pavel. You write short overnight notes for those adult children. "
    "Warm, plain, and concise — one or two sentences. You are not a doctor and you "
    "never diagnose or suggest treatments, doses, or medication changes. For a "
    "possible medical emergency — chest pain, trouble breathing, a fall, stroke "
    "signs — tell the family to call 911 right away. For changes that worry but "
    "are not life-threatening, tell the family to call Mom's doctor before trying "
    "anything on their own."
)


class NightWatchRunError(RuntimeError):
    """Base class for controlled NightWatchRun failures."""


class MockFailRunError(NightWatchRunError):
    """Raised immediately after start when the mock-only MOCK_FAIL_RUN knob is on."""


class RunAlreadyActiveError(NightWatchRunError):
    """Raised when a second start hits a run that is still executing."""


class SnapshotProvider(Protocol):
    """The telemetry surface the run consults per step (any MetricsSource fits)."""

    def snapshot(self) -> MetricsSnapshot | None: ...


class CallLogEntry(BaseModel):
    """One LLM call as it lands in the run's call log (success or failure)."""

    ts: float
    step: str
    mode: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    ok: bool
    error: str | None = None
    params: dict[str, int]
    events: list[str] = Field(default_factory=list)


def severity_rank(event: Event, urgency: str) -> int:
    """Processing rank: fall > confusion > missed dose > other urgent > attention > routine."""
    if urgency != "urgent":
        return {"attention": _ATTENTION_RANK, "routine": _ROUTINE_RANK}.get(urgency, _ROUTINE_RANK)
    matched = matched_safety_keywords(event)
    for rank, keywords in enumerate(SEVERITY_TIERS):
        if any(keyword in keywords for keyword in matched):
            return rank
    return _OTHER_URGENT_RANK


def order_for_processing(
    events: Sequence[Event], classifications: Sequence[Classification]
) -> list[tuple[Event, Classification]]:
    """Stable severity-first order: urgent tiers before non-urgent, timestamp within each tier."""
    pairs = list(zip(events, classifications, strict=True))

    def sort_key(pair: tuple[Event, Classification]) -> tuple[int, Any]:
        event, classification = pair
        return (severity_rank(event, classification.urgency), event.timestamp)

    return sorted(pairs, key=sort_key)


def summarize_messages(events: Sequence[Event], context: Sequence[str]) -> list[dict[str, str]]:
    """Render one summarization prompt covering a single event or a batch of them."""
    if len(events) == 1:
        event = events[0]
        lines = [
            f"Event: {event.id} at {event.timestamp:%H:%M} — {event.source}/{event.kind}: "
            f"{event.text}",
            "",
            "Write one short, warm summary line for the family.",
        ]
    else:
        lines = [
            f"Summarize these {len(events)} overnight events together in one short, warm note.",
            "",
            "Events:",
        ]
        lines += [
            f"{number}. {event.id} at {event.timestamp:%H:%M} — {event.source}/{event.kind}: "
            f"{event.text}"
            for number, event in enumerate(events, start=1)
        ]
        lines += ["", "Write one short, warm summary note for the family."]
    if context:
        lines += ["", "Recent notes from tonight:"]
        lines += [f"- {note}" for note in context]
    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def digest_messages(notes: Sequence[str]) -> list[dict[str, str]]:
    """Prompt for the digest's opening line, grounded in the night's notable events."""
    lines = [
        "Write the opening line for the morning digest of Mom's overnight watch.",
        "",
        "Notes:",
    ]
    lines += list(notes)
    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _event_note(event: ProcessedEvent) -> str:
    """One compact event line for the digest prompt."""
    return f"{event.id} at {event.timestamp:%H:%M} — {event.source}/{event.kind}: {event.text}"


def _digest_notes(
    alerts: Sequence[ProcessedEvent], processed: Sequence[ProcessedEvent], limit: int
) -> list[str]:
    """Alerts first, then the most recent other events, capped at `limit` notes."""
    alert_ids = {event.id for event in alerts}
    notes = [_event_note(event) for event in alerts]
    remaining = limit - len(notes)
    if remaining > 0:
        recent = [event for event in processed if event.id not in alert_ids]
        notes += [_event_note(event) for event in recent[-remaining:]]
    return notes


def _messages_text(messages: Sequence[dict[str, Any]]) -> str:
    """All message content as one string (for token estimates on failed calls)."""
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append(
                " ".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
            )
    return "\n".join(parts)


def _params_dict(params: ModeParams) -> dict[str, int]:
    return {
        "max_tokens": params.max_tokens,
        "context_items": params.context_items,
        "batch_size": params.batch_size,
    }


def _recent_summaries(processed: list[ProcessedEvent], context_items: int) -> list[str]:
    """The most recent `context_items` summaries, never more than the cap."""
    if context_items <= 0:
        return []
    return [event.summary for event in processed[-context_items:]]


class _ClassificationClient:
    """Binds one call-log step around the wrapped client so classification is logged."""

    def __init__(
        self, run: NightWatchRun, *, step: str, mode: str, params: ModeParams, event_id: str
    ) -> None:
        self._run = run
        self._step = step
        self._mode = mode
        self._params = params
        self._event_id = event_id

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema_hint: str | dict[str, Any],
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> dict[str, Any]:
        run = self._run
        started = time.perf_counter()
        prompt_tokens = _estimate_tokens(_messages_text(messages))
        try:
            payload = run._client.complete_json(
                messages,
                schema_hint,
                phi=phi,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                mode_tag=mode_tag,
            )
        except Exception as exc:
            run._log(
                step=self._step,
                mode=self._mode,
                params=self._params,
                events=[self._event_id],
                latency_ms=round((time.perf_counter() - started) * 1000),
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        run._log(
            step=self._step,
            mode=self._mode,
            params=self._params,
            events=[self._event_id],
            latency_ms=round((time.perf_counter() - started) * 1000),
            prompt_tokens=prompt_tokens,
            completion_tokens=_estimate_tokens(json.dumps(payload)),
            ok=True,
            error=None,
        )
        return payload


class NightWatchRun:
    """One overnight watch: classify → order → summarize per policy mode → digest.

    `execute()` is synchronous and re-entrancy-guarded; the app runs it in a
    worker thread (e.g. `asyncio.to_thread`) and polls `status()` / `alerts()` /
    `processed_events()` / `calls()` from the request handlers. State transitions
    are assignment-atomic and the lists are append-only, so a concurrent reader
    only ever sees a prefix of a consistent run.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        events: Sequence[Event] | None = None,
        metrics_source: SnapshotProvider | None = None,
        settings: Settings | None = None,
        forced_mode: str | None = None,
        step_delay_s: float = 0.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings if settings is not None else Config
        self._client = (
            client
            if client is not None
            else EgressGuard(LLMClient(), self._settings.egress_allowlist)
        )
        self._events = list(events) if events is not None else generate_events()
        self._metrics_source = (
            metrics_source
            if metrics_source is not None
            else build_metrics_source(self._settings)
        )
        self._forced_mode = forced_mode.upper() if forced_mode else None
        if self._forced_mode is not None:
            params_for_mode(self._forced_mode)  # validate eagerly
        if step_delay_s < 0:
            raise ValueError(f"step_delay_s must be >= 0, got {step_delay_s}")
        self._step_delay_s = step_delay_s
        self._sleeper = sleeper
        self._phi = self._settings.phi_mode
        self.run_id = uuid.uuid4().hex[:12]

        self._state = "idle"
        self._error: str | None = None
        self._decision: ModeDecision | None = None
        self._processed: list[ProcessedEvent] = []
        self._alerts: list[ProcessedEvent] = []
        self._alert_meta: dict[str, dict[str, Any]] = {}
        self._t0 = time.perf_counter()
        self._calls: list[CallLogEntry] = []
        self._deferred_ids: list[str] = []
        self._digest: Digest | None = None

    # ------------------------------------------------------------- state machine

    @property
    def run_state(self) -> str:
        """One of idle / running / done / failed."""
        return self._state

    @property
    def is_active(self) -> bool:
        """True while `execute()` is on the stack."""
        return self._state == "running"

    @property
    def digest(self) -> Digest | None:
        """The finished digest, or None until the run completes."""
        return self._digest

    @property
    def deferred_event_ids(self) -> list[str]:
        """Event ids whose summaries a SURGE step deferred into the final batch."""
        return list(self._deferred_ids)

    def execute(self) -> Digest:
        """Run the whole watch synchronously; re-raises after marking the run failed."""
        if self._state == "running":
            raise RunAlreadyActiveError(f"night-watch run {self.run_id} is already active")
        self._state = "running"
        self._error = None
        try:
            digest = self._execute()
        except Exception as exc:
            self._state = "failed"
            self._error = f"{type(exc).__name__}: {exc}"
            raise
        self._state = "done"
        return digest

    def _execute(self) -> Digest:
        if self._settings.mock_fail_run and self._settings.is_mock_llm:
            raise MockFailRunError(MOCK_FAIL_RUN_MESSAGE)
        self._t0 = time.perf_counter()
        self._fast_lane()
        classifications = self._classify_all()
        self._retract_overruled_fast_lane(classifications)
        ordered = order_for_processing(self._events, classifications)
        self._process(ordered)
        return self._build_digest()

    # ------------------------------------------------------------------ polling

    def status(self) -> dict[str, Any]:
        """The run's current operating state (safe to poll from another thread)."""
        decision = self._decision
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "run_state": self._state,
            "running": self._state == "running",
            "mode": decision.mode if decision else None,
            "reason": decision.reason if decision else None,
            "params": asdict(decision.params) if decision else None,
            "urgent_params": asdict(decision.urgent_params) if decision else None,
            "events_total": len(self._events),
            "events_processed": len(self._processed),
            "alerts_count": len(self._alerts),
            "digest_ready": self._digest is not None,
        }
        if self._state == "failed":
            payload["error"] = self._error
        return payload

    def processed_events(self) -> list[dict[str, Any]]:
        """Processed events in processing order, each carrying its seq (0-based)."""
        return [
            {
                "seq": number,
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "kind": event.kind,
                "text": event.text,
                "urgency": event.urgency,
                "summary": event.summary,
                "ok": event.ok,
                "deferred": event.deferred,
            }
            for number, event in enumerate(self._processed)
        ]

    def alerts(self) -> list[dict[str, Any]]:
        """Urgent alerts in emit order (severity-first, the fall first by default)."""
        return [
            {
                "seq": number,
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "time": event.timestamp.strftime("%H:%M"),
                "source": event.source,
                "kind": event.kind,
                "text": event.text,
                "summary": event.summary,
                "rationale": event.rationale,
                "fast_lane": self._alert_meta.get(event.id, {}).get("fast_lane", False),
                "raised_ms": self._alert_meta.get(event.id, {}).get("raised_ms"),
            }
            for number, event in enumerate(list(self._alerts))
        ]

    def calls(self) -> list[dict[str, Any]]:
        """The full call log, one entry per LLM call, in call order."""
        return [entry.model_dump() for entry in self._calls]

    # -------------------------------------------------------------- the pipeline

    def _decide(self) -> ModeDecision:
        """Consult the policy with the latest snapshot (or the forced mode)."""
        if self._forced_mode is not None:
            decision = ModeDecision(
                mode=self._forced_mode,  # type: ignore[arg-type]
                params=params_for_mode(self._forced_mode),
                reason=f"forced mode: {self._forced_mode} (evaluation override)",
            )
        else:
            try:
                snapshot = (
                    self._metrics_source.snapshot() if self._metrics_source is not None else None
                )
            except Exception:
                snapshot = None  # telemetry must never crash the watch
            decision = choose_mode(snapshot, self._settings)
        self._decision = decision
        return decision

    def _fast_lane(self) -> None:
        """Raise safety-net alerts before any LLM call so time-to-alert ignores GPU load."""
        fast: list[ProcessedEvent] = []
        for event in self._events:
            verdict = fast_lane_classification(event)
            if verdict is None:
                continue
            fast.append(
                ProcessedEvent(
                    id=event.id,
                    timestamp=event.timestamp,
                    source=event.source,
                    kind=event.kind,
                    text=event.text,
                    urgency="urgent",
                    rationale=verdict.rationale,
                    summary=event.text,
                )
            )
        fast.sort(key=lambda item: (severity_rank(item, "urgent"), item.timestamp))  # type: ignore[arg-type]
        raised_ms = round((time.perf_counter() - self._t0) * 1000)
        for item in fast:
            self._alert_meta[item.id] = {"fast_lane": True, "raised_ms": raised_ms}
            self._alerts.append(item)

    def _retract_overruled_fast_lane(self, classifications: Sequence[Classification]) -> None:
        """Drop a fast-lane alert the authoritative verdict does not call urgent.

        Unreachable with the real classifier (its safety net forces urgent on the same
        keywords); it keeps the final classification the single source of truth.
        """
        overruled = {
            event.id
            for event, verdict in zip(self._events, classifications, strict=True)
            if verdict.urgency != "urgent"
        }
        if overruled & {alert.id for alert in self._alerts}:
            self._alerts = [alert for alert in self._alerts if alert.id not in overruled]

    def _publish_alert(self, processed: ProcessedEvent) -> None:
        """Add an alert, or enrich the fast-lane alert of the same event id in place."""
        alerts = list(self._alerts)
        for position, existing in enumerate(alerts):
            if existing.id == processed.id:
                alerts[position] = processed
                break
        else:
            alerts.append(processed)
            self._alert_meta[processed.id] = {
                "fast_lane": False,
                "raised_ms": round((time.perf_counter() - self._t0) * 1000),
            }
        alerts.sort(key=lambda item: (severity_rank(item, "urgent"), item.timestamp))  # type: ignore[arg-type]
        self._alerts = alerts

    def _classify_all(self) -> list[Classification]:
        """Classify every event with fixed, mode-independent params (one phase)."""
        decision = self._decide()
        params = ModeParams(
            max_tokens=CLASSIFY_MAX_TOKENS, context_items=CLASSIFY_CONTEXT_ITEMS, batch_size=1
        )
        classifications: list[Classification] = []
        for event in self._events:
            client = _ClassificationClient(
                self,
                step=f"classify:{event.id}",
                mode=decision.mode,
                params=params,
                event_id=event.id,
            )
            classifier = EventClassifier(client=client, phi=self._phi)
            classifications.append(classifier.classify_event(event, mode_tag=decision.mode))
        return classifications

    def _process(self, ordered: list[tuple[Event, Classification]]) -> None:
        index = 0
        total = len(ordered)
        batch_number = 0
        while index < total:
            self._pause_between_steps()
            decision = self._decide()
            event, classification = ordered[index]

            if classification.urgency == "urgent":
                self._summarize_step(
                    [ordered[index]],
                    decision=decision,
                    step=f"summarize:{event.id}",
                    params=decision.urgent_params,
                    deferred=False,
                )
                index += 1
                continue

            if decision.mode == "SURGE":
                batch = self._take_non_urgent(ordered, index, count=total - index)
                self._deferred_ids = [batch_event.id for batch_event, _ in batch]
                self._summarize_step(
                    batch,
                    decision=decision,
                    step="summarize:deferred",
                    params=decision.params,
                    deferred=True,
                )
                index += len(batch)
                continue

            if decision.mode == "CONSERVE":
                batch = self._take_non_urgent(
                    ordered, index, count=decision.params.batch_size
                )
                batch_number += 1
                self._summarize_step(
                    batch,
                    decision=decision,
                    step=f"summarize:batch-{batch_number}",
                    params=decision.params,
                    deferred=False,
                )
                index += len(batch)
                continue

            self._summarize_step(
                [ordered[index]],
                decision=decision,
                step=f"summarize:{event.id}",
                params=decision.params,
                deferred=False,
            )
            index += 1

    @staticmethod
    def _take_non_urgent(
        ordered: list[tuple[Event, Classification]], start: int, *, count: int
    ) -> list[tuple[Event, Classification]]:
        """Up to `count` pairs from `start`, stopping before any urgent event."""
        batch: list[tuple[Event, Classification]] = []
        for pair in ordered[start : start + count]:
            if pair[1].urgency == "urgent":
                break
            batch.append(pair)
        return batch

    def _summarize_step(
        self,
        pairs: list[tuple[Event, Classification]],
        *,
        decision: ModeDecision,
        step: str,
        params: ModeParams,
        deferred: bool,
    ) -> None:
        """One summarization call over one event or a batch; appends processed events."""
        events = [event for event, _ in pairs]
        context = _recent_summaries(self._processed, params.context_items)
        messages = summarize_messages(events, context)
        started = time.perf_counter()
        try:
            result = self._client.complete(
                messages,
                phi=self._phi,
                max_tokens=params.max_tokens,
                enable_thinking=params.enable_thinking,
                mode_tag=decision.mode,
            )
        except Exception as exc:
            self._log(
                step=step,
                mode=decision.mode,
                params=params,
                events=events,
                latency_ms=round((time.perf_counter() - started) * 1000),
                prompt_tokens=_estimate_tokens(_messages_text(messages)),
                completion_tokens=0,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            summary, ok = CANNED_FALLBACK_SUMMARY, False
        else:
            self._log(
                step=step,
                mode=decision.mode,
                params=params,
                events=events,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                ok=True,
                error=None,
            )
            summary, ok = result.content, True

        for event, classification in pairs:
            processed = ProcessedEvent(
                id=event.id,
                timestamp=event.timestamp,
                source=event.source,
                kind=event.kind,
                text=event.text,
                urgency=classification.urgency,
                rationale=classification.rationale,
                summary=summary,
                ok=ok,
                deferred=deferred,
            )
            self._processed.append(processed)
            if processed.urgency == "urgent":
                self._publish_alert(processed)  # enriches the fast-lane alert in place

    def _build_digest(self) -> Digest:
        """One overview call, then the deterministic digest document."""
        decision = self._decide()
        overview = self._digest_overview(decision)
        extra_stats = {
            "llm_calls": len(self._calls),
            "failed_calls": sum(1 for call in self._calls if not call.ok),
            "modes_used": sorted({call.mode for call in self._calls}),
        }
        self._digest = build_digest(
            self._processed,
            overview=overview,
            deferred_event_ids=self._deferred_ids,
            extra_stats=extra_stats,
        )
        return self._digest

    def _digest_overview(self, decision: ModeDecision) -> str:
        """Ask the model for the digest's opening line; fall back to the canned one."""
        params = decision.params
        notes = _digest_notes(self._alerts, self._processed, params.context_items)
        messages = digest_messages(notes)
        started = time.perf_counter()
        try:
            result = self._client.complete(
                messages,
                phi=self._phi,
                max_tokens=params.max_tokens,
                enable_thinking=params.enable_thinking,
                mode_tag=decision.mode,
            )
        except Exception as exc:
            self._log(
                step="digest",
                mode=decision.mode,
                params=params,
                events=[],
                latency_ms=round((time.perf_counter() - started) * 1000),
                prompt_tokens=_estimate_tokens(_messages_text(messages)),
                completion_tokens=0,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return DEFAULT_OVERVIEW
        self._log(
            step="digest",
            mode=decision.mode,
            params=params,
            events=[],
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            ok=True,
            error=None,
        )
        return result.content

    def _pause_between_steps(self) -> None:
        """Demo pacing hook: sleep once per processing step (never in tests by default)."""
        if self._step_delay_s > 0:
            self._sleeper(self._step_delay_s)

    def _log(
        self,
        *,
        step: str,
        mode: str,
        params: ModeParams,
        events: Sequence[Event | str],
        latency_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        ok: bool,
        error: str | None,
    ) -> None:
        event_ids = [event if isinstance(event, str) else event.id for event in events]
        self._calls.append(
            CallLogEntry(
                ts=time.time(),
                step=step,
                mode=mode,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ok=ok,
                error=error,
                params=_params_dict(params),
                events=event_ids,
            )
        )
