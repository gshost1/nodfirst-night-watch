"""Urgent-recall eval — the judging claim: urgent recall stays 100% even in SURGE.

Runs the full Night Watch pipeline over the 20 labeled events (`eval/labeled_events.json`,
generated from `nightwatch.dataset.generate_labeled_events()` — the ground truth) in
mock mode, twice:

- once under NORMAL params — one summarization call per event;
- once under SURGE params — urgent events still processed immediately with the fixed
  urgent params, every non-urgent summary deferred into one final batch.

Classification params are fixed and mode-independent by design; the urgent-first
queue and the keyword safety net are exactly what keep recall at 1.0 in SURGE.

Each mode reports urgent recall (ground-truth-urgent events that produced urgent
alerts), precision (emitted alerts whose event is ground-truth urgent; vacuously 1.0
when a mode emitted no alerts — the recall column carries the miss), tokens, latency,
and event counts, printed as one table. Exit code 0 only when urgent recall is 1.0 in
EVERY mode.

`--inject-failure` (the mock-only self-test) forces one known-urgent event's
classification to routine — deliberately past the safety net — proving that a triage
failure drops recall below 1.0 and makes this process exit non-zero: the check is
enforced, not just printed.

Usage: `uv run python eval/run_eval.py` (see `--help`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:  # type-only: never imported at runtime (see the bootstrap below)
    from config import Settings
    from nightwatch.dataset import LabeledEvent

# Direct-file execution (`python eval/run_eval.py`) starts with only eval/ on
# sys.path; the flat layout needs the repo root importable (`config`, `core`, ...).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = [
    "DEFAULT_LABELED_EVENTS_PATH",
    "EVAL_MODES",
    "REQUIRED_RECALL",
    "EvalError",
    "ModeMetrics",
    "build_parser",
    "default_inject_event_id",
    "format_table",
    "load_labeled_events",
    "main",
    "recall_precision",
    "run_mode",
]

#: The committed ground truth, regenerated from the dataset generator (seed 42).
DEFAULT_LABELED_EVENTS_PATH = Path(__file__).resolve().parent / "labeled_events.json"

#: The eval's two forced modes (classification params stay fixed in both).
EVAL_MODES: tuple[str, ...] = ("NORMAL", "SURGE")

#: The gate: anything below perfect urgent recall fails the eval (non-zero exit).
REQUIRED_RECALL = 1.0

#: Rationale stamped on the injected misclassification, visible in the run's alerts log.
INJECTED_RATIONALE = (
    "INJECTED FAILURE (eval self-test): this known-urgent event was forced to routine "
    "to prove the urgent-recall gate fails the process."
)

_TABLE_HEADER = (
    f"{'MODE':<8}  {'EVENTS':>6}  {'URGENT':>6}  {'ALERTS':>6}  "
    f"{'RECALL':>6}  {'PRECISION':>9}  {'TOKENS':>8}  {'MEAN_LAT_MS':>12}  {'WALL_MS':>7}"
)


class EvalError(RuntimeError):
    """Raised when the eval cannot trust its inputs (missing or stale labeled events)."""


class ModeMetrics(BaseModel):
    """One mode's eval row: a full pipeline run reduced to its scored metrics."""

    mode: str
    events_total: int
    ground_truth_urgent_ids: list[str]
    alert_ids: list[str]
    missed_urgent_ids: list[str]
    false_alert_ids: list[str]
    recall: float
    precision: float
    tokens: int
    calls: int
    ok_calls: int
    steps: list[str]
    deferred_count: int
    digest_ready: bool
    mean_latency_ms: float
    wall_ms: int
    injected_event_id: str | None = None


def load_labeled_events(path: Path | None = None) -> list[LabeledEvent]:
    """Load the committed 20-event labeled set, verified against the dataset generator.

    The generator is the eval's ground truth: a file that drifted from
    `generate_labeled_events()` is refused loudly rather than scored against stale
    labels.
    """
    from nightwatch.dataset import LabeledEvent, generate_labeled_events

    resolved = path if path is not None else DEFAULT_LABELED_EVENTS_PATH
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalError(f"labeled events file not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise EvalError(f"labeled events file is not valid JSON: {resolved} ({exc})") from exc

    raw_events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(raw_events, list):
        raise EvalError(f"{resolved} must hold a JSON array of events (or an 'events' key)")

    try:
        labeled = [LabeledEvent.model_validate(item) for item in raw_events]
    except ValidationError as exc:
        raise EvalError(
            f"{resolved} has an event that does not match the Event shape: {exc}"
        ) from exc

    expected = generate_labeled_events()
    if labeled != expected:
        raise EvalError(
            f"{resolved} does not match nightwatch.dataset.generate_labeled_events() — "
            "the generator is the eval's ground truth; regenerate the file from it"
        )
    return labeled


def recall_precision(
    alert_ids: Sequence[str], ground_truth_urgent_ids: Sequence[str]
) -> tuple[float, float]:
    """Urgent recall and alert precision from the run's emitted alert ids.

    Recall: ground-truth-urgent events that produced urgent alerts. Precision:
    emitted alerts whose event is ground-truth urgent — vacuously 1.0 when a mode
    emitted no alerts (the recall column carries that miss).
    """
    truth = set(ground_truth_urgent_ids)
    emitted = set(alert_ids)
    recall = len(truth & emitted) / len(truth) if truth else 1.0
    precision = len(truth & emitted) / len(emitted) if emitted else 1.0
    return recall, precision


def default_inject_event_id(labeled: Sequence[LabeledEvent]) -> str:
    """The demo's most safety-critical event: the possible fall, else the first urgent."""
    urgent = [event for event in labeled if event.urgency == "urgent"]
    for event in urgent:
        if event.kind == "possible_fall":
            return event.id
    if not urgent:
        raise EvalError("cannot inject a failure: no ground-truth-urgent event in the labeled set")
    return urgent[0].id


def run_mode(
    mode: str,
    labeled: Sequence[LabeledEvent],
    *,
    settings: Settings,
    inject_event_id: str | None = None,
) -> ModeMetrics:
    """Run the full pipeline once under one forced mode, then score it against the labels."""
    from core.egress_guard import EgressGuard
    from core.llm import LLMClient
    from nightwatch.classify import Classification
    from nightwatch.dataset import Event
    from nightwatch.pipeline import NightWatchRun

    class ForcedRoutineRun(NightWatchRun):
        """Eval-only run whose triage fails for exactly one event (the self-test).

        `--inject-failure` needs a known-urgent event to come out routine so the
        eval's recall gate provably fires. Overriding `_classify_all` after the
        real (logged) classification calls flips one verdict, deliberately past
        the keyword safety net: this measures the eval's sensitivity to a missed
        urgent alert, not the net itself.
        """

        def __init__(self, *, inject_id: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._inject_id = inject_id

        def _classify_all(self) -> list[Classification]:
            classifications = super()._classify_all()
            return [
                Classification(urgency="routine", rationale=INJECTED_RATIONALE)
                if event.id == self._inject_id
                else classification
                for event, classification in zip(self._events, classifications, strict=True)
            ]

    # The run must see exactly the production Event shape — never the labels.
    events = [
        Event(
            id=event.id,
            timestamp=event.timestamp,
            source=event.source,
            kind=event.kind,
            text=event.text,
        )
        for event in labeled
    ]
    client = EgressGuard(LLMClient(), settings.egress_allowlist)
    kwargs: dict[str, Any] = {
        "client": client,
        "events": events,
        "settings": settings,
        "forced_mode": mode,
        "step_delay_s": 0.0,
    }
    run: NightWatchRun = (
        ForcedRoutineRun(inject_id=inject_event_id, **kwargs)
        if inject_event_id is not None
        else NightWatchRun(**kwargs)
    )

    started = time.perf_counter()
    run.execute()
    wall_ms = round((time.perf_counter() - started) * 1000)

    calls = run.calls()
    alert_ids = [alert["id"] for alert in run.alerts()]
    truth = [event.id for event in labeled if event.urgency == "urgent"]
    recall, precision = recall_precision(alert_ids, truth)
    truth_set, alert_set = set(truth), set(alert_ids)
    return ModeMetrics(
        mode=mode,
        events_total=len(events),
        ground_truth_urgent_ids=truth,
        alert_ids=alert_ids,
        missed_urgent_ids=[event_id for event_id in truth if event_id not in alert_set],
        false_alert_ids=[event_id for event_id in alert_ids if event_id not in truth_set],
        recall=recall,
        precision=precision,
        tokens=sum(call["prompt_tokens"] + call["completion_tokens"] for call in calls),
        calls=len(calls),
        ok_calls=sum(1 for call in calls if call["ok"]),
        steps=[call["step"] for call in calls],
        deferred_count=len(run.deferred_event_ids),
        digest_ready=run.digest is not None,
        mean_latency_ms=(sum(call["latency_ms"] for call in calls) / len(calls)) if calls else 0.0,
        wall_ms=wall_ms,
        injected_event_id=inject_event_id,
    )


def format_table(metrics: Sequence[ModeMetrics]) -> str:
    """The per-mode table: one row per forced mode, every scored field in its column."""
    lines = [_TABLE_HEADER, "-" * len(_TABLE_HEADER)]
    for metric in metrics:
        lines.append(
            f"{metric.mode:<8}  {metric.events_total:>6}  "
            f"{len(metric.ground_truth_urgent_ids):>6}  {len(metric.alert_ids):>6}  "
            f"{metric.recall:>6.3f}  {metric.precision:>9.3f}  {metric.tokens:>8}  "
            f"{metric.mean_latency_ms:>12.1f}  {metric.wall_ms:>7}"
        )
    return "\n".join(lines)


def _mode_detail(metric: ModeMetrics) -> str:
    """One evidence line per mode: full-pipeline call breakdown and deferral."""
    classify = sum(1 for step in metric.steps if step.startswith("classify:"))
    summarize = sum(1 for step in metric.steps if step.startswith("summarize:"))
    digest = sum(1 for step in metric.steps if step == "digest")
    deferred = (
        f", {metric.deferred_count} non-urgent deferred into 1 batch"
        if metric.deferred_count
        else ""
    )
    failed = metric.calls - metric.ok_calls
    failures = f", {failed} FAILED calls" if failed else ""
    return (
        f"{metric.mode}: {metric.calls} calls ({classify} classify, {summarize} summarize, "
        f"{digest} digest){deferred}{failures}"
    )


def build_parser() -> argparse.ArgumentParser:
    """The eval CLI: table output, an injectable failure, no other knobs."""
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description=(
            "Night Watch urgent-recall eval: run the full pipeline over the 20 labeled "
            "events in mock mode under NORMAL and SURGE params; exit non-zero unless "
            "urgent recall is 1.0 in every mode."
        ),
    )
    parser.add_argument(
        "--labeled-events",
        default=str(DEFAULT_LABELED_EVENTS_PATH),
        help="path to the labeled events JSON (default: eval/labeled_events.json)",
    )
    parser.add_argument(
        "--inject-failure",
        action="store_true",
        help=(
            "self-test: force one known-urgent event to routine; the eval must then "
            "report recall < 1.0 and exit non-zero"
        ),
    )
    parser.add_argument(
        "--inject-event",
        metavar="EVENT_ID",
        default=None,
        help="which ground-truth-urgent event --inject-failure targets (default: the fall)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run the eval; returns the process exit code (0 only when recall is 1.0 everywhere)."""
    from config import load_settings

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.inject_event is not None and not args.inject_failure:
        parser.error("--inject-event requires --inject-failure")

    resolved = settings if settings is not None else load_settings()
    try:
        labeled = load_labeled_events(Path(args.labeled_events))
    except EvalError as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 2

    inject_event_id: str | None = None
    if args.inject_failure:
        inject_event_id = args.inject_event or default_inject_event_id(labeled)
        by_id = {event.id: event for event in labeled}
        target = by_id.get(inject_event_id)
        if target is None:
            parser.error(f"--inject-event: {inject_event_id} is not in the labeled set")
        if target.urgency != "urgent":
            parser.error(f"--inject-event: {inject_event_id} is not ground-truth urgent")
        print(
            f"INJECTED FAILURE: {target.id} ({target.timestamp:%H:%M} {target.kind}) "
            "forced to routine — the recall gate must catch this"
        )
        print()

    endpoint = (
        "mock (deterministic canned replies)" if resolved.is_mock_llm else resolved.llm_base_url
    )
    print(f"Night Watch urgent-recall eval — {len(labeled)} labeled events (seed 42)")
    print(f"inference endpoint: {endpoint} · modes: {', '.join(EVAL_MODES)}")
    print()

    metrics = [
        run_mode(mode, labeled, settings=resolved, inject_event_id=inject_event_id)
        for mode in EVAL_MODES
    ]
    print(format_table(metrics))
    print()
    for metric in metrics:
        print(_mode_detail(metric))
    print()

    failing = [metric for metric in metrics if metric.recall < REQUIRED_RECALL]
    if not failing:
        print(f"RESULT: PASS — urgent recall {REQUIRED_RECALL:.3f} in every mode")
        return 0
    for metric in failing:
        missed = ", ".join(metric.missed_urgent_ids) or "(none)"
        print(
            f"RESULT: FAIL — urgent recall {metric.recall:.3f} < {REQUIRED_RECALL:.3f} in "
            f"{metric.mode} (missed urgent: {missed})"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
