"""Morning digest for one Night Watch run — structured data plus rendered markdown.

`build_digest` is a pure function of the run's processed events (plus the overview
line the pipeline asked the model for): the same inputs always produce the same
digest. Sibling tasks are derived deterministically from the events themselves, so
the offline mock demo always yields the two headline assignments — Anna contacts
Mom's doctor about the missed metformin dose, Pavel checks the bathroom night
light and trip hazards — and a "deferred items" note appears exactly when a SURGE
step deferred non-urgent work.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel

__all__ = [
    "ANNA",
    "DEFAULT_OVERVIEW",
    "HEADER_LINE",
    "PAVEL",
    "DeferredItems",
    "Digest",
    "DigestTask",
    "ProcessedEvent",
    "build_digest",
]

#: Rendered verbatim as the digest title; the demo's 1:25 moment depends on it.
HEADER_LINE = "Mom (Ludmila, 81) — overnight watch 21:00–06:30"

#: Used when the caller passes no overview line (e.g. the model call failed).
DEFAULT_OVERVIEW = "Here is how Mom's night went and what the family can do next."

ANNA = "Anna"
PAVEL = "Pavel"


class ProcessedEvent(BaseModel):
    """One overnight event as the pipeline finished with it (summary included)."""

    id: str
    timestamp: datetime
    source: str
    kind: str
    text: str
    urgency: str
    rationale: str
    summary: str
    ok: bool = True
    deferred: bool = False


class DigestTask(BaseModel):
    """The tasks one sibling owns this morning."""

    sibling: str
    tasks: list[str]


class DeferredItems(BaseModel):
    """What a SURGE step held back, summarized together at the end of the run."""

    count: int
    note: str
    event_ids: list[str]


class Digest(BaseModel):
    """The morning digest: structured fields and the rendered markdown."""

    date: str
    header: str
    overview: str
    urgent_alerts: list[dict[str, Any]]
    tasks: list[DigestTask]
    attention_items: list[dict[str, Any]]
    deferred_items: DeferredItems | None = None
    stats: dict[str, Any]
    markdown: str

    @property
    def data(self) -> dict[str, Any]:
        """The structured digest without the markdown (both carry the same items)."""
        return {
            "date": self.date,
            "header": self.header,
            "overview": self.overview,
            "urgent_alerts": self.urgent_alerts,
            "tasks": [task.model_dump() for task in self.tasks],
            "attention_items": self.attention_items,
            "deferred_items": self.deferred_items.model_dump() if self.deferred_items else None,
            "stats": self.stats,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready shape for the API: the structured data plus the markdown."""
        return {**self.data, "markdown": self.markdown}


def _haystack(event: ProcessedEvent) -> str:
    """The same case-insensitive haystack the classifier's safety net searches."""
    return f"{event.kind.replace('_', ' ')} {event.text}".casefold().replace("\u2019", "'")


def _is_missed_dose(event: ProcessedEvent) -> bool:
    haystack = _haystack(event)
    return "missed dose" in haystack or ("missed" in haystack and "dose" in haystack)


def _is_bathroom_trip(event: ProcessedEvent) -> bool:
    haystack = _haystack(event)
    return event.source == "bathroom_sensor" or "bathroom trip" in haystack


def _is_bathroom_or_night_light(event: ProcessedEvent) -> bool:
    haystack = _haystack(event)
    return (
        _is_bathroom_trip(event)
        or "night light" in haystack
        or "night-light" in haystack
        or "nightlight" in haystack
    )


def _anna_tasks(processed: list[ProcessedEvent]) -> list[str]:
    """Anna owns the medical follow-ups: the doctor conversation, morning check-in."""
    tasks: list[str] = []
    missed = [event for event in processed if _is_missed_dose(event)]
    if missed:
        times = ", ".join(event.timestamp.strftime("%H:%M") for event in missed)
        tasks.append(
            "Contact Mom's doctor about the missed evening metformin dose "
            f"(med box flagged it at {times})."
        )
    else:
        tasks.append(
            "Call Mom in the morning to hear how the night went and to confirm "
            "today's medications."
        )
    if any("confus" in _haystack(event) for event in processed):
        tasks.append("Mention the overnight confusion episode to Mom's doctor at the same time.")
    return tasks


def _pavel_tasks(processed: list[ProcessedEvent]) -> list[str]:
    """Pavel owns the home-safety walk-through: the night light, trip hazards."""
    signals = [event for event in processed if _is_bathroom_or_night_light(event)]
    if not signals:
        return [
            "Do a quick evening walk-through of Mom's hallway and bathroom to clear "
            "anything out of place."
        ]
    trips = sum(1 for event in signals if _is_bathroom_trip(event))
    task = "Check the bathroom night light and clear trip hazards before tonight"
    if trips:
        return [f"{task} (Mom had {trips} bathroom trips during the watch)."]
    return [f"{task}."]


def _alert_dict(event: ProcessedEvent) -> dict[str, Any]:
    """JSON-ready urgent alert entry (rendered order = processing order)."""
    return {
        "id": event.id,
        "timestamp": event.timestamp.isoformat(),
        "time": event.timestamp.strftime("%H:%M"),
        "source": event.source,
        "kind": event.kind,
        "text": event.text,
        "summary": event.summary,
        "rationale": event.rationale,
    }


def _attention_dict(event: ProcessedEvent) -> dict[str, Any]:
    """JSON-ready attention entry for things worth a caregiver's check."""
    return {
        "id": event.id,
        "timestamp": event.timestamp.isoformat(),
        "time": event.timestamp.strftime("%H:%M"),
        "source": event.source,
        "kind": event.kind,
        "summary": event.summary,
    }


def _deferred_items(
    deferred_event_ids: Sequence[str], by_id: dict[str, ProcessedEvent]
) -> DeferredItems | None:
    """The deferred-items record, or None when nothing was deferred."""
    if not deferred_event_ids:
        return None
    events = [by_id[event_id] for event_id in deferred_event_ids if event_id in by_id]
    times = sorted(event.timestamp for event in events)
    first = times[0].strftime("%H:%M") if times else "--:--"
    last = times[-1].strftime("%H:%M") if times else "--:--"
    count = len(deferred_event_ids)
    return DeferredItems(
        count=count,
        note=(
            f"During an inference surge, {count} non-urgent events were held back and "
            f"summarized together in one final batch (times {first} to {last})."
        ),
        event_ids=list(deferred_event_ids),
    )


def _render_markdown(
    *,
    date: str,
    overview: str,
    alerts: list[dict[str, Any]],
    tasks: list[DigestTask],
    attention: list[dict[str, Any]],
    deferred: DeferredItems | None,
    stats: dict[str, Any],
) -> str:
    lines: list[str] = [f"# {HEADER_LINE}", ""]
    if date:
        lines += [f"Morning digest for {date}.", ""]
    if overview:
        lines += [overview, ""]

    if alerts:
        lines += ["## Urgent alerts", ""]
        for number, alert in enumerate(alerts, start=1):
            lines.append(
                f"{number}. **{alert['time']} — {alert['kind']}** "
                f"({alert['id']}, {alert['source']})"
            )
            lines.append(f"   {alert['summary']}")
            if alert.get("rationale"):
                lines.append(f"   Why: {alert['rationale']}")
        lines.append("")

    for task in tasks:
        lines += [f"## Tasks for {task.sibling}", ""]
        lines += [f"- {item}" for item in task.tasks]
        lines.append("")

    if attention:
        lines += ["## Watch for these", ""]
        lines += [
            f"- {item['time']} {item['kind']} ({item['id']}): {item['summary']}"
            for item in attention
        ]
        lines.append("")

    if deferred is not None:
        lines += ["## Deferred items", "", f"- {deferred.note}", ""]

    lines += ["## Overnight stats", ""]
    lines.append(
        f"- {stats['events_total']} events processed: {stats['urgent']} urgent, "
        f"{stats['attention']} attention, {stats['routine']} routine."
    )
    if "llm_calls" in stats:
        lines.append(
            f"- {stats['llm_calls']} LLM calls ({stats.get('failed_calls', 0)} failed); "
            f"modes used: {', '.join(stats.get('modes_used', []))}."
        )
    return "\n".join(lines).rstrip() + "\n"


def build_digest(
    processed: Sequence[ProcessedEvent],
    *,
    overview: str = "",
    deferred_event_ids: Sequence[str] = (),
    extra_stats: dict[str, Any] | None = None,
) -> Digest:
    """Build the morning digest from processed events: structured data + markdown."""
    processed = list(processed)
    earliest = min((event.timestamp for event in processed), default=None)
    date = earliest.date().isoformat() if earliest else ""

    urgent = [event for event in processed if event.urgency == "urgent"]
    attention = [event for event in processed if event.urgency == "attention"]
    tasks = [
        DigestTask(sibling=ANNA, tasks=_anna_tasks(processed)),
        DigestTask(sibling=PAVEL, tasks=_pavel_tasks(processed)),
    ]
    deferred = _deferred_items(deferred_event_ids, {event.id: event for event in processed})
    stats: dict[str, Any] = {
        "events_total": len(processed),
        "urgent": len(urgent),
        "attention": len(attention),
        "routine": len(processed) - len(urgent) - len(attention),
        **(extra_stats or {}),
    }

    alert_dicts = [_alert_dict(event) for event in urgent]
    attention_dicts = [_attention_dict(event) for event in attention]
    markdown = _render_markdown(
        date=date,
        overview=overview or DEFAULT_OVERVIEW,
        alerts=alert_dicts,
        tasks=tasks,
        attention=attention_dicts,
        deferred=deferred,
        stats=stats,
    )
    return Digest(
        date=date,
        header=HEADER_LINE,
        overview=overview or DEFAULT_OVERVIEW,
        urgent_alerts=alert_dicts,
        tasks=tasks,
        attention_items=attention_dicts,
        deferred_items=deferred,
        stats=stats,
        markdown=markdown,
    )
