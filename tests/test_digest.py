"""Offline contract tests for the morning digest.

Covers VAL-DIGEST-001: date + "Mom (Ludmila, 81) — overnight watch 21:00–06:30"
header, urgent alerts first, Anna → contact Mom's doctor re: the missed metformin
dose, Pavel → check the bathroom night light / trip hazards, and the deferred-items
note exactly when SURGE deferred work — in both the structured data and the
rendered markdown.
"""

from __future__ import annotations

from datetime import datetime

from nightwatch.digest import (
    DEFAULT_OVERVIEW,
    HEADER_LINE,
    ProcessedEvent,
    build_digest,
)


def _processed(
    event_id: str,
    hour: int,
    minute: int,
    *,
    day: int = 11,
    source: str = "smart_speaker",
    kind: str = "mood_check",
    text: str = "Mom says she is comfortable.",
    urgency: str = "routine",
    summary: str = "A calm overnight check-in.",
) -> ProcessedEvent:
    return ProcessedEvent(
        id=event_id,
        timestamp=datetime(2026, 10, day, hour, minute),
        source=source,
        kind=kind,
        text=text,
        urgency=urgency,
        rationale="A quiet reading for the family.",
        summary=summary,
    )


def _default_results() -> list[ProcessedEvent]:
    """A miniature of the default mock run, in processing order (fall alert first)."""
    return [
        _processed(
            "event-27",
            3,
            12,
            day=12,
            kind="possible_fall",
            urgency="urgent",
            text='Transcript: a loud thud, then no verbal response to "Mom, are you okay?" '
            "Possible fall.",
            summary="A warm note for the family: possible fall at 03:12.",
        ),
        _processed(
            "event-14",
            23,
            39,
            kind="confusion_episode",
            urgency="urgent",
            text="Mom asks where her mother is and sounds confused about being at home.",
            summary="Mom sounded confused about being at home.",
        ),
        _processed(
            "event-03",
            21,
            15,
            source="med_box",
            kind="missed_dose",
            urgency="urgent",
            text="Evening metformin dose appears missed: the med box shows the compartment "
            "still closed.",
            summary="The evening metformin dose appears missed.",
        ),
        _processed(
            "event-01", 21, 0, text="Mom says she feels settled.", summary="A settled evening."
        ),
        _processed(
            "event-04",
            21,
            27,
            source="bathroom_sensor",
            kind="bathroom_trip",
            text="Bathroom sensor recorded a trip; Mom returned to her room.",
            summary="One bathroom trip, no help needed.",
        ),
        _processed(
            "event-20",
            1,
            21,
            day=12,
            source="bathroom_sensor",
            kind="bathroom_trip",
            text="Bathroom sensor recorded a trip; the hall night-light did not turn on.",
            summary="A bathroom trip; the hall light did not turn on.",
        ),
    ]


def test_digest_header_and_date_are_exact() -> None:
    digest = build_digest(_default_results())

    assert digest.header == "Mom (Ludmila, 81) — overnight watch 21:00–06:30"
    assert digest.date == "2026-10-11"
    assert digest.data["header"] == HEADER_LINE
    assert digest.data["date"] == "2026-10-11"
    assert digest.markdown.startswith(f"# {HEADER_LINE}")
    assert "2026-10-11" in digest.markdown


def test_urgent_alerts_come_first_in_data_and_markdown() -> None:
    digest = build_digest(_default_results())

    data = digest.data
    assert [alert["id"] for alert in data["urgent_alerts"]] == ["event-27", "event-14", "event-03"]
    assert data["urgent_alerts"][0]["time"] == "03:12"

    markdown = digest.markdown
    assert markdown.index("## Urgent alerts") < markdown.index("## Tasks for Anna")
    assert markdown.index("## Tasks for Anna") < markdown.index("## Tasks for Pavel")
    for alert in data["urgent_alerts"]:
        assert alert["id"] in markdown


def test_anna_task_contacts_doctor_about_the_missed_metformin() -> None:
    digest = build_digest(_default_results())

    anna = next(task for task in digest.data["tasks"] if task["sibling"] == "Anna")
    matching = [
        item for item in anna["tasks"]
        if "doctor" in item.casefold() and "metformin" in item.casefold()
    ]
    assert matching
    for item in anna["tasks"]:
        assert item in digest.markdown


def test_pavel_task_checks_the_bathroom_night_light_and_trip_hazards() -> None:
    digest = build_digest(_default_results())

    pavel = next(task for task in digest.data["tasks"] if task["sibling"] == "Pavel")
    assert any(
        "night light" in item.casefold() and "trip hazard" in item.casefold()
        for item in pavel["tasks"]
    )
    for item in pavel["tasks"]:
        assert item in digest.markdown


def test_deferred_note_present_only_when_work_was_deferred() -> None:
    deferred_ids = ["event-01", "event-04", "event-20"]

    with_deferred = build_digest(_default_results(), deferred_event_ids=deferred_ids)
    assert with_deferred.data["deferred_items"] is not None
    assert with_deferred.data["deferred_items"]["count"] == 3
    assert with_deferred.data["deferred_items"]["event_ids"] == deferred_ids
    assert "3 non-urgent events" in with_deferred.data["deferred_items"]["note"]
    assert "one final batch" in with_deferred.data["deferred_items"]["note"]
    assert "## Deferred items" in with_deferred.markdown
    assert with_deferred.data["deferred_items"]["note"] in with_deferred.markdown

    without_deferred = build_digest(_default_results())
    assert without_deferred.data["deferred_items"] is None
    assert "## Deferred items" not in without_deferred.markdown


def test_digest_is_deterministic() -> None:
    first = build_digest(_default_results(), deferred_event_ids=["event-01"])
    second = build_digest(_default_results(), deferred_event_ids=["event-01"])

    assert first.markdown == second.markdown
    assert first.data == second.data


def test_fallback_tasks_when_no_specific_signals() -> None:
    results = [
        _processed("event-01", 21, 0, text="Mom says she feels settled."),
        _processed("event-02", 22, 0, text="Mom says she is enjoying a quiet evening."),
    ]
    digest = build_digest(results)

    sections = {task["sibling"]: task["tasks"] for task in digest.data["tasks"]}
    assert sections["Anna"] and sections["Pavel"]
    assert "## Tasks for Anna" in digest.markdown
    assert "## Tasks for Pavel" in digest.markdown


def test_attention_items_are_listed() -> None:
    results = [
        *_default_results(),
        _processed(
            "event-31",
            4,
            16,
            day=12,
            urgency="attention",
            kind="check_in",
            summary="Worth a check-in with Mom in the morning.",
        ),
    ]
    digest = build_digest(results)

    assert [item["id"] for item in digest.data["attention_items"]] == ["event-31"]
    assert "## Watch for these" in digest.markdown
    assert "event-31" in digest.markdown


def test_stats_count_the_night() -> None:
    digest = build_digest(_default_results(), extra_stats={"llm_calls": 20, "failed_calls": 1})

    stats = digest.data["stats"]
    assert stats["events_total"] == 6
    assert stats["urgent"] == 3
    assert stats["attention"] == 0
    assert stats["routine"] == 3
    assert stats["llm_calls"] == 20
    assert stats["failed_calls"] == 1


def test_empty_overview_uses_the_default_line() -> None:
    digest = build_digest(_default_results())
    assert digest.data["overview"] == DEFAULT_OVERVIEW
    assert DEFAULT_OVERVIEW in digest.markdown

    custom = build_digest(_default_results(), overview="Mom had a mostly quiet night.")
    assert custom.data["overview"] == "Mom had a mostly quiet night."
    assert "Mom had a mostly quiet night." in custom.markdown
