from datetime import time

from nightwatch.dataset import generate_events, generate_labeled_events


def test_generate_events_returns_40_reproducible_seeded_events() -> None:
    first = generate_events(seed=42)
    second = generate_events(seed=42)
    another_seed = generate_events(seed=7)

    assert len(first) == 40
    assert [event.model_dump(mode="json") for event in first] == [
        event.model_dump(mode="json") for event in second
    ]
    assert [event.model_dump(mode="json") for event in first] != [
        event.model_dump(mode="json") for event in another_seed
    ]
    assert {event.source for event in first} <= {
        "smart_speaker",
        "med_box",
        "sibling_message",
        "bathroom_sensor",
    }


def test_events_span_the_overnight_window_in_timestamp_order() -> None:
    events = generate_events()

    assert [event.timestamp for event in events] == sorted(
        event.timestamp for event in events
    )
    assert events[0].timestamp.time() == time(21, 0)
    assert events[-1].timestamp.time() == time(6, 30)
    assert events[-1].timestamp.date() > events[0].timestamp.date()


def test_dataset_contains_all_demo_critical_scenario_beats() -> None:
    events = generate_events()

    missed_dose = next(
        event
        for event in events
        if event.source == "med_box" and "missed" in event.text.lower()
    )
    assert missed_dose.timestamp.time() == time(21, 15)
    assert "metformin" in missed_dose.text.lower()

    bathroom_trips = [
        event
        for event in events
        if event.source == "bathroom_sensor" and event.kind == "bathroom_trip"
    ]
    assert 5 <= len(bathroom_trips) <= 7

    possible_fall = next(event for event in events if "possible fall" in event.text.lower())
    assert possible_fall.source == "smart_speaker"
    assert possible_fall.timestamp.time() == time(3, 12)
    assert "loud thud" in possible_fall.text.lower()
    assert "no verbal response" in possible_fall.text.lower()
    assert "Mom, are you okay?" in possible_fall.text

    confusion = next(event for event in events if event.kind == "confusion_episode")
    assert confusion.timestamp.time().hour == 23
    assert 35 <= confusion.timestamp.time().minute <= 45

    sibling_texts = [
        event.text.lower() for event in events if event.source == "sibling_message"
    ]
    assert any("anna" in text for text in sibling_texts)
    assert any("pavel" in text for text in sibling_texts)
    assert any(
        event.kind in {"mood_check", "check_in"}
        and any(word in event.text.lower() for word in ("mood", "check-in", "check in"))
        for event in events
    )


def test_generate_labeled_events_returns_20_ground_truth_labeled_events() -> None:
    events = generate_labeled_events()

    assert len(events) == 20
    assert all(event.urgency in {"urgent", "attention", "routine"} for event in events)
    assert {event.urgency for event in events} == {"urgent", "attention", "routine"}
    assert all(
        event.id and event.timestamp and event.source and event.kind and event.text
        for event in events
    )
