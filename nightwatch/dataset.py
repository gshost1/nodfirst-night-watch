"""Deterministic synthetic overnight events for the Night Watch demo and eval."""

import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal

from pydantic import BaseModel

EventSource = Literal["smart_speaker", "med_box", "sibling_message", "bathroom_sensor"]
Urgency = Literal["urgent", "attention", "routine"]


class Event(BaseModel):
    """One synthetic observation from the overnight watch."""

    id: str
    timestamp: datetime
    source: EventSource
    kind: str
    text: str


class LabeledEvent(Event):
    """An event carrying its ground-truth urgency for evaluation."""

    urgency: Urgency

    @property
    def ground_truth_urgency(self) -> Urgency:
        """Return the target label using its explicit descriptive name."""
        return self.urgency


@dataclass(frozen=True)
class _EventSpec:
    hour: int
    minute: int
    source: EventSource
    kind: str
    text_options: tuple[str, ...]
    urgency: Urgency


# The fixed date avoids dependence on wall-clock time while preserving the
# overnight ordering across midnight for consumers that sort timestamps.
_OVERNIGHT_DATE = date(2026, 10, 11)
_SCENARIO: tuple[_EventSpec, ...] = (
    _EventSpec(21, 0, "smart_speaker", "mood_check", (
        "Mom says she feels settled and ready for a quiet evening.",
        "Evening check-in: Mom says her mood is calm and comfortable.",
    ), "routine"),
    _EventSpec(21, 8, "sibling_message", "family_check_in", (
        "Anna texts that she will call Mom in the morning.",
        "Anna checks in by message and says good night to Mom.",
    ), "routine"),
    _EventSpec(21, 15, "med_box", "missed_dose", (
        "Evening metformin dose appears missed: the med box shows the compartment still closed.",
    ), "urgent"),
    _EventSpec(21, 27, "bathroom_sensor", "bathroom_trip", (
        "Bathroom sensor recorded a trip; Mom returned to her room.",
        "A bathroom trip was recorded, followed by a return to the bedroom.",
    ), "attention"),
    _EventSpec(21, 42, "smart_speaker", "mood_check", (
        "Mom says she is comfortable and in a good mood.",
        "Mood check-in: Mom describes the evening as pleasant.",
    ), "routine"),
    _EventSpec(21, 55, "smart_speaker", "check_in", (
        "Mom answers the check-in and says she is ready for bed.",
        "Routine check-in: Mom says she is doing all right before bed.",
    ), "routine"),
    _EventSpec(22, 5, "bathroom_sensor", "bathroom_trip", (
        "Bathroom sensor recorded a trip during the night.",
        "Another bathroom trip was recorded; Mom returned without calling for help.",
    ), "attention"),
    _EventSpec(22, 18, "med_box", "device_status", (
        "The med box reports that the evening compartment remains closed.",
        "Medication box status: the evening compartment has not opened.",
    ), "attention"),
    _EventSpec(22, 32, "sibling_message", "family_check_in", (
        "Pavel texts that he can pick up groceries for Mom tomorrow.",
        "Pavel sends Mom a good-night message and offers to help tomorrow.",
    ), "routine"),
    _EventSpec(22, 46, "smart_speaker", "mood_check", (
        "Mom says she is relaxed and listening to the radio.",
        "Routine mood check: Mom says she is enjoying a quiet evening.",
    ), "routine"),
    _EventSpec(23, 1, "bathroom_sensor", "bathroom_trip", (
        "Bathroom sensor recorded a trip and a return to the bedroom.",
        "Bathroom door sensor logged a trip shortly after eleven.",
    ), "attention"),
    _EventSpec(23, 12, "sibling_message", "family_check_in", (
        "Anna messages that she will check on Mom after breakfast.",
        "Anna checks in by text and asks Mom to call if she needs anything.",
    ), "routine"),
    _EventSpec(23, 25, "smart_speaker", "check_in", (
        "Mom responds to the speaker check-in and says she is comfortable.",
        "Routine check-in: Mom says she is all right and does not need anything.",
    ), "routine"),
    _EventSpec(23, 39, "smart_speaker", "confusion_episode", (
        "Mom asks where her mother is and sounds confused about being at home.",
    ), "urgent"),
    _EventSpec(23, 55, "smart_speaker", "mood_check", (
        "Mom says she feels calmer after a brief conversation.",
        "After a short conversation, Mom says she is comfortable and ready to rest.",
    ), "routine"),
    _EventSpec(0, 12, "bathroom_sensor", "bathroom_trip", (
        "Bathroom sensor recorded a trip shortly after midnight.",
        "A bathroom trip was recorded; the sensor registered Mom returning.",
    ), "attention"),
    _EventSpec(0, 29, "sibling_message", "family_check_in", (
        "Pavel sends a message asking Anna to share the morning update.",
        "Pavel messages that he will be available by phone in the morning.",
    ), "routine"),
    _EventSpec(0, 45, "smart_speaker", "routine_chatter", (
        "Mom asks the speaker to play a quiet song.",
        "Mom asks for the weather in the morning and then says good night.",
    ), "routine"),
    _EventSpec(1, 4, "smart_speaker", "check_in", (
        "Routine check-in: Mom says she is resting comfortably.",
        "Mom answers the check-in and says she is doing fine.",
    ), "routine"),
    _EventSpec(1, 21, "bathroom_sensor", "bathroom_trip", (
        "Bathroom sensor recorded a trip; the hall night-light did not turn on.",
        "Bathroom trip recorded: Mom says the hall night-light was hard to find.",
    ), "attention"),
    _EventSpec(1, 25, "smart_speaker", "routine_chatter", (
        "Mom says the hallway night-light is difficult to see from her room.",
        "Check-in note: Mom would like the bathroom night-light checked tomorrow.",
    ), "routine"),
    _EventSpec(1, 39, "sibling_message", "family_check_in", (
        "Anna messages Pavel that she will call Mom after breakfast.",
        "Anna tells Pavel by text that she will handle the morning check-in.",
    ), "routine"),
    _EventSpec(1, 56, "smart_speaker", "mood_check", (
        "Mom says she is feeling sleepy and comfortable.",
        "Routine mood check: Mom says she is tired and ready to rest.",
    ), "routine"),
    _EventSpec(2, 11, "med_box", "device_status", (
        "The medication organizer shows the breakfast compartment ready for later.",
        "Med box status: the morning compartment is prepared for breakfast time.",
    ), "attention"),
    _EventSpec(2, 27, "bathroom_sensor", "bathroom_trip", (
        "Bathroom sensor recorded a trip and a return to Mom's room.",
        "Bathroom door sensor recorded another overnight trip.",
    ), "attention"),
    _EventSpec(2, 46, "sibling_message", "family_check_in", (
        "Pavel texts Anna to remind her to check the bathroom night-light.",
        "Pavel messages that he can check the hall light and floor tomorrow.",
    ), "routine"),
    _EventSpec(3, 12, "smart_speaker", "possible_fall", (
        'Transcript: a loud thud, then no verbal response to "Mom, are you okay?" Possible fall.',
    ), "urgent"),
    _EventSpec(3, 19, "smart_speaker", "check_in", (
        "After a pause, Mom says the speaker is difficult to hear from bed.",
        "The speaker check-in receives a quiet reply that Mom is resting.",
    ), "routine"),
    _EventSpec(3, 39, "smart_speaker", "routine_chatter", (
        "Mom asks the time and says she would like to rest a little longer.",
        "Routine speaker exchange: Mom asks for the time and says good night.",
    ), "routine"),
    _EventSpec(3, 58, "med_box", "device_status", (
        "Med box battery status is normal; the next reminder is scheduled for breakfast.",
        "Medication box reports normal battery and a pending morning reminder.",
    ), "routine"),
    _EventSpec(4, 16, "sibling_message", "family_check_in", (
        "Anna tells Pavel she will call Mom as soon as she wakes.",
        "Anna texts Pavel that she is keeping her phone nearby overnight.",
    ), "routine"),
    _EventSpec(4, 34, "smart_speaker", "mood_check", (
        "Mom says she feels tired but comfortable.",
        "Routine mood check: Mom says she is resting and doing all right.",
    ), "routine"),
    _EventSpec(4, 52, "smart_speaker", "routine_chatter", (
        "Mom says she would like tea with breakfast.",
        "Mom asks whether it will be sunny later this morning.",
    ), "routine"),
    _EventSpec(5, 10, "med_box", "device_status", (
        "Breakfast medication reminder remains queued for its scheduled time.",
        "The med box confirms the breakfast reminder is ready for later.",
    ), "routine"),
    _EventSpec(5, 28, "sibling_message", "family_check_in", (
        "Pavel messages Anna that he can visit Mom later in the day.",
        "Pavel checks in with Anna and offers to help with the morning errands.",
    ), "routine"),
    _EventSpec(5, 43, "smart_speaker", "mood_check", (
        "Mom says she is awake and feeling okay this morning.",
        "Morning mood check-in: Mom says she slept some and feels comfortable.",
    ), "routine"),
    _EventSpec(5, 58, "sibling_message", "family_check_in", (
        "Anna messages that she is on her way to call Mom.",
        "Anna sends a good-morning message and says she will call shortly.",
    ), "routine"),
    _EventSpec(6, 12, "smart_speaker", "check_in", (
        "Mom answers the morning check-in and says she is ready for breakfast.",
        "Routine check-in: Mom says she is doing well and would like breakfast.",
    ), "routine"),
    _EventSpec(6, 22, "med_box", "device_status", (
        "The med box is charged and ready for the day's scheduled reminders.",
        "Medication box status is normal for the morning.",
    ), "routine"),
    _EventSpec(6, 30, "smart_speaker", "check_in", (
        "End-of-watch check-in: Mom says she is comfortable and ready for breakfast.",
        "The overnight watch ends with Mom saying she is comfortable this morning.",
    ), "routine"),
)

_LABELED_EVENT_IDS = (
    "event-01",
    "event-03",
    "event-04",
    "event-05",
    "event-07",
    "event-09",
    "event-11",
    "event-12",
    "event-14",
    "event-15",
    "event-16",
    "event-17",
    "event-19",
    "event-20",
    "event-24",
    "event-25",
    "event-27",
    "event-29",
    "event-33",
    "event-40",
)


def generate_events(seed: int = 42) -> list[Event]:
    """Build the 40-event overnight scenario using only the supplied seed."""
    rng = random.Random(seed)
    events = []
    for index, spec in enumerate(_SCENARIO, start=1):
        event_date = _OVERNIGHT_DATE + timedelta(days=int(spec.hour < 21))
        timestamp = datetime.combine(event_date, time(spec.hour, spec.minute))
        events.append(
            Event(
                id=f"event-{index:02d}",
                timestamp=timestamp,
                source=spec.source,
                kind=spec.kind,
                text=rng.choice(spec.text_options),
            )
        )
    return events


def generate_labeled_events(seed: int = 42) -> list[LabeledEvent]:
    """Return a deterministic 20-event eval subset with ground-truth urgency."""
    by_id = {event.id: event for event in generate_events(seed)}
    urgency_by_id = {
        f"event-{index:02d}": spec.urgency for index, spec in enumerate(_SCENARIO, start=1)
    }
    return [
        LabeledEvent(**by_id[event_id].model_dump(), urgency=urgency_by_id[event_id])
        for event_id in _LABELED_EVENT_IDS
    ]
