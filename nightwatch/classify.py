"""Urgency classification for overnight care events.

The classifier asks the owned model for a structured JSON verdict through
`complete_json` (which extracts the first balanced JSON object and repairs one
unparseable reply) and falls back to a deterministic canned classification when
the model cannot be used at all — a failed classification never escapes as an
exception. On top of that, a case-insensitive keyword safety net runs on every
event, independent of the model, and forces `urgent`.

Classification params are **fixed and mode-independent** (`max_tokens=256`,
`context_items=6`, `enable_thinking=False`): the self-tuning policy governs
summarization, batching and deferral, never the safety-critical triage step.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from config import Config
from core.llm import LLMClient
from nightwatch.dataset import Event, Urgency

#: Safety-net roster (architecture.md §nightwatch/classify.py), matched
#: case-insensitively against the event's kind and text.
SAFETY_KEYWORDS: tuple[str, ...] = (
    "fall",
    "fell",
    "thud",
    "no response",
    "unresponsive",
    "chest pain",
    "can't breathe",
    "trouble breathing",
    "confusion",
    "confused",
    "missed dose",
    "didn't take",
    "unsteady",
    "bleeding",
    "stroke",
    "911",
)

#: The only labels the family-facing digest knows how to render.
VALID_URGENCIES: tuple[Urgency, ...] = ("urgent", "attention", "routine")

# Fixed, mode-independent classification budget (see module docstring).
CLASSIFY_MAX_TOKENS = 256
CLASSIFY_CONTEXT_ITEMS = 6
CLASSIFY_ENABLE_THINKING = False

SCHEMA_HINT: dict[str, str] = {
    "urgency": "one of: urgent | attention | routine",
    "rationale": "one short sentence explaining the urgency to the family",
}

#: A model that cannot be reached is flagged for a human rather than dropped as routine.
CANNED_FALLBACK_URGENCY: Urgency = "attention"
CANNED_FALLBACK_RATIONALE = (
    "Automatic fallback: the classifier model was unavailable, so this event is "
    "flagged for a caregiver to review."
)
UNRECOGNIZED_LABEL_RATIONALE = (
    "The model returned an urgency label we do not recognize, so this event is "
    "flagged for a caregiver to review."
)
DEFAULT_RATIONALE = "The model did not give a rationale; flagged for a caregiver to review."
SAFETY_NET_RATIONALE = (
    "Safety language in this event means it may need help right now: check on Mom and "
    "call 911 if she is unresponsive."
)

SYSTEM_PROMPT = (
    "You triage overnight elder-care events for an adult child coordinating with "
    "siblings. Reply with JSON only. Choose exactly one urgency label: 'urgent' for "
    "anything that may need help right now, 'attention' for something worth a "
    "caregiver's check, 'routine' for ordinary check-ins and chatter. Never diagnose. "
    "Give one short, warm rationale the family can act on."
)


class Classification(BaseModel):
    """The classifier's verdict for a single overnight event."""

    urgency: Urgency
    rationale: str


class JsonCompletionClient(Protocol):
    """The slice of the LLM client surface the classifier depends on."""

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
        """Return the first JSON object in a completion, repairing invalid output once."""
        ...


class EventClassifier:
    """Classifies events with one LLM client (reused across the 40-event run)."""

    def __init__(
        self,
        client: JsonCompletionClient | None = None,
        *,
        phi: bool | None = None,
    ) -> None:
        self.client: JsonCompletionClient = client if client is not None else LLMClient()
        self.phi = Config.phi_mode if phi is None else phi

    def classify_event(self, event: Event, *, mode_tag: str | None = None) -> Classification:
        """Classify one event, then let the safety net override the model if needed."""
        urgency, rationale = self._model_verdict(event, mode_tag=mode_tag)
        matched = matched_safety_keywords(event)
        if matched:
            keywords = ", ".join(f"'{keyword}'" for keyword in matched)
            note = f"Safety net matched {keywords} and forced urgent."
            if urgency == "urgent":
                rationale = f"{note} {rationale}"
            else:
                # A model that said "routine" would contradict the escalation, so replace it.
                rationale = f"{note} {SAFETY_NET_RATIONALE}"
            urgency = "urgent"
        return Classification(urgency=urgency, rationale=rationale.strip())

    def _model_verdict(self, event: Event, *, mode_tag: str | None) -> tuple[Urgency, str]:
        """Ask the model once (one internal repair) or return the canned classification."""
        try:
            payload = self.client.complete_json(
                classify_messages(event),
                SCHEMA_HINT,
                phi=self.phi,
                max_tokens=CLASSIFY_MAX_TOKENS,
                enable_thinking=CLASSIFY_ENABLE_THINKING,
                mode_tag=mode_tag,
            )
        except Exception:  # transport, timeout and JSON failures all degrade the same way
            return CANNED_FALLBACK_URGENCY, CANNED_FALLBACK_RATIONALE
        return normalize_payload(payload)


def classify_event(
    event: Event,
    *,
    client: JsonCompletionClient | None = None,
    phi: bool | None = None,
    mode_tag: str | None = None,
) -> Classification:
    """Classify one event, building a default client when none is supplied."""
    return EventClassifier(client=client, phi=phi).classify_event(event, mode_tag=mode_tag)


def classify_messages(event: Event) -> list[dict[str, str]]:
    """Render the event as a classification prompt for the owned model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify the urgency of this overnight care event.\n"
                f"Time: {event.timestamp.isoformat()}\n"
                f"Source: {event.source}\n"
                f"Kind: {event.kind}\n"
                f"Text: {event.text}"
            ),
        },
    ]


def matched_safety_keywords(event: Event) -> list[str]:
    """Return the safety-net keywords present in the event's kind or text."""
    haystack = f"{event.kind.replace('_', ' ')} {event.text}".casefold().replace("\u2019", "'")
    return [keyword for keyword in SAFETY_KEYWORDS if keyword in haystack]


def normalize_payload(payload: Any) -> tuple[Urgency, str]:
    """Coerce a model payload into a valid label plus a non-empty rationale."""
    label = ""
    rationale = ""
    if isinstance(payload, dict):
        raw_label = payload.get("urgency")
        raw_rationale = payload.get("rationale")
        if isinstance(raw_label, str):
            label = raw_label.strip().casefold()
        if isinstance(raw_rationale, str):
            rationale = raw_rationale.strip()

    for candidate in VALID_URGENCIES:
        if label == candidate:
            return candidate, rationale or DEFAULT_RATIONALE
    return CANNED_FALLBACK_URGENCY, UNRECOGNIZED_LABEL_RATIONALE
