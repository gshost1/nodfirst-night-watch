"""Offline contract tests for the overnight event classifier.

Covers VAL-CLASSIFY-001 (structured JSON, one repair retry, deterministic canned
fallback) and VAL-CLASSIFY-002 (the keyword safety net forces urgent, including
the 03:12 possible-fall event end-to-end in mock mode).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

import core.llm as core_llm
from config import Config
from core.llm import LLMClient, LLMJsonError
from nightwatch import classify as classify_module
from nightwatch.classify import (
    CLASSIFY_CONTEXT_ITEMS,
    CLASSIFY_ENABLE_THINKING,
    CLASSIFY_MAX_TOKENS,
    SAFETY_KEYWORDS,
    Classification,
    classify_event,
)
from nightwatch.dataset import Event, generate_events

#: Roster copied from architecture.md §nightwatch/classify.py — the spec's list.
ARCHITECTURE_KEYWORDS: tuple[str, ...] = (
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


def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "mock")
    monkeypatch.setenv("LLM_API_KEY", "EMPTY")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def _event(text: str, *, kind: str = "mood_check", source: str = "smart_speaker") -> Event:
    return Event(
        id="event-test",
        timestamp=datetime(2026, 10, 11, 22, 0),
        source=source,
        kind=kind,
        text=text,
    )


class RecordingClient:
    """Duck-typed client that records the exact completion request and returns a scripted dict."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = (
            payload
            if payload is not None
            else {"urgency": "routine", "rationale": "Quiet evening."}
        )
        self.error = error
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "messages": messages,
                "schema_hint": schema_hint,
                "phi": phi,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "mode_tag": mode_tag,
            }
        )
        if self.error is not None:
            raise self.error
        return self.payload


class FakeCompletions:
    """Stands in for `client.chat.completions` on the OpenAI SDK surface."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=None,
            model="response-model",
        )


def _http_client_with_responses(
    monkeypatch: pytest.MonkeyPatch, contents: list[str]
) -> tuple[LLMClient, FakeCompletions]:
    """A real LLMClient wired to a scripted (non-mock) transport."""
    _mock_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9998/v1")
    completions = FakeCompletions(contents)
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("core.llm.OpenAI", lambda **_: sdk_client)
    return LLMClient(), completions


# --- VAL-CLASSIFY-001: structured JSON, fixed params, repair, canned fallback ---


def test_classify_event_returns_valid_label_and_nonempty_rationale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env(monkeypatch)
    result = classify_event(_event("Mom says she is comfortable and ready for bed."))
    assert isinstance(result, Classification)
    assert result.urgency in {"urgent", "attention", "routine"}
    assert result.rationale.strip()


def test_classification_uses_fixed_mode_independent_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    classify_event(_event("Mom says she is resting comfortably."), client=client, mode_tag="SURGE")
    call = client.calls[0]
    assert call["max_tokens"] == CLASSIFY_MAX_TOKENS == 256
    assert call["enable_thinking"] is CLASSIFY_ENABLE_THINKING is False
    assert CLASSIFY_CONTEXT_ITEMS == 6
    assert call["mode_tag"] == "SURGE"

    relaxed = RecordingClient()
    classify_event(_event("Mom says she is resting comfortably."), client=relaxed, mode_tag=None)
    assert relaxed.calls[0]["max_tokens"] == call["max_tokens"]
    assert relaxed.calls[0]["enable_thinking"] == call["enable_thinking"]


def test_classification_sends_event_details_schema_hint_and_phi_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    event = _event("Mom asks for tea with breakfast.", kind="routine_chatter")
    classify_event(event, client=client)
    call = client.calls[0]
    hint = call["schema_hint"]
    hint_text = hint if isinstance(hint, str) else str(hint)
    assert "urgency" in hint_text and "rationale" in hint_text
    assert call["phi"] is Config.phi_mode
    prompt_text = "\n".join(message["content"] for message in call["messages"])
    assert event.text in prompt_text
    assert event.kind in prompt_text

    non_phi = RecordingClient()
    classify_event(event, client=non_phi, phi=False)
    assert non_phi.calls[0]["phi"] is False


def test_unparseable_first_reply_is_repaired_once_then_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, completions = _http_client_with_responses(
        monkeypatch,
        [
            "I am not able to answer that.",
            'Prose first. {"urgency": "attention", "rationale": "Worth a caregiver check."} done.',
        ],
    )
    result = classify_event(_event("Mom says she is comfortable."), client=client)
    assert result.urgency == "attention"
    assert result.rationale == "Worth a caregiver check."
    assert len(completions.calls) == 2
    repair_message = completions.calls[1]["messages"][-1]
    assert "Return ONLY valid JSON, no prose" in repair_message["content"]
    assert completions.calls[0]["max_tokens"] == 256
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.parametrize(
    "error",
    [LLMJsonError("no json after one repair"), RuntimeError("transport exploded")],
)
def test_total_model_failure_returns_canned_classification(error: Exception) -> None:
    first = classify_event(
        _event("Mom says she is comfortable."), client=RecordingClient(error=error)
    )
    second = classify_event(
        _event("Mom says she is comfortable."), client=RecordingClient(error=error)
    )
    assert first == second
    assert first.urgency == classify_module.CANNED_FALLBACK_URGENCY
    assert first.urgency in {"urgent", "attention", "routine"}
    assert first.rationale.strip()


def test_unrecognized_label_and_blank_rationale_still_yield_valid_classification() -> None:
    unknown = classify_event(
        _event("Mom says she is comfortable."),
        client=RecordingClient({"urgency": "high", "rationale": "seems fine"}),
    )
    assert unknown.urgency in {"urgent", "attention", "routine"}
    assert unknown.rationale.strip()

    blank = classify_event(
        _event("Mom says she is comfortable."),
        client=RecordingClient({"urgency": "URGENT", "rationale": "   "}),
    )
    assert blank.urgency == "urgent"
    assert blank.rationale.strip()


# --- VAL-CLASSIFY-002: the keyword safety net ---


def test_safety_keyword_roster_matches_the_architecture_list() -> None:
    assert SAFETY_KEYWORDS == ARCHITECTURE_KEYWORDS
    # The mock provider sniffs the same roster; keep the two lists from drifting apart.
    assert set(core_llm._SAFETY_KEYWORDS) == set(SAFETY_KEYWORDS)


@pytest.mark.parametrize("keyword", ARCHITECTURE_KEYWORDS)
def test_every_safety_keyword_forces_urgent_over_a_routine_model_reply(keyword: str) -> None:
    client = RecordingClient({"urgency": "routine", "rationale": "Quiet evening."})
    result = classify_event(_event(f"Overnight note: {keyword} observed."), client=client)
    assert result.urgency == "urgent"
    assert result.rationale.strip()
    assert keyword in result.rationale.lower()
    assert client.calls[0]["max_tokens"] == 256


def test_safety_net_is_case_insensitive_and_handles_curly_apostrophes() -> None:
    upper = classify_event(
        _event("Mom FELL and there was NO RESPONSE when we called."),
        client=RecordingClient({"urgency": "routine", "rationale": "Quiet evening."}),
    )
    assert upper.urgency == "urgent"

    curly = classify_event(
        _event("Mom says she can\u2019t breathe right now."),
        client=RecordingClient({"urgency": "routine", "rationale": "Quiet evening."}),
    )
    assert curly.urgency == "urgent"


def test_safety_net_forces_urgent_on_the_canned_fallback_path() -> None:
    result = classify_event(
        _event("Bathroom sensor reported a loud thud."),
        client=RecordingClient(error=RuntimeError("transport exploded")),
    )
    assert result.urgency == "urgent"
    assert "thud" in result.rationale.lower()


def test_safety_net_replaces_a_contradictory_routine_rationale() -> None:
    override = classify_event(
        _event("Mom says the hallway is dark after a fall scare."),
        client=RecordingClient({"urgency": "routine", "rationale": "Quiet evening."}),
    )
    assert override.urgency == "urgent"
    assert "Quiet evening" not in override.rationale

    agree = classify_event(
        _event("Mom says the hallway is dark after a fall scare."),
        client=RecordingClient({"urgency": "urgent", "rationale": "Check on Mom now."}),
    )
    assert agree.urgency == "urgent"
    assert "Check on Mom now." in agree.rationale


def test_missed_dose_event_is_urgent_via_its_kind() -> None:
    med_event = next(event for event in generate_events(seed=42) if event.kind == "missed_dose")
    assert (med_event.timestamp.hour, med_event.timestamp.minute) == (21, 15)
    result = classify_event(
        med_event, client=RecordingClient({"urgency": "routine", "rationale": "Quiet evening."})
    )
    assert result.urgency == "urgent"


def test_fall_event_classifies_urgent_end_to_end_in_mock_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env(monkeypatch)
    client = LLMClient()
    assert client.is_mock
    fall_event = next(event for event in generate_events(seed=42) if event.kind == "possible_fall")
    assert (fall_event.timestamp.hour, fall_event.timestamp.minute) == (3, 12)
    result = classify_event(fall_event, client=client)
    assert result.urgency == "urgent"
    assert result.rationale.strip()


def test_routine_event_in_mock_mode_is_not_escalated(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    result = classify_event(
        _event("Mom says she is comfortable and in a good mood."),
        client=LLMClient(),
        phi=Config.phi_mode,
    )
    assert result.urgency == "routine"
