"""Offline contract tests for the environment-driven LLM client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from core.llm import LLMClient, LLMJsonError


def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "mock")
    monkeypatch.setenv("LLM_API_KEY", "EMPTY")
    monkeypatch.setenv("LLM_MODEL", "test-model")


class FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content)),
            ],
            usage=None,
            model="response-model",
        )


def _client_with_responses(
    monkeypatch: pytest.MonkeyPatch, contents: list[str]
) -> tuple[LLMClient, FakeCompletions]:
    _mock_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9998/v1")
    completions = FakeCompletions(contents)
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("core.llm.OpenAI", lambda **_: sdk_client)
    return LLMClient(), completions


def test_client_uses_current_env_config_and_sdk_timeout_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "changed-model")
    monkeypatch.setattr("core.llm.OpenAI", fake_openai)

    client = LLMClient()

    assert client.base_url == "http://localhost:9999/v1"
    assert client.api_key == "test-key"
    assert client.model == "changed-model"
    assert captured["base_url"] == client.base_url
    assert captured["api_key"] == client.api_key
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 2


def test_mock_completion_is_deterministic_and_never_uses_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "MoCk")
    send_calls: list[httpx.Request] = []

    def record_send(_client: httpx.Client, request: httpx.Request, **_: Any) -> Any:
        send_calls.append(request)
        raise AssertionError("mock mode must not send HTTP requests")

    monkeypatch.setattr(httpx.Client, "send", record_send)
    client = LLMClient()
    messages = [{"role": "user", "content": "How can I reassure Mom after a quiet evening?"}]

    first = client.complete(messages, phi=True, max_tokens=80)
    second = client.complete(messages, phi=True, max_tokens=80)

    assert first == second
    assert first.content == (
        "I’m here to help care for Mom. A calm check-in and a note for the family "
        "can make the next step feel manageable."
    )
    assert first.latency_ms == 0
    assert first.model == "test-model"
    assert send_calls == []


def test_mock_classification_and_safety_keyword_replies_are_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env(monkeypatch)
    client = LLMClient()

    ordinary = client.complete(
        [{"role": "user", "content": "Classify this event: Mom listened to music."}],
        phi=True,
        max_tokens=100,
    )
    urgent = client.complete(
        [{"role": "user", "content": "Classify event: Mom missed dose of metformin."}],
        phi=True,
        max_tokens=100,
    )

    ordinary_json = json.loads(ordinary.content)
    assert ordinary_json["urgency"] in {"urgent", "attention", "routine"}
    assert ordinary_json["rationale"]
    urgent_json = json.loads(urgent.content)
    assert urgent_json["urgency"] == "urgent"
    assert urgent_json["rationale"]


def test_mock_summary_mentions_input_and_default_is_caregiver_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_env(monkeypatch)
    client = LLMClient()
    summary_prompt = [
        {
            "role": "user",
            "content": "Summarize this check-in for Anna: Mom had two bathroom trips.",
        }
    ]

    summary = client.complete(summary_prompt, phi=True, max_tokens=100)
    fall_summary = client.complete(
        [{"role": "user", "content": "Summarize check-in: Mom had a loud thud and no response."}],
        phi=True,
        max_tokens=100,
    )
    default = client.complete(
        [{"role": "user", "content": "What should I do next?"}],
        phi=True,
        max_tokens=100,
    )

    assert "Mom had two bathroom trips" in summary.content
    assert "family" in summary.content.lower()
    assert "warm note" in fall_summary.content.lower()
    assert "loud thud and no response" in fall_summary.content
    assert "care" in default.content.lower()


def test_complete_json_extracts_first_balanced_object_and_strips_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = (
        "<think>private thought {not json}</think>Here is the result:\n"
        "```json\n{\"first\": {\"text\": \"a } brace\"}}\n```\n"
        "{\"ignored\": true}<think>unfinished private thought"
    )
    client, completions = _client_with_responses(monkeypatch, [response])

    result = client.complete_json(
        [{"role": "user", "content": "Return one JSON object."}],
        schema_hint="an object with a first property",
        phi=True,
        max_tokens=100,
    )

    assert result == {"first": {"text": "a } brace"}}
    assert len(completions.calls) == 1
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_complete_json_repairs_once_then_raises_llm_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, completions = _client_with_responses(monkeypatch, ["not json", "still not json"])

    with pytest.raises(LLMJsonError):
        client.complete_json(
            [{"role": "user", "content": "Return a value."}],
            schema_hint="an object with a value property",
            phi=True,
            max_tokens=80,
        )

    assert len(completions.calls) == 2
    assert completions.calls[1]["messages"][-1] == {
        "role": "user",
        "content": "Return ONLY valid JSON, no prose",
    }
    for call in completions.calls:
        assert call["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


def test_complete_json_returns_valid_json_from_single_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, completions = _client_with_responses(
        monkeypatch, ["the first response is broken", "Repaired: {\"urgency\": \"attention\"}"]
    )

    result = client.complete_json(
        [{"role": "user", "content": "Classify the check-in."}],
        schema_hint={"urgency": "urgent, attention, or routine"},
        phi=True,
        max_tokens=80,
    )

    assert result == {"urgency": "attention"}
    assert len(completions.calls) == 2
    assert completions.calls[1]["messages"][-1]["content"] == (
        "Return ONLY valid JSON, no prose"
    )


def test_complete_strips_closed_and_unterminated_think_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, completions = _client_with_responses(
        monkeypatch,
        ["before <think>hidden</think> after <think>unfinished hidden"],
    )

    result = client.complete(
        [{"role": "user", "content": "Reply briefly."}],
        phi=False,
        max_tokens=60,
        enable_thinking=True,
        mode_tag="NORMAL",
    )

    assert result.content == "before  after"
    assert result.mode_tag == "NORMAL"
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
