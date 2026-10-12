"""Offline contract tests for the PHI egress guard (the privacy core)."""

from __future__ import annotations

import socket
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from config import Config
from core.egress_guard import EgressBlocked, EgressGuard, resolve_host
from core.llm import LLMClient, LLMResult

ALLOWED_HOST = "gpu.nodfirst.internal"
THIRD_PARTY_BASE_URL = "https://api.openai.com/v1"


class FakeCompletions:
    """Spy SDK completions; any call proves transport work was attempted."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"urgency": "routine"}'))],
            usage=None,
            model="response-model",
        )


def _real_client(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> tuple[LLMClient, FakeCompletions]:
    """An LLMClient whose OpenAI SDK is replaced by a spy (no sockets involved)."""
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    monkeypatch.setenv("LLM_API_KEY", "EMPTY")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    completions = FakeCompletions()
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("core.llm.OpenAI", lambda **_: sdk_client)
    return LLMClient(), completions


class RecordingClient:
    """Minimal stand-in for LLMClient that records every delegated call."""

    def __init__(self, base_url: str) -> None:
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
        self.calls.append({"method": "complete", "phi": phi, "max_tokens": max_tokens})
        return LLMResult(
            content="delegated",
            latency_ms=0,
            prompt_tokens=1,
            completion_tokens=1,
            model="stub",
            mode_tag=mode_tag,
        )

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
        self.calls.append({"method": "complete_json", "phi": phi, "max_tokens": max_tokens})
        return {"urgency": "routine"}


def test_resolve_host_extracts_lowercased_host_and_mock_sentinel() -> None:
    assert resolve_host("https://GPU.NodFirst.internal:8000/v1") == ALLOWED_HOST
    assert resolve_host("http://localhost:9998/v1") == "localhost"
    assert resolve_host("MoCk") == "mock"
    assert resolve_host("mock") == "mock"


def test_phi_to_non_allowlisted_host_raises_before_any_network_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, completions = _real_client(monkeypatch, THIRD_PARTY_BASE_URL)
    sends: list[httpx.Request] = []
    connects: list[Any] = []
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: sends.append(a))
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: connects.append(addr))
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}))

    with pytest.raises(EgressBlocked) as excinfo:
        guard.complete(
            [{"role": "user", "content": "Mom missed her metformin"}], phi=True, max_tokens=64
        )

    assert excinfo.value.host == "api.openai.com"
    assert excinfo.value.allowlist == frozenset({ALLOWED_HOST})
    assert "api.openai.com" in str(excinfo.value)
    assert completions.calls == []
    assert sends == []
    assert connects == []


def test_phi_complete_json_to_non_allowlisted_host_raises_before_any_network_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, completions = _real_client(monkeypatch, THIRD_PARTY_BASE_URL)
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}))

    with pytest.raises(EgressBlocked):
        guard.complete_json(
            [{"role": "user", "content": "classify this event"}],
            {"urgency": "urgent|attention|routine"},
            phi=True,
            max_tokens=256,
        )

    assert completions.calls == []


def test_allowlisted_host_phi_passes_through_with_exactly_one_call() -> None:
    client = RecordingClient("http://gpu.nodfirst.internal:8000/v1")
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}))

    result = guard.complete([{"role": "user", "content": "hi"}], phi=True, max_tokens=64)

    assert result.content == "delegated"
    assert len(client.calls) == 1
    assert client.calls[0] == {"method": "complete", "phi": True, "max_tokens": 64}


def test_allowlisted_host_complete_json_passes_through_with_exactly_one_call() -> None:
    client = RecordingClient("http://gpu.nodfirst.internal:8000/v1")
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}))

    result = guard.complete_json(
        [{"role": "user", "content": "x"}], "{}", phi=True, max_tokens=32
    )

    assert result == {"urgency": "routine"}
    assert len(client.calls) == 1
    assert client.calls[0]["method"] == "complete_json"


def test_non_phi_payload_to_non_allowlisted_host_is_not_blocked() -> None:
    client = RecordingClient(THIRD_PARTY_BASE_URL)
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}))

    result = guard.complete([{"role": "user", "content": "no PHI here"}], phi=False, max_tokens=64)

    assert result.content == "delegated"
    assert len(client.calls) == 1
    assert client.calls[0]["phi"] is False


@pytest.mark.parametrize("phi", [True, False])
def test_mock_endpoint_never_blocked_regardless_of_phi(phi: bool) -> None:
    client = RecordingClient("MoCk")
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}))

    result = guard.complete([{"role": "user", "content": "x"}], phi=phi, max_tokens=16)
    assert result.content == "delegated"
    payload = guard.complete_json(
        [{"role": "user", "content": "x"}], "{}", phi=phi, max_tokens=16
    )
    assert payload == {"urgency": "routine"}
    assert len(client.calls) == 2


def test_egress_state_mock_endpoint() -> None:
    guard = EgressGuard(RecordingClient("mock"), frozenset({ALLOWED_HOST}), phi_mode=True)

    assert guard.egress_state() == {
        "endpoint_host": "mock",
        "is_mock": True,
        "allowlist": [ALLOWED_HOST],
        "phi_mode": True,
        "owned": False,
    }


def test_egress_state_allowlisted_vllm_host_is_owned() -> None:
    guard = EgressGuard(
        RecordingClient("http://gpu.nodfirst.internal:8000/v1"), frozenset({ALLOWED_HOST})
    )

    state = guard.egress_state()

    assert state["endpoint_host"] == ALLOWED_HOST
    assert state["is_mock"] is False
    assert state["owned"] is True
    assert state["allowlist"] == [ALLOWED_HOST]


def test_egress_state_third_party_host_is_not_owned() -> None:
    guard = EgressGuard(RecordingClient(THIRD_PARTY_BASE_URL), frozenset({ALLOWED_HOST}))

    state = guard.egress_state()

    assert state == {
        "endpoint_host": "api.openai.com",
        "is_mock": False,
        "allowlist": [ALLOWED_HOST],
        "phi_mode": Config.phi_mode,
        "owned": False,
    }
    assert set(state) == {"endpoint_host", "is_mock", "allowlist", "phi_mode", "owned"}


def test_egress_state_allowlist_is_sorted_and_lowercased() -> None:
    guard = EgressGuard(RecordingClient("mock"), frozenset({"B.Host", "a.host"}), phi_mode=False)

    assert guard.egress_state()["allowlist"] == ["a.host", "b.host"]
    assert guard.egress_state()["phi_mode"] is False


def test_phi_blocking_follows_the_per_call_flag_not_config_phi_mode() -> None:
    client = RecordingClient(THIRD_PARTY_BASE_URL)
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}), phi_mode=False)

    with pytest.raises(EgressBlocked):
        guard.complete([{"role": "user", "content": "x"}], phi=True, max_tokens=16)


def test_base_url_override_supports_the_preflight_demo_without_io() -> None:
    client = RecordingClient("mock")
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}), base_url=THIRD_PARTY_BASE_URL)

    with pytest.raises(EgressBlocked) as excinfo:
        guard.complete([{"role": "user", "content": "PHI"}], phi=True, max_tokens=64)

    assert excinfo.value.host == "api.openai.com"
    assert client.calls == []
    assert guard.egress_state()["endpoint_host"] == "api.openai.com"


def test_egress_blocked_as_dict_reports_host_allowlist_and_message() -> None:
    client = RecordingClient(THIRD_PARTY_BASE_URL)
    guard = EgressGuard(client, frozenset({ALLOWED_HOST}))

    with pytest.raises(EgressBlocked) as excinfo:
        guard.complete([{"role": "user", "content": "x"}], phi=True, max_tokens=16)

    details = excinfo.value.as_dict()
    assert details["host"] == "api.openai.com"
    assert details["allowlist"] == [ALLOWED_HOST]
    assert "0 bytes" in details["message"]
