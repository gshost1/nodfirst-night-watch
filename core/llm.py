"""Environment-driven OpenAI-compatible LLM client with an offline mock mode."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from config import MOCK_ENDPOINT, Settings, load_settings

_REPAIR_INSTRUCTION = "Return ONLY valid JSON, no prose"
_SAFETY_KEYWORDS = (
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
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?(?:</think\s*>|$)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Text and usage details returned by one completion."""

    content: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    model: str
    mode_tag: str | None = None


class LLMJsonError(ValueError):
    """Raised when a completion and its single repair attempt are not valid JSON."""


class LLMClient:
    """OpenAI-compatible completion client; configuration is refreshed per instance."""

    def __init__(self) -> None:
        self.settings: Settings = load_settings()
        self.base_url = self.settings.llm_base_url
        self.api_key = self.settings.llm_api_key
        self.model = self.settings.llm_model
        self.is_mock = self.base_url.lower() == MOCK_ENDPOINT
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=30,
            max_retries=2,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> LLMResult:
        """Generate a completion, using deterministic canned text for the mock endpoint."""
        del phi, enable_thinking  # Accepted by the shared interface; mock mode never sends data.
        start = time.perf_counter()

        if self.is_mock:
            content = _mock_reply(messages)
            return LLMResult(
                content=content,
                latency_ms=0,
                prompt_tokens=_estimate_tokens(_all_message_text(messages)),
                completion_tokens=_estimate_tokens(content),
                model=self.model,
                mode_tag=mode_tag,
            )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latency_ms = round((time.perf_counter() - start) * 1000)
        choice = response.choices[0]
        raw_content = _content_text(choice.message.content)
        content = _strip_think_blocks(raw_content)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        return LLMResult(
            content=content,
            latency_ms=latency_ms,
            prompt_tokens=(
                _estimate_tokens(_all_message_text(messages))
                if prompt_tokens is None
                else prompt_tokens
            ),
            completion_tokens=(
                _estimate_tokens(content) if completion_tokens is None else completion_tokens
            ),
            model=getattr(response, "model", None) or self.model,
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
        """Return the first JSON object in a completion, repairing invalid output once."""
        hint = schema_hint if isinstance(schema_hint, str) else json.dumps(schema_hint)
        prompted_messages = [
            {"role": "system", "content": f"Return a JSON object matching this schema: {hint}"},
            *messages,
        ]

        first = self.complete(
            prompted_messages,
            phi=phi,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            mode_tag=mode_tag,
        )
        try:
            return _parse_first_json_object(first.content)
        except (ValueError, TypeError):
            repair_messages = [
                *prompted_messages,
                {"role": "user", "content": _REPAIR_INSTRUCTION},
            ]

        repaired = self.complete(
            repair_messages,
            phi=phi,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            mode_tag=mode_tag,
        )
        try:
            return _parse_first_json_object(repaired.content)
        except (ValueError, TypeError) as repair_error:
            raise LLMJsonError(
                "model did not return a valid JSON object after one repair attempt"
            ) from repair_error


def _mock_reply(messages: list[dict[str, Any]]) -> str:
    """Choose a stable canned response based on the user's intent and salient input."""
    user_text = _user_message_text(messages)
    normalized = user_text.casefold().replace("’", "'")
    all_text = _all_message_text(messages).casefold()
    is_classification = "classif" in all_text or "urgency" in all_text
    has_safety_keyword = any(keyword in normalized for keyword in _SAFETY_KEYWORDS)
    if is_classification:
        if has_safety_keyword:
            return _urgent_classification()
        return json.dumps(
            {
                "urgency": "routine",
                "rationale": "No immediate safety concern was mentioned.",
            }
        )

    is_summary = any(word in all_text for word in ("summar", "digest", "check-in summary"))
    if is_summary:
        fragment = _salient_fragment(user_text)
        return (
            f"A warm note for the family: {fragment}. "
            "We can take the next step together."
        )
    if has_safety_keyword:
        return _urgent_classification()

    return (
        "I’m here to help care for Mom. A calm check-in and a note for the family "
        "can make the next step feel manageable."
    )


def _urgent_classification() -> str:
    """Render the canned safety-net result."""
    return json.dumps(
        {
            "urgency": "urgent",
            "rationale": "A safety concern needs prompt caregiver attention.",
        }
    )


def _salient_fragment(text: str) -> str:
    """Extract a compact source fragment from a summary or digest prompt."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    labeled = re.compile(r"^(?:event|input|check-in|message|notes)\s*:\s*(.+)$", re.IGNORECASE)
    for line in reversed(lines):
        match = labeled.match(line)
        if match:
            return _compact_fragment(match.group(1))

    if ":" in text:
        return _compact_fragment(text.split(":", maxsplit=1)[1])
    return _compact_fragment(text)


def _compact_fragment(text: str) -> str:
    """Normalize whitespace and keep canned summaries concise."""
    fragment = " ".join(text.split()).strip(" \t\r\n.-")
    return fragment[:180] or "Mom had a quiet overnight check-in"


def _user_message_text(messages: list[dict[str, Any]]) -> str:
    contents = [
        _content_text(message.get("content"))
        for message in messages
        if message.get("role") == "user"
    ]
    if contents:
        return "\n".join(contents)
    return _all_message_text(messages)


def _all_message_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_content_text(message.get("content")) for message in messages)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(fragments)
    return ""


def _strip_think_blocks(content: str) -> str:
    """Remove closed thinking blocks and discard any unterminated trailing block."""
    return _THINK_BLOCK.sub("", content).strip()


def _parse_first_json_object(content: str) -> dict[str, Any]:
    """Parse the first balanced JSON object, ignoring surrounding prose and fences."""
    start = content.find("{")
    if start < 0:
        raise ValueError("completion did not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(content[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("completion JSON must be an object")
                return value

    raise ValueError("completion contained an unbalanced JSON object")


def _estimate_tokens(text: str) -> int:
    """Provide a deterministic rough token count when usage is unavailable."""
    return (len(text) + 3) // 4
