"""PHI egress guard — the privacy core: PHI-tagged payloads stay on owned hosts.

`EgressGuard` wraps an LLM client (`core.llm.LLMClient`) and exposes the same
`complete` / `complete_json` surface. Every call is checked first: when `phi=True`
and the client's resolved `base_url` host is not in the allowlist, `EgressBlocked`
is raised *before* the wrapped client is touched, so zero bytes leave the machine.

The `mock` base_url is always allowed — no egress is possible by construction — and
`egress_state()` reports it as `is_mock=True`, `owned=False`.

Enforcement always follows the **per-call** `phi` argument. `PHI_MODE` from `config`
is informational: it is what callers (e.g. the pipeline) use to decide the `phi`
argument, and it is the flag the dashboard badge renders, but a `phi=True` call is
blocked even when `PHI_MODE` is off.

For pre-flight checks that must never reach a network — the dashboard's
"Attempt PHI Egress" demo — pass `base_url="https://api.openai.com/v1"` to the
constructor. The guard resolves that host and raises before any I/O; when calls are
expected to pass through instead, `base_url` must match the client's real endpoint.

Usage: `guard = EgressGuard(LLMClient(), Config.egress_allowlist)`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol
from urllib.parse import urlsplit

from config import MOCK_ENDPOINT, Config

__all__ = ["EgressBlocked", "EgressGuard", "CompletionClient", "resolve_host"]


class CompletionClient(Protocol):
    """The LLM client surface the guard wraps and delegates to."""

    base_url: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> Any: ...

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema_hint: str | dict[str, Any],
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> dict[str, Any]: ...


def resolve_host(base_url: str) -> str:
    """Lowercased hostname for a base_url.

    Returns ``"mock"`` for the offline endpoint and an empty string when the
    base_url has no parsable host (which then fails closed for PHI calls).
    """
    stripped = (base_url or "").strip()
    if stripped.lower() == MOCK_ENDPOINT:
        return MOCK_ENDPOINT
    return (urlsplit(stripped).hostname or "").lower()


def _normalize_hosts(allowlist: Iterable[str] | str) -> frozenset[str]:
    """Normalize an allowlist to a lowercased host set (accepts a comma-separated string)."""
    if isinstance(allowlist, str):
        entries: Sequence[str] = allowlist.split(",")
    else:
        entries = list(allowlist)
    return frozenset(entry.strip().lower() for entry in entries if entry and entry.strip())


class EgressBlocked(RuntimeError):
    """Raised when a PHI-tagged payload targets a host outside the egress allowlist."""

    def __init__(self, host: str, allowlist: Iterable[str] | str) -> None:
        self.host = host
        self.allowlist = _normalize_hosts(allowlist)
        super().__init__(
            f"EgressBlocked: {host or '<no host>'} not in egress allowlist "
            f"{sorted(self.allowlist)} — 0 bytes left this machine"
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready details for the API response that renders the blocked attempt."""
        return {
            "host": self.host,
            "allowlist": sorted(self.allowlist),
            "message": str(self),
        }


class EgressGuard:
    """Blocks PHI-tagged calls to non-allowlisted hosts before any network I/O."""

    def __init__(
        self,
        client: CompletionClient,
        allowlist: Iterable[str] | str,
        *,
        base_url: str | None = None,
        phi_mode: bool | None = None,
    ) -> None:
        self._client = client
        self.allowlist = _normalize_hosts(allowlist)
        self.base_url = client.base_url if base_url is None else base_url
        self.endpoint_host = resolve_host(self.base_url)
        self.is_mock = self.base_url.strip().lower() == MOCK_ENDPOINT
        self.phi_mode = Config.phi_mode if phi_mode is None else bool(phi_mode)

    @property
    def owned(self) -> bool:
        """True when PHI may legitimately target this host (real, allowlisted endpoint)."""
        return not self.is_mock and self.endpoint_host in self.allowlist

    def check(self, phi: bool) -> None:
        """Raise `EgressBlocked` when a PHI-tagged call would leave the allowlist."""
        if not phi or self.is_mock:
            return
        if self.endpoint_host not in self.allowlist:
            raise EgressBlocked(self.endpoint_host, self.allowlist)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        phi: bool,
        max_tokens: int,
        enable_thinking: bool = False,
        mode_tag: str | None = None,
    ) -> Any:
        """Guard, then delegate one completion to the wrapped client."""
        self.check(phi)
        return self._client.complete(
            messages,
            phi=phi,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
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
        """Guard, then delegate one JSON completion to the wrapped client."""
        self.check(phi)
        return self._client.complete_json(
            messages,
            schema_hint,
            phi=phi,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            mode_tag=mode_tag,
        )

    def egress_state(self) -> dict[str, Any]:
        """JSON-ready privacy state driving the dashboard badge and the /api/status payload."""
        return {
            "endpoint_host": self.endpoint_host,
            "is_mock": self.is_mock,
            "allowlist": sorted(self.allowlist),
            "phi_mode": self.phi_mode,
            "owned": self.owned,
        }
