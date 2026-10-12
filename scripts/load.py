"""The load simulator CLI: drive the mock curve via the app, or fire real load on site.

Mock mode (`LLM_BASE_URL=mock` — the only supported path tonight) POSTs
`/api/simulate-load` on the running app, which ramps the synthetic metrics curve
so the policy walks NORMAL → CONSERVE → SURGE and back. No inference endpoint is
touched: exactly one localhost request, zero completions.

Real mode (the on-site env swap only) holds `--n` concurrent workers issuing
non-PHI dummy completions against the configured endpoint for `--duration`
seconds. The branch is a pure function of the settings (`decide_branch`), so the
real path is unreachable while the endpoint is mock.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx

if TYPE_CHECKING:  # type-only: never imported at runtime (see the bootstrap below)
    from config import Settings

# Direct-file execution (`python scripts/load.py`) starts with only scripts/ on
# sys.path; the flat layout needs the repo root importable (`config`, `core`, ...).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_DURATION_S",
    "DEFAULT_INTENSITY",
    "app_base_url",
    "build_parser",
    "build_real_client",
    "decide_branch",
    "drive_mock_curve",
    "fire_concurrent_completions",
    "main",
    "run_load",
]

#: CLI defaults (mirroring the app's `LoadRequest` and its real-load concurrency).
DEFAULT_DURATION_S = 30.0
DEFAULT_INTENSITY = 1.0
DEFAULT_CONCURRENCY = 8

#: The mock branch makes exactly one request; give it a short, definite timeout.
HTTP_TIMEOUT_S = 5.0
SIMULATE_PATH = "/api/simulate-load"
APP_START_HINT = "uv run uvicorn app.app:app --host 127.0.0.1 --port 8765"

#: Real-branch probe shape (never exercised in mock mode).
REAL_LOAD_MAX_TOKENS = 64
REAL_LOAD_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Generate a short neutral status line for a load test."}
]

MOCK_BRANCH = "app"
REAL_BRANCH = "endpoint"


class CompletionClient(Protocol):
    """The completion surface the real branch needs (LLMClient or its guard fit)."""

    def complete(
        self, messages: list[dict[str, str]], *, phi: bool, max_tokens: int
    ) -> Any: ...


def decide_branch(settings: Settings) -> str:
    """'app' while the LLM endpoint is mock (no real inference exists to load)."""
    return MOCK_BRANCH if settings.is_mock_llm else REAL_BRANCH


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {raw!r}")
    return value


def _non_negative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw!r}")
    return value


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The `load.py` CLI: `--duration`, `--intensity`, `--n` (plus the app address)."""
    parser = argparse.ArgumentParser(
        prog="load.py",
        description=(
            "Simulate inference load: the mock curve via the app's /api/simulate-load "
            "(mock mode), or N concurrent non-PHI completions against the real endpoint."
        ),
    )
    parser.add_argument(
        "--duration", type=_positive_float, default=DEFAULT_DURATION_S,
        help="load window in seconds (default: 30)",
    )
    parser.add_argument(
        "--intensity", type=_non_negative_float, default=DEFAULT_INTENSITY,
        help="load intensity scale (default: 1.0)",
    )
    parser.add_argument(
        "--n", type=_positive_int, default=DEFAULT_CONCURRENCY,
        help="concurrent workers for the real-endpoint branch (default: 8)",
    )
    parser.add_argument(
        "--url", default=None,
        help="app base URL for the mock branch (default: http://127.0.0.1:$PORT)",
    )
    return parser


def app_base_url(settings: Settings, url: str | None = None) -> str:
    """Where the app is expected: localhost, on the configured port, unless told else."""
    return (url or f"http://127.0.0.1:{settings.port}").rstrip("/")


def drive_mock_curve(
    http: httpx.Client,
    base_url: str,
    *,
    duration_s: float,
    intensity: float,
    n: int,
) -> int:
    """POST `/api/simulate-load` and report the accepted window (the only mock path)."""
    response = http.post(
        f"{base_url}{SIMULATE_PATH}", json={"duration_s": duration_s, "intensity": intensity}
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("started"):
        print(f"load: the app did not accept the load: {body}", file=sys.stderr)
        return 1
    accepted_duration = body.get("duration_s", duration_s)
    accepted_intensity = body.get("intensity", intensity)
    print(
        "load: simulate-load accepted by the app (mock curve) — "
        f"duration {accepted_duration:.1f}s, intensity {accepted_intensity}"
    )
    print(
        f"load: --n {n} applies to the real-endpoint branch only "
        "(unused in mock mode: the synthetic curve carries the load)"
    )
    return 0


async def fire_concurrent_completions(
    client: CompletionClient,
    *,
    n: int,
    duration_s: float,
    max_tokens: int = REAL_LOAD_MAX_TOKENS,
) -> None:
    """Hold `n` workers issuing non-PHI dummy completions until the window closes."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_s

    async def worker() -> None:
        while loop.time() < deadline:
            with contextlib.suppress(Exception):  # load probing must never crash the CLI
                await asyncio.to_thread(
                    client.complete, REAL_LOAD_MESSAGES, phi=False, max_tokens=max_tokens
                )

    await asyncio.gather(*(worker() for _ in range(max(1, n))))


def build_real_client(settings: Settings) -> CompletionClient:
    """The guarded completion client for the real branch (never built in mock mode)."""
    from core.egress_guard import EgressGuard
    from core.llm import LLMClient

    return EgressGuard(LLMClient(), settings.egress_allowlist)


def run_load(
    settings: Settings,
    *,
    duration_s: float,
    intensity: float,
    n: int,
    url: str | None = None,
    http: httpx.Client | None = None,
    completion_client: CompletionClient | None = None,
) -> int:
    """Execute the decided branch (`decide_branch`) and return the exit code."""
    if decide_branch(settings) == MOCK_BRANCH:
        owns_client = http is None
        client = http if http is not None else httpx.Client(timeout=HTTP_TIMEOUT_S)
        try:
            return drive_mock_curve(
                client,
                app_base_url(settings, url),
                duration_s=duration_s,
                intensity=intensity,
                n=n,
            )
        finally:
            if owns_client:
                client.close()

    client = completion_client if completion_client is not None else build_real_client(settings)
    print(f"load: real branch — {n} concurrent workers for {duration_s:.1f}s (non-PHI probes)")
    asyncio.run(fire_concurrent_completions(client, n=n, duration_s=duration_s))
    print("load: done")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    http: httpx.Client | None = None,
    completion_client: CompletionClient | None = None,
) -> int:
    """CLI entry point: returns the process exit code (0 on a reported success)."""
    from config import load_settings

    args = build_parser().parse_args(argv)
    resolved = settings if settings is not None else load_settings()
    try:
        return run_load(
            resolved,
            duration_s=args.duration,
            intensity=args.intensity,
            n=args.n,
            url=args.url,
            http=http,
            completion_client=completion_client,
        )
    except (httpx.HTTPError, ValueError) as exc:
        print(f"load: cannot reach the Night Watch app — {exc}", file=sys.stderr)
        print(f"load: start it first with: {APP_START_HINT}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
