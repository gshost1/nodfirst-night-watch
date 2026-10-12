"""The scripted console demo: one full Night Watch run, start to morning digest.

Prefers the running app (`POST /api/run`, then polls `/api/status`) and falls
back to an isolated in-process run when no app is listening — either way the
run is started, polled to completion, and its results printed: urgent alerts in
emit order, the morning digest (markdown), and the full call-log table. Mock
mode only: the app path talks to 127.0.0.1 alone and the in-process path makes
no requests at all, so the demo performs zero outbound inference calls.

Usage: `uv run python scripts/demo.py` (see `--help` for the knobs).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx

if TYPE_CHECKING:  # type-only: never imported at runtime (see the bootstrap below)
    from config import Settings

# Direct-file execution (`python scripts/demo.py`) starts with only scripts/ on
# sys.path; the flat layout needs the repo root importable (`config`, `core`, ...).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = [
    "ALERTS_TITLE",
    "CALLS_TITLE",
    "DIGEST_TITLE",
    "AppTransport",
    "InProcessTransport",
    "build_parser",
    "main",
    "probe_app",
]

#: Pacing defaults: match the app, so a run streams for ~10s (watchable on video).
DEFAULT_STEP_DELAY_S = 0.25
DEFAULT_POLL_INTERVAL_S = 0.5
DEFAULT_TIMEOUT_S = 120.0

PROBE_TIMEOUT_S = 1.5
HTTP_TIMEOUT_S = 5.0
APP_START_HINT = "uv run uvicorn app.app:app --host 127.0.0.1 --port 8765"

ALERTS_TITLE = "URGENT ALERTS (emit order)"
DIGEST_TITLE = "MORNING DIGEST"
CALLS_TITLE = "CALL LOG"


class DemoTransport(Protocol):
    """One uniform surface over the two ways a run can be started and read."""

    def header_lines(self) -> list[str]: ...
    def start(self) -> str: ...
    def status(self) -> dict[str, Any]: ...
    def alerts(self) -> list[dict[str, Any]]: ...
    def digest(self) -> dict[str, Any]: ...
    def calls(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


class AppTransport:
    """Start and read a run over the app's HTTP API (localhost only)."""

    def __init__(
        self, settings: Settings, base_url: str, http: httpx.Client, probe: dict[str, Any]
    ) -> None:
        self.settings = settings
        self.base_url = base_url
        self.http = http
        self.probe = probe

    def _get(self, path: str) -> Any:
        response = self.http.get(f"{self.base_url}{path}", timeout=HTTP_TIMEOUT_S)
        response.raise_for_status()
        return response.json()

    def header_lines(self) -> list[str]:
        flags = self.probe.get("flags", {})
        egress = self.probe.get("egress", {})
        allowlist = ", ".join(egress.get("allowlist", [])) or "none"
        return [
            f"transport: app API at {self.base_url}",
            f"config: mock LLM={flags.get('mock_llm')}, "
            f"mock metrics={flags.get('mock_metrics')}, PHI mode={flags.get('phi_mode')}",
            f"egress: {egress.get('endpoint_host')} — owned={egress.get('owned')} "
            f"(allowlist: {allowlist})",
        ]

    def start(self) -> str:
        response = self.http.post(f"{self.base_url}/api/run", timeout=HTTP_TIMEOUT_S)
        if response.status_code == 409:
            return "a night-watch run is already active — following it to completion"
        response.raise_for_status()
        run_id = response.json().get("run_id", "?")
        return f"started night-watch run {run_id}"

    def status(self) -> dict[str, Any]:
        return self._get("/api/status")

    def alerts(self) -> list[dict[str, Any]]:
        return self._get("/api/alerts")

    def digest(self) -> dict[str, Any]:
        return self._get("/api/digest")

    def calls(self) -> list[dict[str, Any]]:
        return self._get("/api/calls")

    def close(self) -> None:
        """The HTTP client belongs to `main` (or the caller); nothing to stop here."""


class InProcessTransport:
    """Run `NightWatchRun` directly in this process (no app server needed)."""

    def __init__(self, settings: Settings, *, step_delay_s: float) -> None:
        from core.egress_guard import EgressGuard
        from core.llm import LLMClient
        from inference.metrics import MockMetricsSource
        from nightwatch.pipeline import NightWatchRun

        self.settings = settings
        self._guard = EgressGuard(LLMClient(), settings.egress_allowlist)
        self._run = NightWatchRun(
            client=self._guard,
            metrics_source=MockMetricsSource(),
            settings=settings,
            step_delay_s=step_delay_s,
        )
        self._thread: threading.Thread | None = None

    def header_lines(self) -> list[str]:
        egress = self._guard.egress_state()
        allowlist = ", ".join(sorted(egress.get("allowlist", []))) or "none"
        return [
            "transport: in-process (no app server detected — running the pipeline directly)",
            f"config: mock LLM={self.settings.is_mock_llm}, "
            f"mock metrics={self.settings.is_mock_metrics}, PHI mode={self.settings.phi_mode}",
            f"egress: {egress.get('endpoint_host')} — owned={egress.get('owned')} "
            f"(allowlist: {allowlist})",
        ]

    def start(self) -> str:
        self._thread = threading.Thread(
            target=self._execute, name=f"nightwatch-demo-{self._run.run_id}", daemon=True
        )
        self._thread.start()
        return f"started night-watch run {self._run.run_id}"

    def _execute(self) -> None:
        with contextlib.suppress(Exception):  # the run records its own failure state
            self._run.execute()

    def status(self) -> dict[str, Any]:
        return self._run.status()

    def alerts(self) -> list[dict[str, Any]]:
        return self._run.alerts()

    def digest(self) -> dict[str, Any]:
        digest = self._run.digest
        if digest is None:
            return {"ready": False}
        return {"ready": True, **digest.to_dict()}

    def calls(self) -> list[dict[str, Any]]:
        return self._run.calls()

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class DemoError(RuntimeError):
    """A demo-blocking condition with a hint (the app is not running, a timeout, ...)."""


def _non_negative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {raw!r}")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The `demo.py` CLI: transport choice plus pacing knobs."""
    parser = argparse.ArgumentParser(
        prog="demo.py",
        description=(
            "Scripted console demo: start a Night Watch run (via the app when it is up, "
            "in-process otherwise), poll it to completion, then print the alerts, the "
            "morning digest, and the call-log table."
        ),
    )
    parser.add_argument(
        "--url", default=None,
        help="force the app API at this base URL (default: probe http://127.0.0.1:$PORT)",
    )
    parser.add_argument(
        "--in-process", action="store_true",
        help="skip the app and run the pipeline in this process",
    )
    parser.add_argument(
        "--step-delay", type=_non_negative_float, default=DEFAULT_STEP_DELAY_S,
        help="per-step pause for the in-process run, seconds (default: 0.25)",
    )
    parser.add_argument(
        "--poll-interval", type=_non_negative_float, default=DEFAULT_POLL_INTERVAL_S,
        help="seconds between status polls (default: 0.5)",
    )
    parser.add_argument(
        "--timeout", type=_positive_float, default=DEFAULT_TIMEOUT_S,
        help="give up if the run takes longer than this, seconds (default: 120)",
    )
    return parser


def probe_app(http: httpx.Client, base_url: str) -> dict[str, Any] | None:
    """The app's status payload when it is up, else `None` (never raises)."""
    try:
        response = http.get(f"{base_url}/api/status", timeout=PROBE_TIMEOUT_S)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def _resolve_transport(
    settings: Settings, args: argparse.Namespace, http: httpx.Client
) -> DemoTransport:
    """Pick the transport: forced in-process, forced app, or probed app-else-in-process."""
    base_url = (args.url or f"http://127.0.0.1:{settings.port}").rstrip("/")
    if args.in_process:
        return InProcessTransport(settings, step_delay_s=args.step_delay)
    probe = probe_app(http, base_url)
    if probe is None:
        if args.url is not None:
            raise DemoError(
                f"cannot reach the Night Watch app at {base_url} — "
                f"start it with: {APP_START_HINT}"
            )
        return InProcessTransport(settings, step_delay_s=args.step_delay)
    return AppTransport(settings, base_url, http, probe)


def _print_progress(number: int, status: dict[str, Any]) -> None:
    """One streaming progress line per poll tick."""
    processed = status.get("events_processed", 0)
    total = status.get("events_total", 0)
    mode = status.get("mode") or "?"
    reason = status.get("reason") or ""
    if len(reason) > 64:
        reason = reason[:61] + "..."
    print(f"  [poll {number:>2}] events {processed}/{total} | mode {mode} | {reason}")


def _print_alerts(alerts: list[dict[str, Any]]) -> None:
    print()
    print(ALERTS_TITLE)
    print("-" * len(ALERTS_TITLE))
    if not alerts:
        print("  (no urgent alerts)")
        return
    for number, alert in enumerate(alerts, start=1):
        when = alert.get("time") or "??:??"
        print(
            f"  {number}. {when} {alert.get('kind', 'urgent')} "
            f"({alert.get('source', 'unknown')})"
        )
        text = (alert.get("text") or "").strip()
        if text:
            print(f'     "{text}"')
        summary = (alert.get("summary") or "").strip()
        if summary:
            print(f"     -> {summary}")


def _print_digest(digest: dict[str, Any]) -> None:
    print()
    print(DIGEST_TITLE)
    print("-" * len(DIGEST_TITLE))
    if not digest.get("ready"):
        print("  (digest not ready)")
        return
    markdown = (digest.get("markdown") or "").strip()
    if markdown:
        print(markdown)
        return
    print(digest.get("header", ""))
    for task in digest.get("tasks", []):
        items = "; ".join(task.get("tasks", []))
        print(f"- {task.get('sibling', '?')}: {items}")


def _print_calls(calls: list[dict[str, Any]]) -> None:
    title = f"{CALLS_TITLE} ({len(calls)} calls)"
    print()
    print(title)
    print("-" * len(title))
    if not calls:
        print("  (no LLM calls logged)")
        return
    print(f"  {'#':>4}  {'step':<16} {'mode':<8} {'latency':>7}  {'tokens p/c':>10}  ok")
    for number, call in enumerate(calls, start=1):
        tokens = f"{call.get('prompt_tokens', 0)}/{call.get('completion_tokens', 0)}"
        state = "yes" if call.get("ok") else f"NO: {call.get('error') or 'failed'}"
        print(
            f"  {number:>4}  {call.get('step', '?'):<16} {call.get('mode', '?'):<8} "
            f"{call.get('latency_ms', 0):>5}ms  {tokens:>10}  {state}"
        )


def _print_failure(status: dict[str, Any], elapsed: float) -> None:
    print(f"run FAILED after {elapsed:.1f}s — {status.get('error', 'unknown error')}")


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    http: httpx.Client | None = None,
) -> int:
    """Run the whole demo; returns the process exit code (0 when the run completes)."""
    from config import load_settings

    args = build_parser().parse_args(argv)
    resolved = settings if settings is not None else load_settings()
    owns_client = http is None
    client = http if http is not None else httpx.Client(timeout=HTTP_TIMEOUT_S)

    print("Night Watch console demo — owned inference, end to end")
    try:
        transport = _resolve_transport(resolved, args, client)
    except DemoError as exc:
        print(f"demo: {exc}", file=sys.stderr)
        if owns_client:
            client.close()
        return 1

    for line in transport.header_lines():
        print(line)

    started = time.monotonic()
    try:
        print(f"run: {transport.start()}")
        poll_number = 0
        status: dict[str, Any] = {}
        state = "idle"
        while True:
            poll_number += 1
            status = transport.status()
            state = status.get("run_state", "idle")
            if state in {"done", "failed"}:
                break
            if time.monotonic() - started > args.timeout:
                print(
                    f"demo: timed out after {args.timeout:.0f}s in state {state!r} — "
                    f"start it with: {APP_START_HINT}",
                    file=sys.stderr,
                )
                return 1
            _print_progress(poll_number, status)
            time.sleep(args.poll_interval)
        alerts = transport.alerts()
        digest = transport.digest()
        call_entries = transport.calls()
    except httpx.HTTPError as exc:
        print(f"demo: app request failed — {exc}", file=sys.stderr)
        print(f"demo: start the app with: {APP_START_HINT}", file=sys.stderr)
        return 1
    finally:
        transport.close()
        if owns_client:
            client.close()

    elapsed = time.monotonic() - started
    if state == "failed":
        _print_failure(status, elapsed)
        return 1

    _print_alerts(alerts)
    _print_digest(digest)
    _print_calls(call_entries)
    print()
    print(
        f"run finished: state={state}, events {status.get('events_processed', 0)}/"
        f"{status.get('events_total', 0)}, urgent alerts {len(alerts)}, "
        f"calls {len(call_entries)}"
    )
    print("demo complete — mock mode, 0 outbound inference calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
