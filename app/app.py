"""Night Watch HTTP surface: the static dashboard plus the JSON API over one in-process run.

`create_app()` wires the whole demo surface:

- `GET /` serves `dashboard.html`; `GET /static/*` serves the vendored Chart.js.
- `POST /api/run` starts a `NightWatchRun` as a background asyncio task (409 while one
  is active) — a new run replaces the previous one, which is how prior run state resets.
- `GET /api/status` is the documented operating state (mock flags + egress state +
  run counters); `error` appears only when the run failed. The app never dies with a
  failed run: the background task swallows the failure the run already recorded.
- `GET /api/metrics` returns the latest telemetry snapshot plus recent history;
  `GET /api/events?since=<n>` is the incremental feed: **1-based `seq`**, returning only
  items with `seq > since` (default 0 = the whole feed), so a consumer polls with the
  count of events it already holds.
- `GET /api/alerts`, `GET /api/digest` (`{"ready": false}` until the digest lands),
  `GET /api/calls`.
- `POST /api/simulate-load` drives the mock curve (`drive_load`); the real-vLLM branch
  (concurrent non-PHI completions) is implemented but unreachable while metrics are mock.
- `POST /api/egress-demo` attempts a `phi=True` completion against a hardcoded
  third-party host through the guard. The guard raises `EgressBlocked` before any
  network I/O, so the demo costs nothing and touches nothing.

All state is module-level and in-process (single-user demo, no persistence), and the
telemetry poller is started with the app's lifespan. Mock mode only: `LLM_BASE_URL=mock`,
`VLLM_METRICS_URL=mock` — no branch here reaches the network.

Usage: `uvicorn app.app:app` (or `create_app(...)` in tests, with `httpx.ASGITransport`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Settings, load_settings
from core.egress_guard import EgressBlocked, EgressGuard
from core.llm import LLMClient
from inference.metrics import HISTORY_LIMIT, POLL_INTERVAL_S, MetricsSource, build_metrics_source
from inference.policy import ModeDecision, choose_mode
from nightwatch.pipeline import NightWatchRun

logger = logging.getLogger("nightwatch.app")

__all__ = ["APP_STEP_DELAY_S", "EGRESS_DEMO_BASE_URL", "NightWatchApp", "app", "create_app"]

#: Pause per processing step, so a full mock run lasts ~10s: long enough to click
#: "Simulate Load" mid-run and watch the mode flip (the demo's self-tuning beat).
APP_STEP_DELAY_S = 0.25

#: The hardcoded third-party host the egress demo tries to send PHI to.
EGRESS_DEMO_BASE_URL = "https://api.openai.com/v1"
EGRESS_DEMO_PROMPT = (
    "Mom (Ludmila, 81) missed her evening metformin again — summarize this for the family."
)
EGRESS_DEMO_MAX_TOKENS = 32

#: Real-vLLM load simulation (never exercised in mock mode).
REAL_LOAD_CONCURRENCY = 8
REAL_LOAD_PROBE_MAX_TOKENS = 64
REAL_LOAD_PROBE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Generate a short neutral status line for a load test."}
]

DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"
STATIC_DIR = Path(__file__).parent / "static"

#: Served only until the app-dashboard feature writes `dashboard.html` (offline, local assets).
PLACEHOLDER_DASHBOARD = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Night Watch</title>
    <script src="/static/chart.umd.js"></script>
  </head>
  <body>
    <h1>Night Watch</h1>
    <p>The API is up (<code>/api/status</code>). The dashboard UI loads from
    <code>app/dashboard.html</code>; this placeholder stands in until it exists.</p>
  </body>
</html>
"""


class LoadRequest(BaseModel):
    """Body of `POST /api/simulate-load` (the mock curve's overload window)."""

    duration_s: float = Field(default=30.0, gt=0)
    intensity: float = Field(default=1.0, ge=0)


class NightWatchApp:
    """In-process app state: one telemetry source, one metrics poller, one active run."""

    def __init__(
        self,
        settings: Settings,
        *,
        guard: EgressGuard | None = None,
        metrics_source: MetricsSource | None = None,
        step_delay_s: float = APP_STEP_DELAY_S,
        poll_interval_s: float = POLL_INTERVAL_S,
    ) -> None:
        self.settings = settings
        self.guard = (
            guard
            if guard is not None
            else EgressGuard(LLMClient(), settings.egress_allowlist)
        )
        self.metrics_source = (
            metrics_source if metrics_source is not None else build_metrics_source(settings)
        )
        self.step_delay_s = step_delay_s
        self.poll_interval_s = poll_interval_s
        self.run: NightWatchRun | None = None
        self.load_task: asyncio.Task[None] | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._poller_task: asyncio.Task[None] | None = None
        self._active = False
        self._lock = asyncio.Lock()
        # Privacy counters for the dashboard hero strip (cumulative for the process).
        self._calls_prior_runs = 0  # LLM calls made by runs that have since been replaced
        self._load_calls = 0  # non-PHI load-probe completions against the owned endpoint
        self._blocked_attempts = 0  # PHI egress attempts the guard refused pre-I/O

    # ------------------------------------------------------------------ run lifecycle

    @property
    def polling(self) -> bool:
        """True while the metrics poller task is alive."""
        return self._poller_task is not None and not self._poller_task.done()

    async def start_run(self) -> tuple[NightWatchRun | None, str | None]:
        """Start a fresh run unless one is active; returns `(run, conflict_reason)`."""
        async with self._lock:
            if self._active:
                active_id = self.run.run_id if self.run is not None else "unknown"
                return None, f"night-watch run {active_id} is already active"
            run = NightWatchRun(
                client=self.guard,
                metrics_source=self.metrics_source,
                settings=self.settings,
                step_delay_s=self.step_delay_s,
            )
            # A fresh run object is the reset mechanism: no prior events/alerts/calls/digest.
            if self.run is not None:
                self._calls_prior_runs += len(self.run.calls())
            self.run = run
            self._active = True
            self._run_task = asyncio.create_task(self._execute_run(run), name="nightwatch-run")
            return run, None

    async def _execute_run(self, run: NightWatchRun) -> None:
        """Run the watch off the event loop; a failure is reported, never fatal."""
        try:
            await asyncio.to_thread(run.execute)
        except Exception:  # the run already recorded run_state="failed" + the error summary
            logger.info("night-watch run %s failed: %s", run.run_id, run.status().get("error"))
        finally:
            self._active = False

    # ------------------------------------------------------------------- read surface

    def decide(self) -> ModeDecision:
        """Current policy decision from the latest telemetry (never raises)."""
        try:
            snapshot = self.metrics_source.snapshot()
        except Exception:  # telemetry must never take the API down
            snapshot = None
        return choose_mode(snapshot, self.settings)

    def status_payload(self) -> dict[str, Any]:
        """`GET /api/status`: the documented operating state (exact key set)."""
        decision = self.decide()
        payload: dict[str, Any] = {
            "run_state": "idle",
            "running": False,
            "mode": decision.mode,
            "reason": decision.reason,
            "params": asdict(decision.params),
            "egress": self.guard.egress_state(),
            "flags": {
                "mock_llm": self.settings.is_mock_llm,
                "mock_metrics": self.settings.is_mock_metrics,
                "phi_mode": self.settings.phi_mode,
            },
            "privacy": self.privacy_payload(),
            "events_total": 0,
            "events_processed": 0,
            "alerts_count": 0,
            "digest_ready": False,
        }
        run = self.run
        if run is None:
            return payload
        reported = run.status()
        payload["run_state"] = reported["run_state"]
        payload["events_total"] = reported["events_total"]
        payload["events_processed"] = reported["events_processed"]
        payload["alerts_count"] = reported["alerts_count"]
        payload["digest_ready"] = reported["digest_ready"]
        # While the run is stepping, its latest decision is the mode actually in force;
        # otherwise (idle/done/failed) the badge follows live telemetry, so it recovers
        # to NORMAL as the load curve decays after the run has finished.
        if reported["run_state"] == "running" and reported["mode"] is not None:
            payload["mode"] = reported["mode"]
            payload["reason"] = reported["reason"]
            payload["params"] = reported["params"]
        else:
            payload["mode"] = decision.mode
            payload["reason"] = decision.reason
            payload["params"] = asdict(decision.params)
        if self._active and payload["run_state"] == "idle":  # the task has not started yet
            payload["run_state"] = "running"
        payload["running"] = payload["run_state"] == "running"
        if "error" in reported:
            payload["error"] = reported["error"]
        return payload

    def privacy_payload(self) -> dict[str, Any]:
        """Hero numbers: every LLM call stayed on the owned host; blocked attempts moved 0 bytes."""
        current = 0 if self.run is None else len(self.run.calls())
        return {
            "phi_calls_on_host": self._calls_prior_runs + current + self._load_calls,
            "blocked_attempts": self._blocked_attempts,
            "bytes_egressed": 0,
            "host": self.guard.endpoint_host,
        }

    def metrics_payload(self, limit: int = HISTORY_LIMIT) -> dict[str, Any]:
        """`GET /api/metrics`: latest snapshot plus chartable history (a poll adds a sample).

        Each history sample also carries the `mode` the policy would pick from it, so the
        charts can band the NORMAL / CONSERVE / SURGE stretches and mark the transitions.
        """
        try:
            latest = self.metrics_source.snapshot()
        except Exception:
            latest = None
        history = []
        for sample in self.metrics_source.history()[-limit:]:
            entry = sample.to_dict()
            try:
                entry["mode"] = choose_mode(sample, self.settings).mode
            except Exception:  # a bad sample must never take the chart down
                entry["mode"] = None
            history.append(entry)
        return {
            "latest": latest.to_dict() if latest is not None else None,
            "history": history,
        }

    def events_since(self, since: int) -> list[dict[str, Any]]:
        """`GET /api/events?since=`: processed events with 1-based `seq > since`."""
        if self.run is None:
            return []
        return [
            {**event, "seq": event["seq"] + 1}
            for event in self.run.processed_events()
            if event["seq"] >= since
        ]

    def alerts_payload(self) -> list[dict[str, Any]]:
        """`GET /api/alerts`: urgent alerts in emit order (the 03:12 fall first by default)."""
        return [] if self.run is None else self.run.alerts()

    def calls_payload(self) -> list[dict[str, Any]]:
        """`GET /api/calls`: one entry per LLM call the run made."""
        return [] if self.run is None else self.run.calls()

    def digest_payload(self) -> dict[str, Any]:
        """`GET /api/digest`: the digest once it is built, else `{"ready": false}`."""
        run = self.run
        if run is None or run.digest is None:
            return {"ready": False}
        return {"ready": True, **run.digest.to_dict()}

    # ------------------------------------------------------------------------ actions

    async def simulate_load(self, duration_s: float, intensity: float) -> dict[str, Any]:
        """`POST /api/simulate-load`: mock curve, or the real-endpoint load branch."""
        driver = getattr(self.metrics_source, "drive_load", None)
        if callable(driver):
            driver(intensity, duration_s)
            return {
                "started": True,
                "mode": "mock",
                "duration_s": float(duration_s),
                "intensity": float(intensity),
                "concurrency": 0,
            }
        # Real-vLLM branch: unreachable while VLLM_METRICS_URL=mock (the source is the
        # synthetic curve). It fires concurrent NON-PHI completions for the window.
        concurrency = max(1, round(intensity * REAL_LOAD_CONCURRENCY))
        self.load_task = asyncio.create_task(
            self._fire_real_load(concurrency, duration_s), name="nightwatch-load"
        )
        return {
            "started": True,
            "mode": "real",
            "duration_s": float(duration_s),
            "intensity": float(intensity),
            "concurrency": concurrency,
        }

    async def _fire_real_load(self, concurrency: int, duration_s: float) -> None:
        """Hold `concurrency` non-PHI completions against the configured endpoint."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration_s

        async def worker() -> None:
            while loop.time() < deadline:
                with contextlib.suppress(Exception):  # load probing must never crash the app
                    await asyncio.to_thread(
                        self.guard.complete,
                        REAL_LOAD_PROBE_MESSAGES,
                        phi=False,
                        max_tokens=REAL_LOAD_PROBE_MAX_TOKENS,
                    )
                    self._load_calls += 1

        await asyncio.gather(*(worker() for _ in range(concurrency)))

    def egress_demo(self) -> dict[str, Any]:
        """`POST /api/egress-demo`: a PHI attempt at a third-party host, blocked pre-I/O."""
        probe = EgressGuard(
            LLMClient(),
            self.settings.egress_allowlist,
            base_url=EGRESS_DEMO_BASE_URL,
            phi_mode=self.settings.phi_mode,
        )
        try:
            probe.complete(
                [{"role": "user", "content": EGRESS_DEMO_PROMPT}],
                phi=True,
                max_tokens=EGRESS_DEMO_MAX_TOKENS,
            )
        except EgressBlocked as exc:
            self._blocked_attempts += 1
            return {
                "blocked": True,
                "attempted_url": EGRESS_DEMO_BASE_URL,
                "bytes_egressed": 0,
                "phi": True,
                **exc.as_dict(),
            }
        return {
            "blocked": False,
            "attempted_url": EGRESS_DEMO_BASE_URL,
            "bytes_egressed": 0,
            "phi": True,
            "host": probe.endpoint_host,
            "allowlist": sorted(probe.allowlist),
            "message": "no block: the PHI host is allowlisted (or the endpoint is mock)",
        }

    # ------------------------------------------------------------- telemetry lifecycle

    def start_poller(self) -> None:
        """Start the telemetry source plus a ticker that keeps its history warm."""
        self.metrics_source.start()
        if not self.polling:
            self._poller_task = asyncio.get_running_loop().create_task(
                self._poll_metrics(), name="nightwatch-metrics-poller"
            )

    async def _poll_metrics(self) -> None:
        """Sample the source on the poll interval; failures never escape."""
        try:
            while True:
                with contextlib.suppress(Exception):
                    self.metrics_source.snapshot()
                await asyncio.sleep(self.poll_interval_s)
        except asyncio.CancelledError:
            raise

    async def stop_poller(self) -> None:
        """Cancel the ticker, stop the source, and close its HTTP client if it has one."""
        task, self._poller_task = self._poller_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.metrics_source.stop()
        aclose = getattr(self.metrics_source, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()
        if self.load_task is not None and not self.load_task.done():
            self.load_task.cancel()


def create_app(
    settings: Settings | None = None,
    *,
    guard: EgressGuard | None = None,
    metrics_source: MetricsSource | None = None,
    step_delay_s: float = APP_STEP_DELAY_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> FastAPI:
    """Build the FastAPI app (module-level `app` is `create_app()` with live config)."""
    resolved = load_settings() if settings is None else settings
    state = NightWatchApp(
        resolved,
        guard=guard,
        metrics_source=metrics_source,
        step_delay_s=step_delay_s,
        poll_interval_s=poll_interval_s,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        state.start_poller()
        try:
            yield
        finally:
            await state.stop_poller()

    api = FastAPI(title="Night Watch", version="0.1.0", lifespan=lifespan)
    api.state.nightwatch = state
    if STATIC_DIR.is_dir():
        api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @api.get("/", include_in_schema=False)
    async def dashboard() -> Response:
        """Serve the dashboard page (the single static file, no build step)."""
        if DASHBOARD_HTML.is_file():
            return FileResponse(DASHBOARD_HTML, media_type="text/html")
        return HTMLResponse(PLACEHOLDER_DASHBOARD)

    @api.get("/api/status")
    async def get_status() -> dict[str, Any]:
        """Current operating state: run state, policy mode/reason/params, flags, counters."""
        return state.status_payload()

    @api.get("/api/metrics")
    async def get_metrics() -> dict[str, Any]:
        """Latest telemetry snapshot plus recent history (drives the charts)."""
        return state.metrics_payload()

    @api.get("/api/events")
    async def get_events(since: Annotated[int, Query(ge=0)] = 0) -> list[dict[str, Any]]:
        """Processed events with `seq > since` (1-based seq; default 0 is the whole feed)."""
        return state.events_since(since)

    @api.get("/api/alerts")
    async def get_alerts() -> list[dict[str, Any]]:
        """Urgent alerts in emit order."""
        return state.alerts_payload()

    @api.get("/api/digest")
    async def get_digest() -> dict[str, Any]:
        """The morning digest, or `{"ready": false}` until it is built."""
        return state.digest_payload()

    @api.get("/api/calls")
    async def get_calls() -> list[dict[str, Any]]:
        """The full LLM call log for the current run."""
        return state.calls_payload()

    @api.post("/api/run")
    async def start_run() -> dict[str, Any]:
        """Start the night watch in the background; 409 while a run is active."""
        run, conflict = await state.start_run()
        if conflict is not None:
            raise HTTPException(status_code=409, detail=conflict)
        assert run is not None  # conflict is None ⇔ a run was started
        return {"run_id": run.run_id, "started": True}

    @api.post("/api/simulate-load")
    async def simulate_load(request: LoadRequest | None = None) -> dict[str, Any]:
        """Drive the mock load curve through NORMAL → CONSERVE → SURGE → NORMAL."""
        body = request if request is not None else LoadRequest()
        return await state.simulate_load(body.duration_s, body.intensity)

    @api.post("/api/egress-demo")
    async def egress_demo() -> dict[str, Any]:
        """Attempt a PHI completion against a third-party host; the guard blocks pre-I/O."""
        return state.egress_demo()

    return api


#: The uvicorn entry point: `uvicorn app.app:app`.
app = create_app()
