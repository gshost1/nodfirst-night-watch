"""Night Watch MCP tool server — the agent's tools over the Model Context Protocol.

`build_server()` assembles a FastMCP server named "nightwatch" (stdio transport,
started by `python -m mcp`) that publishes the same six tools the app uses. Each tool
is a thin adapter over the shared modules — no logic is duplicated here:

- `get_parent_summary` / `log_med` / `add_appointment` / `assign_sibling_task` call
  the `core.agent.Store` methods (the single tool implementation).
- `run_night_watch` executes one isolated in-process `NightWatchRun` built exactly
  like the FastAPI app builds it (same guard, telemetry source, settings), minus the
  demo pacing. The stdio server is its own process, so the API's 409 double-run rule
  does not apply; the tool returns that run's digest.
- `get_inference_mode` reads `inference.policy.choose_mode` over the latest
  telemetry snapshot.

Mirrors the Akamai workshop's MCP module pattern: the server is a separate process
whose lifetime a client owns, and reads/writes flow through the same domain code as
the rest of the product.
"""

from __future__ import annotations

from typing import Any

from config import Settings, load_settings
from core.agent import Store
from core.egress_guard import EgressGuard
from core.llm import LLMClient
from inference.metrics import MetricsSource, build_metrics_source
from inference.policy import choose_mode
from mcp.server.fastmcp import FastMCP
from nightwatch.pipeline import NightWatchRun

__all__ = ["SERVER_INSTRUCTIONS", "SERVER_NAME", "build_server"]

#: The server name clients see during initialization (VAL-MCP-001).
SERVER_NAME = "nightwatch"

#: One-paragraph description FastMCP reports at initialization.
SERVER_INSTRUCTIONS = (
    "Night Watch caregiver tools for Mom (Ludmila, 81): read her care summary, log "
    "medications, add appointments, assign sibling tasks, run an isolated overnight "
    "watch, and read the agent's current self-tuning inference mode."
)


def build_server(
    *,
    store: Store | None = None,
    settings: Settings | None = None,
    metrics_source: MetricsSource | None = None,
    guard: EgressGuard | None = None,
) -> FastMCP:
    """Assemble the Night Watch MCP server around one shared store and telemetry source.

    Defaults come from the environment (mock-only in this mission): a fresh `Store`
    seeded with Mom/Anna/Pavel, the configured metrics source, and an `EgressGuard`
    wrapping the env-driven `LLMClient` — the same construction the app uses.
    """
    resolved = load_settings() if settings is None else settings
    store = store if store is not None else Store()
    source = metrics_source if metrics_source is not None else build_metrics_source(resolved)
    guard = guard if guard is not None else EgressGuard(LLMClient(), resolved.egress_allowlist)

    server: FastMCP = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    @server.tool()
    def get_parent_summary() -> str:
        """Mom's care picture: logged medications, schedule, appointments, sibling tasks."""
        return store.get_parent_summary()

    @server.tool()
    def log_med(med: str, time: str, taken: bool) -> dict[str, Any]:
        """Record that Mom took (or missed) a medication; reflects in get_parent_summary."""
        entry = store.log_med(med=med, time=time, taken=taken)
        return {"ok": True, "med_log_entry": entry.model_dump()}

    @server.tool()
    def add_appointment(title: str, datetime: str) -> dict[str, Any]:
        """Add an upcoming appointment to the family calendar."""
        appointment = store.add_appointment(title=title, datetime=datetime)
        return {"ok": True, "appointment": appointment.model_dump()}

    @server.tool()
    def assign_sibling_task(sibling: str, task: str) -> dict[str, Any]:
        """Assign a task to Anna or Pavel (case-insensitive); unknown names return ok=False."""
        try:
            entry = store.assign_sibling_task(sibling=sibling, task=task)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "sibling_task": entry.model_dump()}

    @server.tool()
    def run_night_watch() -> dict[str, Any]:
        """Run one isolated in-process mock Night Watch pass and return its morning digest."""
        run = NightWatchRun(
            client=guard,
            metrics_source=source,
            settings=resolved,
            step_delay_s=0.0,
        )
        return run.execute().to_dict()

    @server.tool()
    def get_inference_mode() -> dict[str, Any]:
        """The agent's current self-tuning mode and the human-readable reason for it."""
        return choose_mode(source.snapshot(), resolved).to_dict()

    return server
