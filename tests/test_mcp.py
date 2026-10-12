"""MCP tool server contract — VAL-MCP-001 (tool listing) and VAL-MCP-002 (round-trip).

The stdio FastMCP server named ``nightwatch`` exposes exactly the six agent tools.
Every tool is verified through the in-memory FastMCP client, plus one real
``python -m mcp`` stdio handshake for the CLI surface. Tools must round-trip against
the same `Store` / `NightWatchRun` / `choose_mode` code the app uses — no duplicated
logic — so the assertions compare their output to the shared modules' own shapes.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from config import Settings
from mcp import ClientSession, StdioServerParameters
from mcp.nightwatch_server import SERVER_NAME, build_server
from nightwatch.digest import HEADER_LINE

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The exact tool set VAL-MCP-001 requires, in the architecture's listed order.
EXPECTED_TOOLS = [
    "get_parent_summary",
    "log_med",
    "add_appointment",
    "assign_sibling_task",
    "run_night_watch",
    "get_inference_mode",
]

#: Hermetic mock env: no `.env` needed, no network possible.
MOCK_ENV = {
    "LLM_BASE_URL": "mock",
    "VLLM_METRICS_URL": "mock",
    "PHI_MODE": "true",
}


def mock_settings(**overrides: str) -> Settings:
    """A mock-mode Settings snapshot independent of the process environment."""
    return Settings.from_env({**MOCK_ENV, **overrides})


def payload(result: Any) -> Any:
    """The tool's JSON payload: structured content, else parsed text, else raw text.

    FastMCP wraps a ``str`` return as ``{"result": <str>}``; a ``dict`` return is
    passed through as structured content.
    """
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if set(structured) == {"result"}:
            return structured["result"]
        return structured
    text = "".join(
        block.text for block in result.content if getattr(block, "type", None) == "text"
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def call_tools(server: Any, calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    """Open one in-memory session and run every call in it, in order."""

    async def _run() -> list[Any]:
        async with create_connected_server_and_client_session(server) as client:
            return [await client.call_tool(name, args) for name, args in calls]

    return asyncio.run(_run())


def list_tools(server: Any) -> Any:
    """List the server's tools over the in-memory transport."""

    async def _run() -> Any:
        async with create_connected_server_and_client_session(server) as client:
            return await client.list_tools()

    return asyncio.run(_run())


# --------------------------------------------------------------- VAL-MCP-001: listing


def test_pyproject_pins_mcp_below_2() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = [dep.replace(" ", "") for dep in data["project"]["dependencies"]]
    assert "mcp>=1,<2" in deps


def test_installed_mcp_is_1x() -> None:
    assert importlib.metadata.version("mcp").split(".")[0] == "1"


def test_server_is_named_nightwatch() -> None:
    server = build_server(settings=mock_settings())
    assert SERVER_NAME == "nightwatch"
    assert server._mcp_server.name == "nightwatch"


def test_lists_exactly_six_tools_each_with_description_and_schema() -> None:
    server = build_server(settings=mock_settings())
    result = list_tools(server)
    names = [tool.name for tool in result.tools]

    assert len(names) == 6
    assert sorted(names) == sorted(EXPECTED_TOOLS)
    for tool in result.tools:
        assert tool.description, f"{tool.name} has no description"
        assert tool.inputSchema["type"] == "object"
        assert "properties" in tool.inputSchema

    by_name = {tool.name: tool for tool in result.tools}
    assert set(by_name["log_med"].inputSchema["properties"]) == {"med", "time", "taken"}
    assert set(by_name["add_appointment"].inputSchema["properties"]) == {"title", "datetime"}
    assert set(by_name["assign_sibling_task"].inputSchema["properties"]) == {"sibling", "task"}


def test_python_dash_m_mcp_lists_tools_over_stdio() -> None:
    """`python -m mcp` boots a real stdio server that answers list_tools."""
    env = {**os.environ, **MOCK_ENV}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp"],
        cwd=str(REPO_ROOT),
        env=env,
    )

    async def _handshake() -> list[str]:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                return [tool.name for tool in listed.tools]

    names = asyncio.run(asyncio.wait_for(_handshake(), timeout=60))
    assert sorted(names) == sorted(EXPECTED_TOOLS)


# ------------------------------------------------------- VAL-MCP-002: round-trip calls


def test_get_parent_summary_returns_seeded_family() -> None:
    server = build_server(settings=mock_settings())
    summary = payload(call_tools(server, [("get_parent_summary", {})])[0])
    assert isinstance(summary, str)
    assert "Ludmila" in summary and "81" in summary
    assert "Anna" in summary and "Pavel" in summary
    assert "metformin" in summary


def test_log_med_mutation_reflected_in_summary() -> None:
    server = build_server(settings=mock_settings())
    before, logged, after = (
        payload(result)
        for result in call_tools(
            server,
            [
                ("get_parent_summary", {}),
                ("log_med", {"med": "metformin", "time": "21:15", "taken": False}),
                ("get_parent_summary", {}),
            ],
        )
    )
    assert "none yet tonight" in before
    assert logged["ok"] is True
    assert logged["med_log_entry"]["med"] == "metformin"
    assert logged["med_log_entry"]["taken"] is False
    assert "metformin at 21:15" in after
    assert "NOT taken" in after


def test_add_appointment_reflected_in_summary() -> None:
    server = build_server(settings=mock_settings())
    added, summary = (
        payload(result)
        for result in call_tools(
            server,
            [
                ("add_appointment", {"title": "Podiatry visit", "datetime": "2026-11-02T10:00"}),
                ("get_parent_summary", {}),
            ],
        )
    )
    assert added["ok"] is True
    assert added["appointment"]["title"] == "Podiatry visit"
    assert "Podiatry visit (2026-11-02T10:00)" in summary


def test_assign_sibling_task_and_unknown_sibling_is_graceful() -> None:
    server = build_server(settings=mock_settings())
    assigned, rejected, summary = (
        payload(result)
        for result in call_tools(
            server,
            [
                ("assign_sibling_task", {"sibling": "anna", "task": "Call the clinic"}),
                ("assign_sibling_task", {"sibling": "Gregor", "task": "n/a"}),
                ("get_parent_summary", {}),
            ],
        )
    )
    assert assigned["ok"] is True
    assert assigned["sibling_task"]["sibling"] == "Anna"
    assert rejected["ok"] is False
    assert "unknown sibling" in rejected["error"]
    assert "Anna: Call the clinic" in summary


def test_get_inference_mode_returns_mode_and_reason() -> None:
    server = build_server(settings=mock_settings())
    decision = payload(call_tools(server, [("get_inference_mode", {})])[0])
    assert decision["mode"] in {"NORMAL", "CONSERVE", "SURGE"}
    assert isinstance(decision["reason"], str) and decision["reason"]
    assert "params" in decision
    # The synthetic baseline is a NORMAL snapshot.
    assert decision["mode"] == "NORMAL"


def test_run_night_watch_returns_app_consistent_digest() -> None:
    server = build_server(settings=mock_settings())
    digest = payload(call_tools(server, [("run_night_watch", {})])[0])

    # Same shape as `GET /api/digest` (`Digest.to_dict()`).
    assert digest["header"] == HEADER_LINE
    assert digest["markdown"].startswith(f"# {HEADER_LINE}")
    assert digest["stats"]["events_total"] == 40
    assert {task["sibling"] for task in digest["tasks"]} == {"Anna", "Pavel"}
    # Urgent alerts come first and the 03:12 possible fall leads the default flow.
    assert digest["urgent_alerts"], "the run must surface urgent alerts"
    assert digest["urgent_alerts"][0]["time"] == "03:12"


def test_servers_do_not_share_store_state() -> None:
    """Each server owns its store; a mutation on one must not leak into another."""
    first = build_server(settings=mock_settings())
    second = build_server(settings=mock_settings())

    call_tools(first, [("log_med", {"med": "metformin", "time": "21:15", "taken": False})])
    summary = payload(call_tools(second, [("get_parent_summary", {})])[0])
    assert "none yet tonight" in summary
