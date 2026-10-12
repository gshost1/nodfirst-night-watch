"""Stdio entry point: `python -m mcp` starts the Night Watch MCP tool server.

The server speaks the MCP stdio transport, so an MCP client (for example the Akamai
workshop's agent) launches and owns it — it is not meant to be run by a human. The
tools live in `mcp/nightwatch_server.py`.
"""

from __future__ import annotations

from mcp.nightwatch_server import build_server

#: Module-level server so tests and tooling can import the wired-up instance.
server = build_server()


def main() -> None:
    """Serve the Night Watch tools over stdio until the client disconnects."""
    server.run("stdio")


if __name__ == "__main__":
    main()
