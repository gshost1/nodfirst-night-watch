"""Scaffold contract: the flat top-level packages exist, import cleanly, and don't clash.

`mcp/` shadows the installed `mcp` SDK on the import path, so both must stay usable.
"""

from __future__ import annotations

import importlib

import pytest

SCAFFOLD_PACKAGES = ["core", "inference", "nightwatch", "app", "eval", "scripts", "tests"]


@pytest.mark.parametrize("name", SCAFFOLD_PACKAGES)
def test_scaffold_package_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_local_mcp_package_and_installed_sdk_coexist() -> None:
    local_mcp = importlib.import_module("mcp")
    assert str(local_mcp.__path__[0]).endswith("/mcp")

    fastmcp = importlib.import_module("mcp.server.fastmcp")
    assert hasattr(fastmcp, "FastMCP")

    client_session = importlib.import_module("mcp.client.session")
    assert hasattr(client_session, "ClientSession")
    # The SDK's flat re-exports survive the shadowing shim.
    assert local_mcp.ClientSession is client_session.ClientSession
