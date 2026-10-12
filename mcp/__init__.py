"""Night Watch MCP server package (stdlib entry point: `python -m mcp`).

The architecture names this package `mcp/`, which shadows the installed `mcp` SDK on
the import path. Two things keep the SDK usable alongside it:

1. `extend_path` merges both package directories, so submodules such as
   `mcp.server.fastmcp` still resolve to the installed SDK.
2. The shadowed SDK `__init__` is re-executed into this namespace, so the SDK's flat
   re-exports (`from mcp import ClientSession`) keep working.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)


def _reexport_shadowed_sdk_api() -> None:
    """Re-exec the installed mcp SDK's `__init__` so its public API stays importable."""
    for entry in list(__path__)[1:]:
        sdk_init = Path(entry) / "__init__.py"
        if sdk_init.is_file():
            source = sdk_init.read_text(encoding="utf-8")
            exec(compile(source, str(sdk_init), "exec"), globals())
            return


_reexport_shadowed_sdk_api()
