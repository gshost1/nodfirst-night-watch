# ruff: noqa: E501  (embedded JS shim strings)
"""Turn a snapshot folder from snapshot_dashboard.sh into one self-contained page.

JupyterLab's /files/ serves HTML in a CSP sandbox, so the page cannot fetch its
own api/* files. This embeds the frozen API responses and a tiny fetch shim, so
the dashboard renders a real run with no server at all.

Usage: uv run python scripts/inline_snapshot.py <snapshot_dir> [title_note]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    snap = Path(sys.argv[1])
    note = sys.argv[2] if len(sys.argv) > 2 else "frozen snapshot of a live run"
    data = {}
    for f in sorted((snap / "api").iterdir()):
        data[f.name] = json.loads(f.read_text())
    shim = (
        "<script>(function(){var D=" + json.dumps(data).replace("</", "<\\/") + ";"
        "var real=window.fetch;window.fetch=function(u,o){var m=String(u).match(/api\\/([a-z-]+)/);"
        "if(m&&D[m[1]]!==undefined){var body=D[m[1]];if(m[1]==='events'&&window.__snapEventsServed){body=[];}"
        "if(m[1]==='events'){window.__snapEventsServed=true;}"
        "return Promise.resolve(new Response(JSON.stringify(body),{status:200,headers:{'Content-Type':'application/json'}}));}"
        "if(m){return Promise.resolve(new Response('{}',{status:200,headers:{'Content-Type':'application/json'}}));}"
        "return real(u,o);};})();</script>"
    )
    chart = (snap / "static" / "chart.umd.js").read_text()
    html = (snap / "index.html").read_text()
    html = html.replace('<script src="static/chart.umd.js"></script>', "<script>" + chart + "</script>")
    html = html.replace("<head>", "<head>" + shim, 1)
    html = html.replace("</body>", f'<div style="position:fixed;bottom:6px;right:10px;font:12px monospace;opacity:.6">{note}</div></body>', 1)
    out = snap / "snapshot.html"
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
