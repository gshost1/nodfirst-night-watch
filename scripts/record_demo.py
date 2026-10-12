"""Record the Nod First / Night Watch demo video with Playwright.

Usage:
    uv run --with playwright python scripts/record_demo.py \
        [APP_URL] [OUT_MP4]

Defaults: http://127.0.0.1:8796/ and upload/demo.mp4 next to the repo.
Captions are injected into the page at runtime; app files are untouched.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "http://127.0.0.1:8796/"
DEFAULT_OUT = ROOT.parent / "upload" / "demo.mp4"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

CAPTION_JS = """
(() => {
  if (document.getElementById('__cap')) return;
  const st = document.createElement('style');
  st.textContent = `
    #__cap{position:fixed;left:50%;bottom:36px;transform:translateX(-50%);
      max-width:1180px;padding:16px 30px;border-radius:999px;
      background:rgba(10,14,22,.86);color:#fff;font:600 28px/1.3 -apple-system,
      'Helvetica Neue',Arial,sans-serif;text-align:center;z-index:2147483646;
      opacity:0;transition:opacity .45s ease;pointer-events:none;
      box-shadow:0 6px 28px rgba(0,0,0,.45)}
    #__cap.on{opacity:1}
    #__ov{position:fixed;inset:0;z-index:2147483645;display:flex;
      flex-direction:column;align-items:center;justify-content:center;
      background:#0b1020;color:#fff;font-family:-apple-system,'Helvetica Neue',
      Arial,sans-serif;text-align:center;opacity:0;transition:opacity .5s ease;
      pointer-events:none}
    #__ov.on{opacity:1}
    #__ov h1{font-size:64px;margin:0 0 20px}
    #__ov h1::before,#__ov h1::after{content:none!important;display:none!important}
    #__ov p{font-size:30px;margin:0;color:#b9c4dd}
    #__ov img{max-width:100%;max-height:100%;object-fit:contain}
  `;
  document.head.appendChild(st);
  const cap = document.createElement('div'); cap.id = '__cap';
  document.body.appendChild(cap);
  const ov = document.createElement('div'); ov.id = '__ov';
  document.body.appendChild(ov);
})();
"""


def caption(page: Page, text: str | None) -> None:
    """Fade the caption pill out, swap its text, and fade back in."""
    page.evaluate(
        """async (t) => {
          const c = document.getElementById('__cap');
          c.classList.remove('on');
          await new Promise(r => setTimeout(r, 450));
          if (t) { c.textContent = t; c.classList.add('on'); }
        }""",
        text,
    )


def overlay(page: Page, html: str | None) -> None:
    page.evaluate(
        """async (h) => {
          const o = document.getElementById('__ov');
          if (h === null) { o.classList.remove('on'); return; }
          o.innerHTML = h; o.classList.add('on');
        }""",
        html,
    )


def img_html(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}">'


def smooth_scroll_to(page: Page, selector: str) -> None:
    page.evaluate(
        """(s) => { const e = document.querySelector(s);
          if (e) e.scrollIntoView({behavior: 'smooth', block: 'start'}); }""",
        selector,
    )


def wait_done(page: Page, cap_s: float = 90) -> bool:
    end = time.time() + cap_s
    while time.time() < end:
        state = page.evaluate(
            "fetch('api/status').then(r => r.json())"
            ".then(j => j.run_state).catch(() => null)"
        )
        if state == "done":
            return True
        page.wait_for_timeout(1000)
    return False


def record(page: Page) -> None:
    page.wait_for_selector("#btn-run")
    page.evaluate(CAPTION_JS)

    # 1. title card
    overlay(
        page,
        "<h1>Nod First · Private Night Watch</h1>"
        "<p>An elder-care agent that owns its inference — "
        "vLLM on Akamai GPU</p>",
    )
    page.wait_for_timeout(5000)
    overlay(page, None)
    page.wait_for_timeout(600)

    # 2. privacy / egress
    caption(page, "Mom's health data never leaves our own GPU.")
    page.wait_for_timeout(3500)
    page.click("#btn-egress")
    page.wait_for_timeout(4000)
    caption(
        page,
        "A PHI call to a third-party API is blocked before any I/O "
        "— 0 bytes out.",
    )
    page.wait_for_timeout(5500)

    # 3. overnight run + load
    caption(page, "Overnight: 40 check-in events from sensors and siblings.")
    page.click("#btn-run")
    page.click("#btn-load")
    page.wait_for_timeout(4500)
    caption(
        page,
        "Load climbs → the agent reads vLLM /metrics and switches itself "
        "NORMAL → CONSERVE → SURGE",
    )
    page.wait_for_timeout(15000)
    caption(
        page,
        "SURGE: shorter outputs, smaller context, batched summaries "
        "— urgent alerts still first",
    )
    page.wait_for_timeout(11000)
    smooth_scroll_to(page, "#alerts-feed")
    caption(page, "03:12 possible fall ranked first — urgency beats timestamps.")
    page.wait_for_timeout(9000)

    # 4. digest
    wait_done(page)
    smooth_scroll_to(page, "#digest-panel")
    caption(page, "Morning digest: tasks assigned to each sibling.")
    page.wait_for_timeout(11000)

    # 5. real GPU evidence
    docs = ROOT / "docs"
    overlay(page, img_html(docs / "dashboard-gpu-surge.jpg"))
    caption(
        page,
        "Real Akamai GPU (Qwen3-4B on vLLM) under stress: KV cache 94%, "
        "queue 168 → agent switched to SURGE",
    )
    page.wait_for_timeout(8000)
    overlay(page, img_html(docs / "dashboard-gpu-fastlane.jpg"))
    caption(
        page,
        "Urgent fast lane: GPU at 98% KV, 169 queued, model at 0/40 — "
        "the 03:12 fall alert is already out (0 ms)",
    )
    page.wait_for_timeout(9000)
    overlay(page, img_html(docs / "dashboard-gpu.jpg"))
    caption(
        page,
        "Real GPU eval: urgent recall 1.0 in NORMAL and SURGE — SURGE uses 65% fewer tokens",
    )
    page.wait_for_timeout(8000)

    # 6. end card
    caption(page, None)
    overlay(
        page,
        "<h1>github.com/gshost1/nodfirst-night-watch</h1>"
        "<p>Nod First · Private Night Watch</p>",
    )
    page.wait_for_timeout(5500)


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    vid_dir = Path(tempfile.mkdtemp(prefix="nw_demo_"))

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(vid_dir),
            record_video_size={"width": 1440, "height": 900},
        )
        page = ctx.new_page()
        page.goto(url, wait_until="load")
        try:
            record(page)
        finally:
            ctx.close()
            browser.close()

    webms = sorted(vid_dir.glob("*.webm"))
    if not webms:
        print("no video recorded", file=sys.stderr)
        return 1
    subprocess.run(
        [
            FFMPEG, "-y", "-i", str(webms[-1]),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-movflags", "+faststart", str(out),
        ],
        check=True,
    )
    shutil.rmtree(vid_dir, ignore_errors=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
