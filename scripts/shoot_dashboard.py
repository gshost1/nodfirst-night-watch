"""Capture dashboard screenshots of a live demo run (headless Chromium).

Usage (app already running on :8765):
    uv run --with playwright python scripts/shoot_dashboard.py [base_url] [out_prefix]

Clicks Attempt PHI Egress, Run Night Watch, then Simulate Load right away (the
validated demo order), and saves one shot under load and one after the run.
"""

from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/dashboard-gpu"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(BASE)
        page.wait_for_timeout(3000)
        page.click("#btn-egress")
        page.wait_for_timeout(1000)
        page.click("#btn-run")
        page.click("#btn-load")
        page.wait_for_timeout(20000)
        page.screenshot(path=f"{OUT}-load.png", full_page=True)
        deadline = time.time() + 600
        while time.time() < deadline:
            state = page.evaluate("fetch('api/status').then(r => r.json()).then(s => s.run_state)")
            if state == "done":
                break
            page.wait_for_timeout(3000)
        page.wait_for_timeout(4000)
        page.screenshot(path=f"{OUT}-done.png", full_page=True)
        browser.close()
    print(f"saved {OUT}-load.png and {OUT}-done.png")


if __name__ == "__main__":
    main()
