# ruff: noqa: E501
"""Saturate a real vLLM endpoint so the self-tuning policy has something to react to.

The in-app "Simulate Load" holds a handful of probe completions, which a GPU
absorbs without noticing (KV cache ~1.5 % on the Akamai RTX Pro 4000). This
fires many concurrent long, non-PHI completions straight at vLLM so KV cache and
the waiting queue climb for real, then reports what /metrics saw.

Usage:
    uv run python scripts/gpu_stress.py --base http://vllm:8000/v1 \
        --model Qwen/Qwen3-4B --concurrency 192 --seconds 60

Reads the API key from LLM_API_KEY or VLLM_API_KEY (never printed).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import time

import httpx

FILLER = (
    "Synthetic load text for capacity testing; it contains no patient data. "
    "Summarise the following list of neutral facts about weather and trains. "
) * 60  # ~1.5k tokens of prompt per request


def _metric(text: str, name: str) -> float | None:
    m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.eE+-]+)$", text, re.M)
    return float(m.group(1)) if m else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("LLM_BASE_URL", "http://vllm:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    args = ap.parse_args()

    key = os.environ.get("LLM_API_KEY") or os.environ.get("VLLM_API_KEY") or ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    metrics_url = args.base.rstrip("/").removesuffix("/v1") + "/metrics"
    deadline = time.monotonic() + args.seconds
    done = 0
    errors = 0
    peak = {"kv": 0.0, "waiting": 0.0, "running": 0.0}

    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(timeout=180, headers=headers, limits=limits) as client:

        async def worker(i: int) -> None:
            nonlocal done, errors
            while time.monotonic() < deadline:
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": f"[{i}] {FILLER}"}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.9,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                try:
                    r = await client.post(f"{args.base}/chat/completions", json=body)
                    r.raise_for_status()
                    done += 1
                except Exception:
                    errors += 1
                    await asyncio.sleep(0.5)

        async def watch() -> None:
            while time.monotonic() < deadline:
                try:
                    text = (await client.get(metrics_url)).text
                    kv = _metric(text, "vllm:kv_cache_usage_perc") or _metric(text, "vllm:gpu_cache_usage_perc") or 0.0
                    waiting = _metric(text, "vllm:num_requests_waiting") or 0.0
                    running = _metric(text, "vllm:num_requests_running") or 0.0
                    peak["kv"] = max(peak["kv"], kv)
                    peak["waiting"] = max(peak["waiting"], waiting)
                    peak["running"] = max(peak["running"], running)
                    left = deadline - time.monotonic()
                    print(f"t-{left:5.1f}s  kv={kv * 100:5.1f}%  running={running:4.0f}  waiting={waiting:4.0f}  done={done}  err={errors}", flush=True)
                except Exception as exc:  # metrics are best effort
                    print(f"metrics error: {type(exc).__name__}", flush=True)
                await asyncio.sleep(2)

        await asyncio.gather(watch(), *(worker(i) for i in range(args.concurrency)))

    print(
        f"PEAK kv={peak['kv'] * 100:.1f}% running={peak['running']:.0f} waiting={peak['waiting']:.0f} "
        f"completed={done} errors={errors}"
    )


if __name__ == "__main__":
    asyncio.run(main())
