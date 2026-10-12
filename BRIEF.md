# Droid Build Brief — HackerSquad + Akamai "Build AI agents that own their inference" — "Nod First Private Night Watch"
Mon Oct 12 2026 · AWS Builder Loft 2F, 525 Market St SF · check-in 9:30, workshop 10:30, build 1pm, **submit by 4pm**, demos 5:30, winners 6:00. Bring a physical photo ID and pre-register with AWS Loft.
Builder: George Shostakovych, solo. Built during the OIS morning, finished at the Loft 2:30–4pm.

## Sponsors & partners
- **HackerSquad**, host (Luma page, per prep sheet. Exact Luma URL in /workspace/event-scout)
- **Akamai Technologies / Akamai Cloud (Linode)**: GPU, vLLM, LKE, Object Storage, Managed PostgreSQL. Repos (verified to exist):
  - https://github.com/akamai-developers/akamai-workshop-ai-inference (vLLM perf notebooks: prefill/decode, FP8, speculative decoding, batching, tuning. Module 9 = agent on the endpoint)
  - https://github.com/akamai-developers/akamai-workshop-solution-architect-agent (8 modules: tools, MCP, guardrails, memory, evals, multi-agent, LKE deploy. Run: `cd agent && PYTHONPATH=src python -m orchestrator`)
- **AWS Builder Loft**: venue (https://events.builder.aws.com/event/ec2397f2-1ed3-4586-a631-1d644434fdba/register)
- **vLLM** (OSS), **Qwen** (`Qwen/Qwen3-4B` default model)
- Judging criteria, prizes, submission platform: **[UNVERIFIED]**. Luma only says "live demos, feedback from judges, recognition." Ask in Discord https://discord.com/invite/akamai-edgecase

## Stack / rules found
- An event sandbox is provided: an access card → JupyterLab with the repo cloned, GPU running, and vLLM already serving (per prep sheet/README). No need for your own GPU.
- vLLM exposes an OpenAI-compatible `/v1` plus Prometheus `/metrics`.
- The Akamai $300 promo **doesn't cover GPU plans** (GPU ~$0.52/hr). **Don't spin up your own GPU** (it costs money).
- Theme requirement: the agent must "own its inference". Make that the visible core, not a footnote.

## Project: Nod First Private Night Watch (distinct angle = *self-aware agent on owned GPU; health data never leaves your infra*)
Overnight, the agent processes a batch of the parent's check-in data (smart-speaker transcripts, med-box events, sibling messages, all synthetic). It produces a morning **sibling digest** plus flagged alerts. It **owns its inference**:
1. **Privacy:** all PHI-like data goes only to the self-hosted vLLM endpoint. A guardrail blocks any outbound call to third-party LLM hosts (allowlist = VLLM host).
2. **Self-tuning:** before each step, the agent reads vLLM `/metrics` (`vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, time-to-first-token histogram). Under load it switches modes: shorter context, lower max_tokens, batches the remaining items into a single request, and defers non-urgent summaries. Urgent alerts always jump the queue.
3. **Inference dashboard:** live panel showing the GPU KV-cache %, queue depth, tokens/sec, and the agent's current "mode" with the reason.
Different from OIS: OIS = routing across hosted models for cost. Akamai = one owned model, data privacy, and the agent adapting to its own server's telemetry.

## Env vars
```
LLM_BASE_URL=$VLLM_HOST            # e.g. http://<sandbox-host>:8000/v1  (from access card)
LLM_API_KEY=EMPTY                  # vLLM accepts any unless --api-key set [verify on card]
LLM_MODEL=Qwen/Qwen3-4B            # confirm via GET /v1/models
VLLM_METRICS_URL=${VLLM_HOST%/v1}/metrics
EGRESS_ALLOWLIST=<sandbox host>
FALLBACK_BASE_URL=https://inference.do-ai.run/v1   # NON-PHI demo fallback only, flagged in UI
```

## Setup
```bash
git clone <your nodfirst-agent repo>   # same core as OIS brief
pip install openai fastapi uvicorn httpx prometheus-client
curl -s $LLM_BASE_URL/models | jq '.data[].id'
curl -s $VLLM_METRICS_URL | grep -E 'num_requests_waiting|gpu_cache_usage|time_to_first_token' | head
# Optional reference: clone both akamai-developers repos and skim module 9 / orchestrator
```

## Droid prompt (step by step; reuses core/ from the OIS brief)
1. "Reuse `core/llm.py` and `core/agent.py`. Add `core/egress_guard.py`: wrap the client so any base_url host not in EGRESS_ALLOWLIST raises `EgressBlocked` when the payload is tagged phi=True. Unit test it."
2. "Add `inference/metrics.py`: poll VLLM_METRICS_URL every 2s with httpx, parse the Prometheus text (prometheus_client.parser), and expose `snapshot()` → {waiting, running, kv_cache_pct, ttft_p50, tokens_per_s}. Tolerate missing metric names (vLLM versions differ). Mock mode generates a synthetic load curve."
3. "Add `inference/policy.py`: `choose_mode(snapshot)` → NORMAL | CONSERVE | SURGE with explicit thresholds in config, returning params (max_tokens, context_items, batch_size, enable_thinking=False for Qwen3) and a human-readable reason."
4. "Add `nightwatch/`: synthetic dataset generator (40 overnight events for 'Mom, 81': missed dose, bathroom trips, a possible fall at 3:12am, sibling messages). Pipeline: classify each event (urgent first), summarize per policy mode, produce a morning digest for siblings Anna and Pavel with owner-assigned tasks. Log every LLM call with mode, latency, tokens."
5. "Add `scripts/load.py` that fires N concurrent dummy requests at vLLM to create visible load for the demo."
6. "FastAPI app + one `dashboard.html`: left = live vLLM metrics charts (Chart.js CDN) + current mode badge and reason. Right = alerts and the digest as they stream. Button 'Run Night Watch', button 'Simulate load'."
7. "README: architecture, 'Why own inference: elder health data stays on our GPU on Akamai Cloud', run instructions, credits to the akamai-developers workshop repos."
Note for Droid: Qwen3-4B tool calling is less reliable. Prefer structured JSON outputs with a retry/repair step over many tools. Disable thinking or strip `<think>` blocks.

## MVP
Night Watch pipeline + metrics reader + 3-mode policy + dashboard, running against the sandbox vLLM (mock mode as a backup).
## Stretch
- Expose tools via MCP (mirrors the workshop's module), or reuse its guardrails/evals pattern
- Speculative decoding / FP8 toggle comparison from the inference repo, shown as tokens/s before/after
- Deploy the agent to LKE with the workshop manifests (only if free in the sandbox)
- Eval: urgent-recall on 20 labeled events, under NORMAL vs SURGE

## 2-minute demo script
0:00 "Families send Mom's health data to someone else's API. Nod First runs its agent on its own GPU." Show the egress guard blocking a third-party call.
0:25 Click Run Night Watch → events stream, and the 3:12am possible fall surfaces first as an alert.
0:50 Click Simulate load → KV-cache climbs, and the agent flips to CONSERVE/SURGE with a reason. Digest keeps flowing, urgent still first.
1:25 Show the morning digest: "Anna: call Mom's doctor about the missed dose. Pavel: check the bathroom night light."
1:45 "vLLM on Akamai GPU, Qwen3-4B, agent that reads its own server's telemetry. Same core runs on any OpenAI-compatible endpoint." nodfirst.com.

## Submission checklist (by 4:00pm)
- [ ] Public GitHub repo + README + .env.example (no sandbox password/URL committed)
- [ ] Submission form/platform: **[UNVERIFIED]**. Ask in Discord tonight or at 10:30
- [ ] Demo video (1–2 min) if required [UNVERIFIED]. Record anyway as a backup
- [ ] Screenshot of the dashboard against the real sandbox endpoint
- [ ] Be present at 5:30 for top demos

## Curveballs
- **Sandbox/GPU not ready or you arrive late (from OIS ~2:30pm):** you'll miss the 10:30 workshop. Pick up the access card on arrival. Meanwhile run mock metrics and the DO fallback with synthetic **non-PHI** data, and label it clearly. Do NOT buy a GPU.
- **No GPU quota on your own account:** irrelevant. Use the event sandbox only.
- **Model mismatch (not Qwen3-4B):** read `/v1/models` into LLM_MODEL. Metric names differ by vLLM version, so the parser tolerates missing names.
- **Endpoint only reachable from inside JupyterLab:** run the agent inside the Jupyter terminal and use JupyterLab's proxy (`/proxy/8000/`) for the dashboard [verify].
- **Wi-Fi:** hotspot, recorded video, mock mode.
- **Fresh repo / build-on-site rule:** create a new repo at the Loft and commit incrementally. Disclose the prep.
- **Team requirement:** unknown. Ask. Solo assumed.
- **Required sponsor usage:** Akamai vLLM must be the live inference in the demo. The DO fallback appears only if it's down, and it's labeled.
- **Two events at once:** OIS demo by ~1:30, leave by 2:00, and submit by 4:00. Have the repo submittable at MVP by 3:15.
