"""Inference telemetry: tolerant vLLM Prometheus polling plus a seeded mock curve.

`VllmMetricsSource` polls a vLLM `/metrics` endpoint every `POLL_INTERVAL_S`
seconds and turns the exposition into a `MetricsSnapshot`. Parsing is
sample-name-driven (prometheus-client groups counter families unreliably) and
tolerant: any metric name a vLLM version does not expose stays `None`, and a
failed poll holds the last good snapshot with `degraded=True`. Nothing here
raises at a caller.

`MockMetricsSource` is the offline twin: a deterministic synthetic curve with
the documented baseline ranges and a `drive_load()` overload window that ramps
waiting / KV% / TTFT up and decays back to baseline — enough to walk the policy
through NORMAL → CONSERVE → SURGE → NORMAL.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx
from prometheus_client.parser import text_string_to_metric_families

from config import Settings

SnapshotSource = Literal["vllm", "mock"]

POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 5.0
HISTORY_LIMIT = 300

# vLLM sample names (verified quirks: match samples by name, never by family).
WAITING_NAME = "vllm:num_requests_waiting"
RUNNING_NAME = "vllm:num_requests_running"
KV_CACHE_NAMES: tuple[str, ...] = (
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",  # fallback candidate across vLLM versions
)
TTFT_NAME = "vllm:time_to_first_token_seconds"
GENERATION_TOKENS_NAMES: tuple[str, ...] = (
    "vllm:request_generation_tokens",
    "vllm:request_generation_tokens_total",  # counter suffix form; `_created` never matches
)

# Mock curve shape: baseline draw ranges and smoothing time constants.
MOCK_SEED = 42
MOCK_BASELINE_WAITING = (0.0, 2.0)
MOCK_BASELINE_RUNNING = (1.0, 4.0)
MOCK_BASELINE_KV = (25.0, 45.0)
MOCK_BASELINE_TTFT = (0.3, 0.8)
MOCK_BASELINE_TOKENS = (40.0, 80.0)
MOCK_RAMP_TAU_S = 2.0  # ~95% of the way to the overload target within ~6s
MOCK_DECAY_TAU_S = 4.5  # back to baseline within ~10–15s after the window
MOCK_MAX_STEP_S = 15.0  # cap one smoothing step after a long read gap
MOCK_MIN_TICK_S = 0.25  # burst of snapshot() calls within one tick share a sample


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """One telemetry reading; `None` means the metric was not available."""

    waiting: float | None = None
    running: float | None = None
    kv_cache_pct: float | None = None  # 0–100
    ttft_p50: float | None = None  # seconds
    tokens_per_s: float | None = None
    collected_at: float = 0.0
    source: SnapshotSource = "mock"
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for the API/JSON surface."""
        return {
            "waiting": self.waiting,
            "running": self.running,
            "kv_cache_pct": self.kv_cache_pct,
            "ttft_p50": self.ttft_p50,
            "tokens_per_s": self.tokens_per_s,
            "collected_at": self.collected_at,
            "source": self.source,
            "degraded": self.degraded,
        }


@dataclass(frozen=True, slots=True)
class VllmMetrics:
    """Fields extracted from one Prometheus exposition; absent names stay `None`."""

    waiting: float | None = None
    running: float | None = None
    kv_cache_pct: float | None = None
    ttft_p50: float | None = None
    generation_tokens: float | None = None

    @property
    def has_any(self) -> bool:
        """True when the exposition yielded at least one usable field."""
        return any(
            value is not None
            for value in (
                self.waiting,
                self.running,
                self.kv_cache_pct,
                self.ttft_p50,
                self.generation_tokens,
            )
        )


def _iter_samples(text: str) -> Iterator[Any]:
    """Yield Prometheus samples; unparseable text stops the stream instead of raising."""
    try:
        for family in text_string_to_metric_families(text):
            yield from family.samples
    except Exception:  # malformed exposition is a data problem, not a crash
        return


def parse_vllm_metrics(text: str) -> VllmMetrics:
    """Extract Night Watch's five fields from a vLLM exposition, tolerantly.

    Sample-name-driven (counter family grouping is unreliable in
    prometheus-client); any missing name leaves its field `None`; unparseable
    text yields all-`None`. Never raises.
    """
    by_name: dict[str, list[Any]] = {}
    for sample in _iter_samples(text):
        by_name.setdefault(sample.name, []).append(sample)

    def first(name: str) -> float | None:
        found = by_name.get(name)  # single-model tonight: one label set per name
        return found[0].value if found else None

    kv_fraction = next(
        (value for name in KV_CACHE_NAMES if (value := first(name)) is not None),
        None,
    )
    return VllmMetrics(
        waiting=first(WAITING_NAME),
        running=first(RUNNING_NAME),
        kv_cache_pct=None if kv_fraction is None else kv_fraction * 100.0,
        ttft_p50=_ttft_p50(by_name.get(f"{TTFT_NAME}_bucket")),
        generation_tokens=next(
            (value for name in GENERATION_TOKENS_NAMES if (value := first(name)) is not None),
            None,
        ),
    )


def _ttft_p50(bucket_samples: list[Any] | None) -> float | None:
    """p50 from cumulative `le` bucket counts; `None` without usable buckets."""
    if not bucket_samples:
        return None
    buckets: list[tuple[float, float]] = []
    for sample in bucket_samples:
        raw_le = sample.labels.get("le")
        if raw_le is None:
            continue
        try:
            buckets.append((float(raw_le), float(sample.value)))
        except ValueError:
            continue
    if not buckets:
        return None
    buckets.sort(key=lambda pair: pair[0])
    return _quantile_from_cumulative_buckets(buckets, 0.5)


def _quantile_from_cumulative_buckets(
    buckets: list[tuple[float, float]], quantile: float
) -> float | None:
    """Linear interpolation inside the bucket containing the quantile."""
    total = buckets[-1][1]  # cumulative count at the widest bound
    if total <= 0:
        return None
    has_finite_bound = any(math.isfinite(bound) for bound, _ in buckets)
    target = quantile * total
    lower_bound, lower_count = 0.0, 0.0
    for bound, count in buckets:
        if count < target:
            lower_bound, lower_count = bound, count
            continue
        if not math.isfinite(bound):
            # quantile beyond the widest finite bucket: report that widest finite bound
            return lower_bound if has_finite_bound else None
        span = count - lower_count
        if span <= 0:  # unreachable with cumulative buckets; defensive
            return lower_bound
        fraction = (target - lower_count) / span
        return lower_bound + fraction * (bound - lower_bound)
    return None


class MetricsSource(ABC):
    """Telemetry provider contract: lifecycle, latest reading, bounded history."""

    def __init__(self, history_limit: int = HISTORY_LIMIT) -> None:
        self._history: deque[MetricsSnapshot] = deque(maxlen=history_limit)

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> MetricsSnapshot | None: ...

    def history(self) -> list[MetricsSnapshot]:
        """Recent snapshots, oldest first (last `HISTORY_LIMIT` by default)."""
        return list(self._history)

    def _record(self, snapshot: MetricsSnapshot) -> None:
        self._history.append(snapshot)


class VllmMetricsSource(MetricsSource):
    """Polls a vLLM /metrics endpoint; missing names degrade, nothing crashes."""

    def __init__(
        self,
        url: str,
        *,
        interval: float = POLL_INTERVAL_S,
        timeout: float = POLL_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
        history_limit: int = HISTORY_LIMIT,
    ) -> None:
        super().__init__(history_limit=history_limit)
        self.url = url
        self.interval = interval
        self.timeout = timeout
        self._transport = transport
        self._clock = clock
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task[None] | None = None
        self._latest: MetricsSnapshot | None = None
        self._previous: tuple[float, float] | None = None  # (collected_at, generation tokens)

    def start(self) -> None:
        """Spawn the poll loop on the running event loop (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        client = self._ensure_client()
        self._task = asyncio.get_running_loop().create_task(
            self._run(client), name="vllm-metrics-poll"
        )

    def stop(self) -> None:
        """Cancel the poll loop; the latest snapshot stays available to readers."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()

    async def aclose(self) -> None:
        """Close the HTTP client (the poll loop also closes it on exit)."""
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()

    def snapshot(self) -> MetricsSnapshot | None:
        """Latest reading; `None` before the first successful poll."""
        return self._latest

    async def poll_once(self) -> None:
        """Run one poll cycle immediately; used by the loop, safe to call directly."""
        client = self._ensure_client()
        try:
            response = await client.get(self.url)
            response.raise_for_status()
            metrics = parse_vllm_metrics(response.text)
        except Exception:  # transport error, non-200, or bad body: hold last good
            self._mark_degraded()
            return
        if not metrics.has_any:
            # HTTP 200 with nothing usable (empty/garbage body) is a failed poll too
            self._mark_degraded()
            return
        self._store(metrics)

    async def _run(self, client: httpx.AsyncClient) -> None:
        try:
            while True:
                await self.poll_once()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            raise
        finally:
            if self._client is client:
                self._client = None
            with contextlib.suppress(Exception):
                await client.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, transport=self._transport)
        return self._client

    def _store(self, metrics: VllmMetrics) -> None:
        now = self._clock()
        tokens_per_s: float | None = None
        if self._previous is not None and metrics.generation_tokens is not None:
            previous_at, previous_tokens = self._previous
            elapsed = now - previous_at
            delta = metrics.generation_tokens - previous_tokens
            if elapsed > 0 and delta >= 0:  # a reset (negative delta) has no rate
                tokens_per_s = delta / elapsed
        snapshot = MetricsSnapshot(
            waiting=metrics.waiting,
            running=metrics.running,
            kv_cache_pct=metrics.kv_cache_pct,
            ttft_p50=metrics.ttft_p50,
            tokens_per_s=tokens_per_s,
            collected_at=now,
            source="vllm",
        )
        self._latest = snapshot
        self._record(snapshot)
        if metrics.generation_tokens is not None:
            self._previous = (now, metrics.generation_tokens)

    def _mark_degraded(self) -> None:
        """Hold the last good reading, flagged degraded (its timestamp stays honest)."""
        if self._latest is None or self._latest.degraded:
            return
        self._latest = replace(self._latest, degraded=True)


@dataclass(slots=True)
class _LoadWindow:
    """One drive_load() overload window and its seeded target levels."""

    intensity: float
    started_at: float
    duration_s: float
    waiting_target: float
    running_target: float
    kv_target: float
    ttft_target: float


class MockMetricsSource(MetricsSource):
    """Seeded synthetic telemetry with the documented baseline and an overload curve.

    Pull-based: every `snapshot()` computes the current point on the curve (calls
    within one `MOCK_MIN_TICK_S` share a sample), so the value sequence is a pure
    function of the seed, the clock, and the `drive_load()` calls. `start()` and
    `stop()` are lifecycle no-ops — there is no background task.
    """

    def __init__(
        self,
        *,
        seed: int = MOCK_SEED,
        clock: Callable[[], float] = time.time,
        history_limit: int = HISTORY_LIMIT,
        min_tick: float = MOCK_MIN_TICK_S,
    ) -> None:
        super().__init__(history_limit=history_limit)
        self._rng = random.Random(seed)
        self._clock = clock
        self._min_tick = min_tick
        self._last: MetricsSnapshot | None = None
        self._computed_at: float | None = None
        self._load: _LoadWindow | None = None
        self._state: tuple[float, float, float, float, float] = self._baseline_draws()

    def start(self) -> None:
        """Lifecycle no-op: the mock curve is computed on read, not by a task."""

    def stop(self) -> None:
        """Lifecycle no-op (see `start`)."""

    def snapshot(self) -> MetricsSnapshot | None:
        now = self._clock()
        if (
            self._last is not None
            and self._computed_at is not None
            and now - self._computed_at < self._min_tick
        ):
            return self._last
        return self._compute(now)

    def drive_load(self, intensity: float, duration_s: float) -> None:
        """Ramp the curve into an overload window; it decays back afterwards."""
        if intensity < 0 or duration_s <= 0:
            raise ValueError("intensity must be >= 0 and duration_s must be > 0")
        self._load = _LoadWindow(
            intensity=intensity,
            started_at=self._clock(),
            duration_s=duration_s,
            waiting_target=6.0 + 6.0 * intensity,
            running_target=4.0 + 4.0 * intensity,
            kv_target=self._rng.uniform(80.0, 95.0),
            ttft_target=self._rng.uniform(4.0, 8.0),
        )

    def _baseline_draws(self) -> tuple[float, float, float, float, float]:
        """One seeded draw per field, in fixed order (waiting, running, kv, ttft, tokens)."""
        return (
            self._rng.uniform(*MOCK_BASELINE_WAITING),
            self._rng.uniform(*MOCK_BASELINE_RUNNING),
            self._rng.uniform(*MOCK_BASELINE_KV),
            self._rng.uniform(*MOCK_BASELINE_TTFT),
            self._rng.uniform(*MOCK_BASELINE_TOKENS),
        )

    def _targets(self, now: float) -> tuple[tuple[float, float, float, float, float], float]:
        load = self._load
        if load is not None and now <= load.started_at + load.duration_s:
            targets = (
                load.waiting_target + self._rng.uniform(-0.5, 0.5),
                load.running_target + self._rng.uniform(-0.5, 0.5),
                load.kv_target + self._rng.uniform(-1.5, 1.5),
                load.ttft_target + self._rng.uniform(-0.3, 0.3),
                self._rng.uniform(*MOCK_BASELINE_TOKENS),
            )
            return targets, MOCK_RAMP_TAU_S
        return self._baseline_draws(), MOCK_DECAY_TAU_S

    def _compute(self, now: float) -> MetricsSnapshot:
        dt = 0.0 if self._computed_at is None else now - self._computed_at
        dt = max(0.0, min(dt, MOCK_MAX_STEP_S))
        targets, tau = self._targets(now)
        alpha = 1.0 - math.exp(-dt / tau)  # exponential approach, poll-rate independent
        waiting, running, kv, ttft, tokens = (
            value + alpha * (target - value)
            for value, target in zip(self._state, targets, strict=True)
        )
        self._state = (waiting, running, kv, ttft, tokens)
        self._computed_at = now
        snapshot = MetricsSnapshot(
            waiting=round(waiting, 2),
            running=round(running, 2),
            kv_cache_pct=round(kv, 1),
            ttft_p50=round(ttft, 3),
            tokens_per_s=round(tokens, 1),
            collected_at=now,
            source="mock",
        )
        self._last = snapshot
        self._record(snapshot)
        return snapshot


def build_metrics_source(settings: Settings) -> MetricsSource:
    """The configured source: the mock curve for the `mock` sentinel, else the poller."""
    if settings.is_mock_metrics:
        return MockMetricsSource()
    return VllmMetricsSource(url=settings.vllm_metrics_url)
