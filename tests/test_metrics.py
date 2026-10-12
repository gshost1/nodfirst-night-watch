"""Offline tests for inference telemetry: tolerant vLLM parsing, poll loop, mock curve."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from config import Config, Settings
from inference.metrics import (
    HISTORY_LIMIT,
    POLL_INTERVAL_S,
    POLL_TIMEOUT_S,
    MetricsSnapshot,
    MockMetricsSource,
    VllmMetricsSource,
    build_metrics_source,
    parse_vllm_metrics,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_ALL_NAMES = "metrics_vllm.txt"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


class FakeClock:
    """Deterministic clock: only moves when the test advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def scripted_transport(script: list[httpx.Response | Exception]) -> httpx.MockTransport:
    """Serve queued responses/exceptions in order, one per request."""
    queue = list(script)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0) if queue else httpx.Response(200, text="")
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler)


def _mode_of(snapshot: MetricsSnapshot) -> str:
    """The policy's mode boundaries, evaluated from config thresholds (0–100 kv)."""
    surge = (
        (snapshot.waiting is not None and snapshot.waiting >= Config.policy_surge_waiting)
        or (snapshot.kv_cache_pct is not None
            and snapshot.kv_cache_pct >= Config.policy_surge_kv * 100)
        or (snapshot.ttft_p50 is not None and snapshot.ttft_p50 >= Config.policy_surge_ttft)
    )
    if surge:
        return "SURGE"
    conserve = (
        (snapshot.waiting is not None and snapshot.waiting >= Config.policy_conserve_waiting)
        or (snapshot.kv_cache_pct is not None
            and snapshot.kv_cache_pct >= Config.policy_conserve_kv * 100)
        or (snapshot.ttft_p50 is not None and snapshot.ttft_p50 >= Config.policy_conserve_ttft)
    )
    return "CONSERVE" if conserve else "NORMAL"


# ---------------------------------------------------------------------------
# VAL-METRICS-001: realistic fixture → all five fields with expected values
# ---------------------------------------------------------------------------


def test_realistic_fixture_extracts_all_five_fields_with_expected_values() -> None:
    parsed = parse_vllm_metrics(fixture(FIXTURE_ALL_NAMES))
    assert parsed.waiting == 2.0
    assert parsed.running == 3.0
    assert parsed.kv_cache_pct == 62.0
    assert parsed.ttft_p50 == pytest.approx(1.5)
    assert parsed.generation_tokens == 8400.0
    # p50 must come from the bucket counts, not _sum/_count (51.2 / 40 = 1.28)
    assert parsed.ttft_p50 != pytest.approx(51.2 / 40.0)


def test_ttft_median_beyond_last_finite_bucket_reports_last_finite_bound() -> None:
    text = (
        "# TYPE vllm:time_to_first_token_seconds histogram\n"
        'vllm:time_to_first_token_seconds_bucket{le="0.5"} 0.0\n'
        'vllm:time_to_first_token_seconds_bucket{le="2.5"} 0.0\n'
        'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10.0\n'
        "vllm:time_to_first_token_seconds_sum 90.0\n"
        "vllm:time_to_first_token_seconds_count 10.0\n"
    )
    assert parse_vllm_metrics(text).ttft_p50 == pytest.approx(2.5)


def test_ttft_with_zero_observations_is_none() -> None:
    text = (
        "# TYPE vllm:time_to_first_token_seconds histogram\n"
        'vllm:time_to_first_token_seconds_bucket{le="0.5"} 0.0\n'
        'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 0.0\n'
    )
    assert parse_vllm_metrics(text).ttft_p50 is None


# ---------------------------------------------------------------------------
# VAL-METRICS-002: renamed/missing names, counter variants, kv fallback, delta
# ---------------------------------------------------------------------------


def test_renamed_fixture_uses_kv_fallback_and_leaves_missing_fields_none() -> None:
    parsed = parse_vllm_metrics(fixture("metrics_vllm_renamed.txt"))
    assert parsed.waiting == 1.0
    assert parsed.kv_cache_pct == 71.0  # fallback candidate vllm:kv_cache_usage_perc
    assert parsed.running is None  # absent name
    assert parsed.ttft_p50 is None  # absent histogram
    assert parsed.generation_tokens == 100.0  # bare `X` counter form, `_created` ignored


def test_missing_fixture_and_empty_body_parse_to_none_without_raising() -> None:
    parsed = parse_vllm_metrics(fixture("metrics_vllm_missing.txt"))
    assert parsed.waiting == 0.0
    assert parsed.kv_cache_pct == 25.0
    assert parsed.running is None
    assert parsed.ttft_p50 is None
    assert parsed.generation_tokens is None

    for empty_text in (fixture("metrics_vllm_empty.txt"), ""):
        empty = parse_vllm_metrics(empty_text)
        assert empty.waiting is None
        assert empty.running is None
        assert empty.kv_cache_pct is None
        assert empty.ttft_p50 is None
        assert empty.generation_tokens is None


def test_garbage_body_parses_to_all_none_without_raising() -> None:
    for text in (fixture("metrics_vllm_garbage.txt"), "total garbage \x00\x01 not prometheus"):
        parsed = parse_vllm_metrics(text)
        assert parsed.waiting is None
        assert parsed.running is None
        assert parsed.kv_cache_pct is None
        assert parsed.ttft_p50 is None
        assert parsed.generation_tokens is None


def test_kv_lookup_tries_candidate_names_in_order() -> None:
    both = (
        "# TYPE vllm:gpu_cache_usage_perc gauge\nvllm:gpu_cache_usage_perc 0.30\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc 0.90\n"
    )
    assert parse_vllm_metrics(both).kv_cache_pct == 30.0  # primary name wins
    only_fallback = both.replace("vllm:gpu_cache_usage_perc", "vllm:cache_other_perc")
    assert parse_vllm_metrics(only_fallback).kv_cache_pct == 90.0


def test_tokens_per_s_is_counter_delta_and_none_on_first_poll() -> None:
    clock = FakeClock(2000.0)
    text = fixture(FIXTURE_ALL_NAMES)
    script = [
        httpx.Response(200, text=text),
        httpx.Response(200, text=text.replace("8400.0", "9100.0")),
        httpx.Response(200, text=text.replace("8400.0", "8800.0")),  # counter reset
    ]

    async def scenario() -> tuple[MetricsSnapshot | None, list[MetricsSnapshot]]:
        source = VllmMetricsSource(
            "http://vllm.test/metrics", transport=scripted_transport(script), clock=clock
        )
        before_start = source.snapshot()
        await source.poll_once()  # t=2000
        clock.advance(2.0)
        await source.poll_once()  # t=2002
        clock.advance(4.0)
        await source.poll_once()  # t=2006
        await source.aclose()
        return before_start, source.history()

    before_start, history = asyncio.run(scenario())
    first, second, third = history
    assert before_start is None  # no data yet, no crash
    assert first.source == "vllm"
    assert first.degraded is False
    assert first.waiting == 2.0 and first.kv_cache_pct == 62.0 and first.ttft_p50 == 1.5
    assert first.tokens_per_s is None  # first poll: no previous counter
    assert second.tokens_per_s == pytest.approx(350.0)  # (9100 - 8400) / 2.0s
    assert third.tokens_per_s is None  # negative delta (server restart) → no rate


# ---------------------------------------------------------------------------
# VAL-METRICS-003: poll loop interval, degradation, clean stop
# ---------------------------------------------------------------------------


def test_source_defaults_match_the_architecture_contract() -> None:
    source = VllmMetricsSource("http://vllm.test/metrics")
    assert source.interval == POLL_INTERVAL_S == 2.0
    assert source.timeout == POLL_TIMEOUT_S == 5.0
    assert HISTORY_LIMIT == 300
    assert source.snapshot() is None  # before any poll: no data, no crash
    assert source.history() == []


def test_poll_client_uses_the_five_second_timeout() -> None:
    async def scenario() -> httpx.AsyncClient:
        source = VllmMetricsSource(
            "http://vllm.test/metrics",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="")),
        )
        await source.poll_once()
        client = source._client
        await source.aclose()
        assert client is not None
        return client

    client = asyncio.run(scenario())
    assert float(client.timeout.read) == POLL_TIMEOUT_S


def test_network_failures_retain_last_good_snapshot_with_degraded_true() -> None:
    clock = FakeClock(3000.0)
    text = fixture(FIXTURE_ALL_NAMES)

    async def scenario(failure: Exception | httpx.Response) -> tuple[
        MetricsSnapshot, MetricsSnapshot, MetricsSnapshot
    ]:
        script = [
            httpx.Response(200, text=text),
            failure,
            failure,
            httpx.Response(200, text=text.replace("8400.0", "8402.0")),
        ]
        source = VllmMetricsSource(
            "http://vllm.test/metrics", transport=scripted_transport(script), clock=clock
        )
        await source.poll_once()
        good = source.snapshot()
        await source.poll_once()
        await source.poll_once()
        degraded = source.snapshot()
        clock.advance(2.0)
        await source.poll_once()
        recovered = source.snapshot()
        await source.aclose()
        assert good is not None and degraded is not None and recovered is not None
        return good, degraded, recovered

    for failure in (
        httpx.Response(500),
        httpx.ConnectError("engine down"),
        httpx.ReadTimeout("metrics too slow"),
    ):
        good, degraded, recovered = asyncio.run(scenario(failure))
        assert good.degraded is False
        assert degraded.degraded is True
        assert degraded.waiting == good.waiting  # last good values retained
        assert degraded.kv_cache_pct == good.kv_cache_pct
        assert degraded.ttft_p50 == good.ttft_p50
        assert degraded.collected_at == good.collected_at  # honest staleness
        assert recovered.degraded is False
        assert recovered.collected_at > good.collected_at


def test_http_200_with_unusable_body_degrades_to_last_good() -> None:
    clock = FakeClock(4000.0)

    async def scenario() -> tuple[MetricsSnapshot, MetricsSnapshot]:
        script = [
            httpx.Response(200, text=fixture(FIXTURE_ALL_NAMES)),
            httpx.Response(200, text=""),  # empty body
            httpx.Response(200, text="<html>gateway melted</html>"),  # garbage body
        ]
        source = VllmMetricsSource(
            "http://vllm.test/metrics", transport=scripted_transport(script), clock=clock
        )
        await source.poll_once()
        good = source.snapshot()
        await source.poll_once()
        await source.poll_once()
        held = source.snapshot()
        await source.aclose()
        assert good is not None and held is not None
        return good, held

    good, held = asyncio.run(scenario())
    assert good.degraded is False and good.waiting == 2.0
    assert held.degraded is True
    assert held.waiting == good.waiting
    assert held.kv_cache_pct == good.kv_cache_pct


def test_poll_loop_polls_at_the_configured_interval_and_stops_cleanly() -> None:
    clock = FakeClock(0.0)
    interval = 0.05
    text = fixture(FIXTURE_ALL_NAMES)

    async def scenario() -> tuple[list[float], bool, bool, list[asyncio.Task]]:
        def handler(request: httpx.Request) -> httpx.Response:
            clock.advance(interval)  # each poll consumes exactly one interval
            return httpx.Response(200, text=text)

        source = VllmMetricsSource(
            "http://vllm.test/metrics",
            interval=interval,
            transport=httpx.MockTransport(handler),
            clock=clock,
        )
        source.start()
        task = source._task
        await asyncio.sleep(0.3)
        source.stop()
        await asyncio.sleep(0.05)
        stamps = [snapshot.collected_at for snapshot in source.history()]
        leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return stamps, bool(task and task.done()), bool(task and task.cancelled()), leftovers

    stamps, done, cancelled, leftovers = asyncio.run(scenario())
    assert len(stamps) >= 3
    gaps = [round(later - earlier, 6) for earlier, later in zip(stamps, stamps[1:], strict=False)]
    assert all(gap == interval for gap in gaps)  # fake clock: exact interval spacing
    assert done and cancelled  # stop() cancelled the poll task
    assert leftovers == []  # no dangling tasks


# ---------------------------------------------------------------------------
# VAL-METRICS-004: mock determinism, baseline ranges, load curve, policy walk
# ---------------------------------------------------------------------------


def test_mock_baseline_is_seed_deterministic_and_within_documented_ranges() -> None:
    clock = FakeClock(0.0)
    seeded = MockMetricsSource(seed=42, clock=clock)
    twin = MockMetricsSource(seed=42, clock=clock)
    other_seed = MockMetricsSource(seed=7, clock=clock)

    sequence: list[MetricsSnapshot] = []
    twin_sequence: list[MetricsSnapshot] = []
    other_sequence: list[MetricsSnapshot] = []
    for _ in range(6):
        clock.advance(1.0)
        sequence.append(seeded.snapshot())
        twin_sequence.append(twin.snapshot())
        other_sequence.append(other_seed.snapshot())

    assert sequence == twin_sequence  # same seed → identical snapshot sequence
    assert sequence != other_sequence  # different seed → different curve
    assert len(seeded.history()) == 6
    for snapshot in sequence:
        assert snapshot.source == "mock"
        assert snapshot.degraded is False
        assert 0.0 <= snapshot.waiting <= 2.0
        assert 25.0 <= snapshot.kv_cache_pct <= 45.0
        assert 0.3 <= snapshot.ttft_p50 <= 0.8
        assert 40.0 <= snapshot.tokens_per_s <= 80.0


def test_rapid_snapshot_calls_share_one_sample_until_the_clock_moves() -> None:
    clock = FakeClock(0.0)
    source = MockMetricsSource(seed=42, clock=clock)
    clock.advance(1.0)
    first = source.snapshot()
    assert source.snapshot() is first  # same tick: cached, not a new sample
    assert len(source.history()) == 1
    clock.advance(0.3)
    fresh = source.snapshot()
    assert fresh is not None and first is not None
    assert fresh is not first
    assert fresh.collected_at > first.collected_at
    assert len(source.history()) == 2


def test_history_is_bounded_to_the_last_snapshots() -> None:
    clock = FakeClock(0.0)
    source = MockMetricsSource(seed=42, clock=clock, history_limit=5)
    for _ in range(10):
        clock.advance(1.0)
        source.snapshot()
    history = source.history()
    assert len(history) == 5
    assert history[0].collected_at == 6.0  # oldest dropped, last five kept
    assert history[-1].collected_at == 10.0


def test_drive_load_walks_the_policy_through_normal_conserve_surge_normal() -> None:
    clock = FakeClock(0.0)
    source = MockMetricsSource(seed=42, clock=clock)
    samples: list[MetricsSnapshot] = []
    for _ in range(3):  # baseline at NORMAL
        clock.advance(1.0)
        samples.append(source.snapshot())
    source.drive_load(intensity=1.0, duration_s=10.0)
    for _ in range(30):  # ramp (≈5–10s), hold, then decay back
        clock.advance(1.0)
        samples.append(source.snapshot())
    modes = [_mode_of(snapshot) for snapshot in samples]

    assert modes[:3] == ["NORMAL", "NORMAL", "NORMAL"]
    first_conserve = modes.index("CONSERVE")
    first_surge = modes.index("SURGE")
    assert first_conserve < first_surge  # it climbs through CONSERVE into SURGE
    assert _mode_of(samples[3]) != "SURGE"  # ramp is gradual, not an instant jump
    window_samples = samples[3:13]  # clock 4..13, load active
    assert any(_mode_of(s) == "SURGE" for s in window_samples)

    in_window = window_samples
    assert 10.0 <= max(s.waiting for s in in_window) <= 14.0  # toward 6 + 6*1
    assert 74.0 <= max(s.kv_cache_pct for s in in_window) <= 96.0  # toward 80–95%
    assert 3.5 <= max(s.ttft_p50 for s in in_window) <= 8.6  # toward 4–8s

    # after the load window: decays back to NORMAL within ~10–15s
    for snapshot, mode in zip(samples[3:], modes[3:], strict=False):
        if snapshot.collected_at >= 13.0 + 10.0:
            assert mode == "NORMAL", f"still {mode} at t={snapshot.collected_at}"
    assert modes[-1] == "NORMAL"


def test_drive_load_waiting_target_scales_with_intensity() -> None:
    clock = FakeClock(0.0)
    source = MockMetricsSource(seed=42, clock=clock)
    clock.advance(1.0)
    source.snapshot()
    source.drive_load(intensity=2.0, duration_s=8.0)
    peaks = []
    for _ in range(8):
        clock.advance(1.0)
        snapshot = source.snapshot()
        assert snapshot is not None
        peaks.append(snapshot.waiting)
    assert max(peaks) >= 15.0  # toward 6 + 6*2 = 18
    assert max(peaks) <= 20.0


def test_mock_source_start_stop_are_safe_lifecycle_noops() -> None:
    clock = FakeClock(0.0)
    source = MockMetricsSource(seed=42, clock=clock)
    source.start()  # no running loop required
    clock.advance(1.0)
    assert source.snapshot() is not None
    source.stop()
    clock.advance(1.0)
    assert source.snapshot() is not None  # pull-based: stop does not freeze reads


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


def test_build_metrics_source_selects_mock_or_vllm_from_settings() -> None:
    mock = build_metrics_source(Settings.from_env({"VLLM_METRICS_URL": "MoCk"}))
    assert isinstance(mock, MockMetricsSource)

    real = build_metrics_source(
        Settings.from_env({"VLLM_METRICS_URL": "http://gpu.local:8000/metrics"})
    )
    assert isinstance(real, VllmMetricsSource)
    assert real.url == "http://gpu.local:8000/metrics"


def test_snapshot_to_dict_exposes_every_field() -> None:
    snapshot = MetricsSnapshot(
        waiting=1.0,
        running=2.0,
        kv_cache_pct=30.0,
        ttft_p50=0.5,
        tokens_per_s=50.0,
        collected_at=123.0,
        source="mock",
    )
    assert snapshot.to_dict() == {
        "waiting": 1.0,
        "running": 2.0,
        "kv_cache_pct": 30.0,
        "ttft_p50": 0.5,
        "tokens_per_s": 50.0,
        "collected_at": 123.0,
        "source": "mock",
        "degraded": False,
    }
