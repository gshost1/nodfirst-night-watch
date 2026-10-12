"""Self-tuning policy tests: the threshold table, per-mode params, and reasons.

Behavioral spec: VAL-POLICY-001 (SURGE → CONSERVE → NORMAL evaluation, per-mode
params, specific reasons) and VAL-POLICY-002 (thresholds sourced from config and
env-overridable; `None`/degraded telemetry defaults to NORMAL). Everything here is
offline and deterministic — snapshots are constructed directly, no metrics source.
"""

from __future__ import annotations

import json

import pytest

import config as config_module
import inference.policy as policy
from config import CONSERVE_PARAMS, NORMAL_PARAMS, SURGE_PARAMS, URGENT_PARAMS, Settings
from inference.metrics import MetricsSnapshot
from inference.policy import choose_mode, params_for_mode

NO_METRICS_REASON = "no metrics available — defaulting to NORMAL"


def snapshot(
    *,
    waiting: float | None = 1.0,
    kv_cache_pct: float | None = 30.0,
    ttft_p50: float | None = 0.5,
) -> MetricsSnapshot:
    """A cold-baseline snapshot with only the policy metrics overridable."""
    return MetricsSnapshot(
        waiting=waiting,
        running=1.0,
        kv_cache_pct=kv_cache_pct,
        ttft_p50=ttft_p50,
        tokens_per_s=60.0,
        collected_at=1000.0,
        source="mock",
    )


def settings(**overrides: str) -> Settings:
    """A hermetic Settings built from a mapping (no .env / process env access)."""
    values = {"LLM_BASE_URL": "mock", "VLLM_METRICS_URL": "mock", "EGRESS_ALLOWLIST": "localhost"}
    values.update(overrides)
    return Settings.from_env(values)


# --- VAL-POLICY-001: mode selection, params, reasons ---------------------------------


def test_normal_below_every_threshold() -> None:
    decision = choose_mode(snapshot())
    assert decision.mode == "NORMAL"
    assert decision.params is NORMAL_PARAMS
    budget = decision.params
    assert (budget.max_tokens, budget.context_items, budget.batch_size) == (512, 12, 1)
    assert budget.enable_thinking is False
    assert "NORMAL" in decision.reason


def test_normal_reason_names_the_observed_values_and_thresholds() -> None:
    reason = choose_mode(snapshot()).reason
    assert "queue=1 < 3" in reason
    assert "kv_cache=30% < 70%" in reason
    assert "ttft_p50=0.5s < 2s" in reason


def test_surge_is_evaluated_before_conserve_when_every_metric_is_hot() -> None:
    decision = choose_mode(snapshot(waiting=9.0, kv_cache_pct=90.0, ttft_p50=6.0))
    assert decision.mode == "SURGE"
    assert decision.params is SURGE_PARAMS


@pytest.mark.parametrize(
    ("waiting", "expected"),
    [
        (2.9, "NORMAL"),
        (3.0, "CONSERVE"),  # CONSERVE boundary is inclusive
        (7.9, "CONSERVE"),
        (8.0, "SURGE"),  # SURGE boundary is inclusive
        (25.0, "SURGE"),
    ],
)
def test_queue_waiting_boundaries(waiting: float, expected: str) -> None:
    assert choose_mode(snapshot(waiting=waiting)).mode == expected


@pytest.mark.parametrize(
    ("kv_cache_pct", "expected"),
    [
        (69.9, "NORMAL"),
        (70.0, "CONSERVE"),
        (84.9, "CONSERVE"),
        (85.0, "SURGE"),
        (99.0, "SURGE"),
    ],
)
def test_kv_cache_boundaries(kv_cache_pct: float, expected: str) -> None:
    assert choose_mode(snapshot(kv_cache_pct=kv_cache_pct)).mode == expected


@pytest.mark.parametrize(
    ("ttft_p50", "expected"),
    [
        (1.99, "NORMAL"),
        (2.0, "CONSERVE"),
        (4.99, "CONSERVE"),
        (5.0, "SURGE"),
        (9.5, "SURGE"),
    ],
)
def test_ttft_boundaries(ttft_p50: float, expected: str) -> None:
    assert choose_mode(snapshot(ttft_p50=ttft_p50)).mode == expected


def test_high_ttft_with_low_queue_and_low_kv_is_conserve() -> None:
    """NORMAL requires all three metrics below their CONSERVE threshold (VAL-POLICY-001)."""
    decision = choose_mode(snapshot(waiting=0.5, kv_cache_pct=25.0, ttft_p50=2.0))
    assert decision.mode == "CONSERVE"
    assert decision.params is CONSERVE_PARAMS
    assert "ttft_p50=2s ≥ 2s" in decision.reason


@pytest.mark.parametrize(
    ("kw", "expected_mode", "expected_params"),
    [
        (
            {},
            "NORMAL",
            (512, 12, 1),
        ),
        ({"kv_cache_pct": 76.0}, "CONSERVE", (256, 6, 4)),
        ({"waiting": 9.0}, "SURGE", (128, 3, 8)),
    ],
)
def test_per_mode_params_are_returned_with_every_decision(
    kw: dict[str, float], expected_mode: str, expected_params: tuple[int, int, int]
) -> None:
    decision = choose_mode(snapshot(**kw))
    assert decision.mode == expected_mode
    assert (
        decision.params.max_tokens,
        decision.params.context_items,
        decision.params.batch_size,
    ) == expected_params
    assert decision.params.enable_thinking is False


@pytest.mark.parametrize("kw", [{}, {"kv_cache_pct": 76.0}, {"waiting": 9.0}])
def test_urgent_params_are_constant_across_modes(kw: dict[str, float]) -> None:
    decision = choose_mode(snapshot(**kw))
    assert decision.urgent_params is URGENT_PARAMS
    assert decision.urgent_params.max_tokens == 160
    assert decision.urgent_params.context_items == 3
    assert decision.urgent_params.enable_thinking is False


def test_surge_defers_non_urgent_and_says_so() -> None:
    decision = choose_mode(snapshot(waiting=9.0))
    assert decision.defers_non_urgent is True
    assert "deferring non-urgent" in decision.reason
    assert "queue=9 ≥ 8" in decision.reason
    assert "SURGE" in decision.reason


def test_conserve_reason_names_the_metric_value_threshold_and_action() -> None:
    decision = choose_mode(snapshot(kv_cache_pct=76.0))
    assert decision.defers_non_urgent is False
    assert "kv_cache=76% ≥ 70%" in decision.reason
    assert "CONSERVE" in decision.reason
    assert "batch" in decision.reason.lower()


def test_every_triggering_metric_appears_in_the_reason() -> None:
    reason = choose_mode(snapshot(waiting=9.0, kv_cache_pct=88.0)).reason
    assert "queue=9 ≥ 8" in reason
    assert "kv_cache=88% ≥ 85%" in reason


def test_partial_snapshot_uses_only_the_metrics_it_has() -> None:
    decision = choose_mode(snapshot(waiting=None, kv_cache_pct=90.0, ttft_p50=None))
    assert decision.mode == "SURGE"
    assert "kv_cache=90% ≥ 85%" in decision.reason
    assert "queue" not in decision.reason


# --- VAL-POLICY-002: config-sourced thresholds, degraded telemetry --------------------


def test_none_snapshot_defaults_to_normal_with_the_documented_reason() -> None:
    decision = choose_mode(None)
    assert decision.mode == "NORMAL"
    assert decision.params is NORMAL_PARAMS
    assert decision.reason == NO_METRICS_REASON


def test_degraded_snapshot_defaults_to_normal_even_when_values_look_hot() -> None:
    degraded = MetricsSnapshot(
        waiting=12.0,
        running=9.0,
        kv_cache_pct=95.0,
        ttft_p50=7.0,
        tokens_per_s=5.0,
        collected_at=1000.0,
        source="vllm",
        degraded=True,
    )
    decision = choose_mode(degraded)
    assert decision.mode == "NORMAL"
    assert decision.reason == NO_METRICS_REASON


def test_snapshot_without_policy_metrics_defaults_to_normal() -> None:
    decision = choose_mode(snapshot(waiting=None, kv_cache_pct=None, ttft_p50=None))
    assert decision.mode == "NORMAL"
    assert decision.reason == NO_METRICS_REASON


def test_thresholds_come_from_config_and_flip_the_mode_for_the_same_snapshot() -> None:
    snap = snapshot(kv_cache_pct=75.0)  # 75% ≥ 70% under the documented defaults
    assert choose_mode(snap).mode == "CONSERVE"
    relaxed = settings(POLICY_CONSERVE_KV="0.90", POLICY_SURGE_KV="0.95")
    assert choose_mode(snap, relaxed).mode == "NORMAL"

    queue = snapshot(waiting=5.0)
    assert choose_mode(queue).mode == "CONSERVE"
    aggressive = settings(POLICY_SURGE_WAITING="4")
    assert choose_mode(queue, aggressive).mode == "SURGE"

    slow = snapshot(ttft_p50=2.5)
    assert choose_mode(slow).mode == "CONSERVE"
    tolerant = settings(POLICY_CONSERVE_TTFT="3.0", POLICY_SURGE_TTFT="6.0")
    assert choose_mode(slow, tolerant).mode == "NORMAL"


def test_thresholds_default_to_the_config_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = snapshot(kv_cache_pct=75.0)
    assert choose_mode(snap).mode == "CONSERVE"
    monkeypatch.setattr(config_module, "Config", settings(POLICY_CONSERVE_KV="0.90"))
    assert choose_mode(snap).mode == "NORMAL"


def test_a_policy_level_config_patch_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    snap = snapshot(kv_cache_pct=75.0)
    monkeypatch.setattr(policy, "Config", settings(POLICY_CONSERVE_KV="0.90"))
    assert choose_mode(snap).mode == "NORMAL"


def test_kv_threshold_is_compared_in_percent_units() -> None:
    """`kv_cache_pct` is 0–100 while the config threshold is a 0–1 fraction."""
    assert choose_mode(snapshot(kv_cache_pct=70.0)).mode == "CONSERVE"
    assert choose_mode(snapshot(kv_cache_pct=0.70)).mode == "NORMAL"


# --- Decision plumbing ---------------------------------------------------------------


def test_choose_mode_is_deterministic_and_leaves_the_snapshot_untouched() -> None:
    snap = snapshot(kv_cache_pct=76.0)
    first = choose_mode(snap)
    second = choose_mode(snap)
    assert first == second
    assert snap.kv_cache_pct == 76.0


def test_mode_decision_to_dict_is_json_safe() -> None:
    payload = choose_mode(snapshot(kv_cache_pct=76.0)).to_dict()
    assert set(payload) == {"mode", "reason", "params", "urgent_params"}
    assert payload["mode"] == "CONSERVE"
    assert payload["params"] == {
        "max_tokens": 256,
        "context_items": 6,
        "batch_size": 4,
        "enable_thinking": False,
    }
    assert payload["urgent_params"]["max_tokens"] == 160
    json.dumps(payload)  # must not raise


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("NORMAL", NORMAL_PARAMS), ("CONSERVE", CONSERVE_PARAMS), ("SURGE", SURGE_PARAMS)],
)
def test_params_for_mode_maps_every_mode(mode: str, expected: object) -> None:
    assert params_for_mode(mode) is expected


def test_params_for_mode_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown policy mode"):
        params_for_mode("TURBO")
