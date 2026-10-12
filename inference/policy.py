"""Self-tuning policy: turn a telemetry snapshot into a NORMAL/CONSERVE/SURGE verdict.

`choose_mode(snapshot)` is the brain of the "owns its inference" story. It reads the
agent's own vLLM telemetry, applies the threshold table from `config.py` (every
threshold is env-overridable via `POLICY_*`), and returns the mode plus the inference
budget (`ModeParams`) and a human-readable reason that renders on the dashboard badge.

Evaluation order is SURGE, then CONSERVE, else NORMAL:

| mode | any one of |
|---|---|
| SURGE | waiting ≥ 8 OR kv_cache ≥ 85% OR ttft_p50 ≥ 5.0s |
| CONSERVE | waiting ≥ 3 OR kv_cache ≥ 70% OR ttft_p50 ≥ 2.0s |
| NORMAL | none of the above (all three metrics below the CONSERVE thresholds) |

Because NORMAL needs *all three* metrics below their CONSERVE thresholds, a slow-ttft
snapshot with an empty queue and a cold KV cache is CONSERVE. `None` or degraded
telemetry is NORMAL — the policy never crashes on missing metrics.

`kv_cache_pct` is 0–100 while `Config.policy_*_kv` is a 0–1 fraction, so the
comparison scales the threshold.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

import config
from config import (
    CONSERVE_PARAMS,
    NORMAL_PARAMS,
    SURGE_PARAMS,
    URGENT_PARAMS,
    Config,
    ModeParams,
    Settings,
)
from inference.metrics import MetricsSnapshot

Mode = Literal["NORMAL", "CONSERVE", "SURGE"]

MODES: tuple[Mode, ...] = ("NORMAL", "CONSERVE", "SURGE")

#: Inference budget per mode, straight from config.py (no magic numbers in logic).
MODE_PARAMS: dict[str, ModeParams] = {
    "NORMAL": NORMAL_PARAMS,
    "CONSERVE": CONSERVE_PARAMS,
    "SURGE": SURGE_PARAMS,
}

#: Action clause appended to every reason ("metric=value ≥ threshold → MODE: action").
MODE_ACTION: dict[str, str] = {
    "NORMAL": "per-event summaries, full context",
    "CONSERVE": "batching summaries, shorter context",
    "SURGE": "deferring non-urgent summaries, urgent still first",
}

#: Reason used when there is no usable telemetry (None, degraded, or all-None metrics).
NO_METRICS_REASON = "no metrics available — defaulting to NORMAL"

# The singleton as imported at module load, so a monkeypatched `policy.Config` is seen.
_IMPORTED_CONFIG = Config


class ModeDecision(BaseModel):
    """The policy verdict: which mode, how much inference it may spend, and why."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mode: Mode
    params: ModeParams
    reason: str
    urgent_params: ModeParams = URGENT_PARAMS

    @property
    def defers_non_urgent(self) -> bool:
        """True in SURGE, where non-urgent summaries defer into one final batch call."""
        return self.mode == "SURGE"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe shape for `/api/status` (mode, reason, params)."""
        return {
            "mode": self.mode,
            "reason": self.reason,
            "params": asdict(self.params),
            "urgent_params": asdict(self.urgent_params),
        }


def _fmt(value: float) -> str:
    """Compact number for reasons: 9 not 9.0, 85.5 stays 85.5."""
    return f"{value:g}"


def _active_settings(settings: Settings | None) -> Settings:
    """Explicit settings win; otherwise the live config singleton (patch-friendly)."""
    if settings is not None:
        return settings
    if Config is not _IMPORTED_CONFIG:  # a test or caller replaced inference.policy.Config
        return Config
    return config.Config


def _has_policy_metrics(snapshot: MetricsSnapshot) -> bool:
    """True when at least one metric the threshold table reads is available."""
    return any(
        value is not None
        for value in (snapshot.waiting, snapshot.kv_cache_pct, snapshot.ttft_p50)
    )


def _triggers(
    snapshot: MetricsSnapshot,
    *,
    waiting_threshold: float,
    kv_threshold: float,
    ttft_threshold: float,
) -> list[str]:
    """The "metric=value ≥ threshold" clauses this snapshot satisfies, in table order."""
    clauses: list[str] = []
    if snapshot.waiting is not None and snapshot.waiting >= waiting_threshold:
        clauses.append(f"queue={_fmt(snapshot.waiting)} ≥ {_fmt(waiting_threshold)}")
    if snapshot.kv_cache_pct is not None and snapshot.kv_cache_pct >= kv_threshold * 100:
        clauses.append(f"kv_cache={_fmt(snapshot.kv_cache_pct)}% ≥ {_fmt(kv_threshold * 100)}%")
    if snapshot.ttft_p50 is not None and snapshot.ttft_p50 >= ttft_threshold:
        clauses.append(f"ttft_p50={_fmt(snapshot.ttft_p50)}s ≥ {_fmt(ttft_threshold)}s")
    return clauses


def _observed(
    snapshot: MetricsSnapshot,
    *,
    waiting_threshold: float,
    kv_threshold: float,
    ttft_threshold: float,
) -> list[str]:
    """The "metric=value < threshold" clauses for a NORMAL verdict (available metrics only)."""
    clauses: list[str] = []
    if snapshot.waiting is not None:
        clauses.append(f"queue={_fmt(snapshot.waiting)} < {_fmt(waiting_threshold)}")
    if snapshot.kv_cache_pct is not None:
        clauses.append(f"kv_cache={_fmt(snapshot.kv_cache_pct)}% < {_fmt(kv_threshold * 100)}%")
    if snapshot.ttft_p50 is not None:
        clauses.append(f"ttft_p50={_fmt(snapshot.ttft_p50)}s < {_fmt(ttft_threshold)}s")
    return clauses


def _reason(clauses: list[str], mode: Mode) -> str:
    """Render "<clauses> → MODE: <action>"; the dashboard badge shows this verbatim."""
    joiner = ", " if mode == "NORMAL" else " and "
    return f"{joiner.join(clauses)} → {mode}: {MODE_ACTION[mode]}"


def _decision(mode: Mode, reason: str) -> ModeDecision:
    return ModeDecision(
        mode=mode,
        params=MODE_PARAMS[mode],
        reason=reason,
        urgent_params=URGENT_PARAMS,
    )


def choose_mode(
    snapshot: MetricsSnapshot | None, settings: Settings | None = None
) -> ModeDecision:
    """Pick the inference mode for a telemetry snapshot, with params and a reason.

    `settings` overrides the thresholds for this call (defaults to the `config.Config`
    singleton); tests use it to prove a single `POLICY_*` override moves the boundary.
    """
    if snapshot is None or snapshot.degraded or not _has_policy_metrics(snapshot):
        return _decision("NORMAL", NO_METRICS_REASON)
    cfg = _active_settings(settings)

    if surge := _triggers(
        snapshot,
        waiting_threshold=cfg.policy_surge_waiting,
        kv_threshold=cfg.policy_surge_kv,
        ttft_threshold=cfg.policy_surge_ttft,
    ):
        return _decision("SURGE", _reason(surge, "SURGE"))

    if conserve := _triggers(
        snapshot,
        waiting_threshold=cfg.policy_conserve_waiting,
        kv_threshold=cfg.policy_conserve_kv,
        ttft_threshold=cfg.policy_conserve_ttft,
    ):
        return _decision("CONSERVE", _reason(conserve, "CONSERVE"))

    observed = _observed(
        snapshot,
        waiting_threshold=cfg.policy_conserve_waiting,
        kv_threshold=cfg.policy_conserve_kv,
        ttft_threshold=cfg.policy_conserve_ttft,
    )
    return _decision("NORMAL", _reason(observed, "NORMAL"))


def params_for_mode(mode: str) -> ModeParams:
    """The `ModeParams` for a mode name (pipeline and eval harnesses select modes by name)."""
    normalized = mode.strip().upper()
    if normalized not in MODE_PARAMS:
        raise ValueError(f"unknown policy mode: {mode!r} (expected one of {', '.join(MODES)})")
    return MODE_PARAMS[normalized]


__all__ = [
    "CONSERVE_PARAMS",
    "MODE_ACTION",
    "MODE_PARAMS",
    "MODES",
    "NORMAL_PARAMS",
    "NO_METRICS_REASON",
    "SURGE_PARAMS",
    "URGENT_PARAMS",
    "Mode",
    "ModeDecision",
    "choose_mode",
    "params_for_mode",
]
