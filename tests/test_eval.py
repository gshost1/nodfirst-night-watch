"""Offline contract tests for the urgent-recall eval (eval/).

Covers VAL-EVAL-001 (the eval runs the 20 labeled events under NORMAL and SURGE
and prints a per-mode table with urgent recall, precision, tokens, latency, and
event counts) and VAL-EVAL-002 (urgent recall is 1.0 in both modes on the clean
run with exit 0; the injected misclassification drops recall below 1.0 and makes
the process exit non-zero — the check is enforced, not just printed).

Also pins eval/labeled_events.json to nightwatch.dataset.generate_labeled_events()
(the generator is the ground-truth source; a stale file fails the eval loudly)
and the recall/precision math, all offline in mock mode.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from config import Settings
from eval.run_eval import (
    DEFAULT_LABELED_EVENTS_PATH,
    EVAL_MODES,
    EvalError,
    default_inject_event_id,
    load_labeled_events,
    main,
    recall_precision,
    run_mode,
)
from nightwatch.dataset import generate_labeled_events

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Ground truth within the 20 labeled events (seed 42): 3 urgent, 7 attention, 10 routine.
GROUND_TRUTH_URGENT_IDS = ["event-03", "event-14", "event-27"]


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Mock-mode settings built from the (monkeypatched) process environment."""
    monkeypatch.setenv("LLM_BASE_URL", "mock")
    monkeypatch.setenv("VLLM_METRICS_URL", "mock")
    monkeypatch.setenv("PHI_MODE", "true")
    monkeypatch.setenv("MOCK_FAIL_RUN", "0")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env(os.environ)


# --------------------------------------------------------------- labeled events


def test_labeled_events_file_is_the_generators_20_labeled_events() -> None:
    """The committed JSON is exactly what generate_labeled_events() produces (VAL-EVAL-001)."""
    labeled = load_labeled_events()

    assert labeled == generate_labeled_events()
    assert len(labeled) == 20
    counts = Counter(event.urgency for event in labeled)
    assert counts == Counter({"routine": 10, "attention": 7, "urgent": 3})
    assert [event.id for event in labeled if event.urgency == "urgent"] == GROUND_TRUTH_URGENT_IDS
    assert labeled[0].id == "event-01" and labeled[-1].id == "event-40"


def test_stale_labeled_events_file_fails_loudly(tmp_path: Path) -> None:
    """A file that drifted from the generator is refused — ground truth must be exact."""
    payload = json.loads(DEFAULT_LABELED_EVENTS_PATH.read_text())
    payload["events"][0]["text"] = "tampered: this is not the generator's text"
    stale = tmp_path / "labeled_events.json"
    stale.write_text(json.dumps(payload))

    with pytest.raises(EvalError, match="generate_labeled_events"):
        load_labeled_events(stale)


# ------------------------------------------------------------- recall/precision


def test_recall_precision_math() -> None:
    """Recall misses are recalled; false alerts only damage precision; empty alerts are vacuous."""
    # everything urgent alerted
    assert recall_precision(["a", "b", "c"], ["a", "b", "c"]) == (1.0, 1.0)
    # one urgent event missed -> recall 2/3, precision stays 1.0
    assert recall_precision(["a", "b"], ["a", "b", "c"]) == (2 / 3, 1.0)
    # one false alert for a non-urgent event -> both drop
    assert recall_precision(["a", "x"], ["a", "b"]) == (0.5, 0.5)
    # no alerts at all -> recall 0; precision is vacuously 1.0 (recall carries the miss)
    assert recall_precision([], ["a"]) == (0.0, 1.0)


# ------------------------------------------------------------ run_mode (engine)


def test_eval_runs_the_full_pipeline_under_normal_and_surge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each mode row comes from one full pipeline run: classify + summarize + digest."""
    settings = _settings(monkeypatch)
    labeled = load_labeled_events()

    normal = run_mode("NORMAL", labeled, settings=settings)
    surge = run_mode("SURGE", labeled, settings=settings)

    # Full pipeline, every event classified and summarized, digest built in both modes.
    assert normal.events_total == surge.events_total == 20
    assert normal.digest_ready and surge.digest_ready
    assert len([step for step in normal.steps if step.startswith("classify:")]) == 20
    assert len([step for step in surge.steps if step.startswith("classify:")]) == 20
    assert "digest" in normal.steps and "digest" in surge.steps
    step_prefixes = ("classify:event-", "summarize:", "digest")
    assert all(step.startswith(step_prefixes) for step in normal.steps)

    # NORMAL: one summarization call per event (3 urgent with urgent params + 17 single).
    assert normal.calls == 20 + 20 + 1  # classify + summarize + digest
    assert normal.ok_calls == normal.calls
    summarize_steps = [s for s in normal.steps if s.startswith("summarize:")]
    assert len(summarize_steps) == 20
    assert normal.deferred_count == 0

    # SURGE: urgent still processed one-by-one, all 17 non-urgent deferred into one batch.
    assert surge.calls == 20 + 3 + 1 + 1  # classify + urgent + deferred batch + digest
    assert "summarize:deferred" in surge.steps
    assert surge.deferred_count == 17

    # Urgent alerts in severity-first emit order (the 03:12 fall first) in every mode.
    assert normal.alert_ids == surge.alert_ids == ["event-27", "event-14", "event-03"]
    assert normal.recall == surge.recall == 1.0
    assert normal.precision == surge.precision == 1.0
    assert normal.tokens > 0 and surge.tokens > 0
    assert normal.mean_latency_ms >= 0.0 and surge.mean_latency_ms >= 0.0


def test_injected_misclassification_bypasses_the_verdict_and_drops_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing one known-urgent event to routine: recall 2/3, precision unchanged (VAL-EVAL-002)."""
    settings = _settings(monkeypatch)
    labeled = load_labeled_events()

    injected = run_mode("NORMAL", labeled, settings=settings, inject_event_id="event-27")

    assert injected.injected_event_id == "event-27"
    assert injected.recall == pytest.approx(2 / 3)
    assert injected.precision == 1.0  # the miss lands on recall, not precision
    assert injected.missed_urgent_ids == ["event-27"]
    assert injected.alert_ids == ["event-14", "event-03"]


def test_default_inject_event_is_the_possible_fall() -> None:
    """The injected-failure default targets the demo's most safety-critical event."""
    labeled = load_labeled_events()

    assert default_inject_event_id(labeled) == "event-27"


# ------------------------------------------------------------------- the CLI


def test_eval_prints_per_mode_table_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """VAL-EVAL-001: NORMAL + SURGE rows with recall/precision/tokens/latency + event counts."""
    settings = _settings(monkeypatch)

    code = main([], settings=settings)

    out = capsys.readouterr().out
    assert code == 0
    assert list(EVAL_MODES) == ["NORMAL", "SURGE"]
    for column in ("MODE", "EVENTS", "URGENT", "ALERTS", "RECALL", "PRECISION",
                   "TOKENS", "MEAN_LAT_MS", "WALL_MS"):
        assert column in out
    assert re.search(r"^NORMAL\s+20\s+3\s+3\s+1\.000\s+1\.000", out, re.MULTILINE)
    assert re.search(r"^SURGE\s+20\s+3\s+3\s+1\.000\s+1\.000", out, re.MULTILINE)
    assert "RESULT: PASS" in out


def test_injected_failure_drops_recall_and_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """VAL-EVAL-002: --inject-failure makes recall < 1.0 and exits non-zero."""
    settings = _settings(monkeypatch)

    code = main(["--inject-failure"], settings=settings)

    out = capsys.readouterr().out
    assert code != 0
    assert re.search(r"^NORMAL\s+20\s+3\s+2\s+0\.667", out, re.MULTILINE)
    assert re.search(r"^SURGE\s+20\s+3\s+2\s+0\.667", out, re.MULTILINE)
    assert "0.667" in out and "RESULT: FAIL" in out
    assert "event-27" in out  # the miss is named, not just a number


def test_inject_event_flag_targets_the_named_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--inject-event picks which known-urgent event is forced to routine."""
    settings = _settings(monkeypatch)

    code = main(["--inject-failure", "--inject-event", "event-03"], settings=settings)

    out = capsys.readouterr().out
    assert code != 0
    assert "event-03" in out
    assert "0.667" in out


def test_inject_event_requires_inject_failure() -> None:
    """--inject-event without --inject-failure is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--inject-event", "event-03"])
    assert excinfo.value.code == 2


def test_cli_process_exit_codes_pin_the_contract() -> None:
    """The real CLI: `python eval/run_eval.py` exits 0; with --inject-failure it exits non-zero."""
    clean = subprocess.run(
        [sys.executable, "eval/run_eval.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert clean.returncode == 0, clean.stderr
    assert "NORMAL" in clean.stdout and "SURGE" in clean.stdout
    assert "1.000" in clean.stdout

    injected = subprocess.run(
        [sys.executable, "eval/run_eval.py", "--inject-failure"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert injected.returncode != 0
    assert "0.667" in injected.stdout
