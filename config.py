"""Environment configuration and policy thresholds — the single source of env parsing.

Values come from `.env` via python-dotenv (`load_dotenv`, which does not override
already-exported process variables), so a naive `grep|cut` reader is never needed —
inline comments in `.env` are handled correctly. No other module reads `os.environ`.

Usage: `from config import Config` then `Config.phi_mode`, `Config.port`, ...
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

#: Every env var this module reads. `.env.example` must list exactly these names
#: (POLICY_* and MOCK_FAIL_RUN as commented-out defaults).
ENV_VAR_NAMES: tuple[str, ...] = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "VLLM_METRICS_URL",
    "EGRESS_ALLOWLIST",
    "FALLBACK_BASE_URL",
    "PHI_MODE",
    "PORT",
    "MOCK_FAIL_RUN",
    "POLICY_CONSERVE_WAITING",
    "POLICY_CONSERVE_KV",
    "POLICY_CONSERVE_TTFT",
    "POLICY_SURGE_WAITING",
    "POLICY_SURGE_KV",
    "POLICY_SURGE_TTFT",
)

#: Sentinel endpoint value meaning "no network: canned deterministic replies".
MOCK_ENDPOINT = "mock"

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_PORT = 8765

# Policy thresholds — documented defaults, overridable via the POLICY_* env vars.
DEFAULT_CONSERVE_WAITING = 3.0
DEFAULT_CONSERVE_KV = 0.70
DEFAULT_CONSERVE_TTFT = 2.0
DEFAULT_SURGE_WAITING = 8.0
DEFAULT_SURGE_KV = 0.85
DEFAULT_SURGE_TTFT = 5.0


@dataclass(frozen=True, slots=True)
class ModeParams:
    """How much inference a step may spend; set once the policy mode is chosen."""

    max_tokens: int
    context_items: int
    batch_size: int
    enable_thinking: bool = False


# Params per mode (architecture table). NORMAL/CONSERVE/SURGE are per-event or
# per-batch summarization settings; urgent alerts always use URGENT_PARAMS and are
# processed one at a time, in every mode.
NORMAL_PARAMS = ModeParams(max_tokens=512, context_items=12, batch_size=1)
CONSERVE_PARAMS = ModeParams(max_tokens=256, context_items=6, batch_size=4)
SURGE_PARAMS = ModeParams(max_tokens=128, context_items=3, batch_size=8)
URGENT_PARAMS = ModeParams(max_tokens=160, context_items=3, batch_size=1)


def _parse_bool(raw: str | None, default: bool) -> bool:
    """Parse common truthy/falsy spellings; unset or empty falls back to the default."""
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean for env var: {raw!r}")


def _parse_int(raw: str | None, default: int, name: str) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _parse_float(raw: str | None, default: float, name: str) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _parse_hosts(raw: str | None) -> frozenset[str]:
    """Comma-separated allowlist → lowercased host set (blanks dropped)."""
    if not raw:
        return frozenset()
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Frozen configuration snapshot. Build it with `load_settings()` or `from_env()`."""

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    vllm_metrics_url: str
    egress_allowlist: frozenset[str]
    fallback_base_url: str
    phi_mode: bool
    port: int
    mock_fail_run: bool
    policy_conserve_waiting: float
    policy_conserve_kv: float
    policy_conserve_ttft: float
    policy_surge_waiting: float
    policy_surge_kv: float
    policy_surge_ttft: float

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Settings:
        """Parse a mapping of env-style values (pure: no file or process-env access)."""
        return cls(
            llm_base_url=environ.get("LLM_BASE_URL", MOCK_ENDPOINT).strip(),
            llm_api_key=environ.get("LLM_API_KEY", "").strip(),
            llm_model=environ.get("LLM_MODEL", DEFAULT_MODEL).strip(),
            vllm_metrics_url=environ.get("VLLM_METRICS_URL", MOCK_ENDPOINT).strip(),
            egress_allowlist=_parse_hosts(environ.get("EGRESS_ALLOWLIST")),
            fallback_base_url=environ.get("FALLBACK_BASE_URL", "").strip(),
            # Privacy-first: PHI handling is on unless explicitly disabled (dev mock runs).
            phi_mode=_parse_bool(environ.get("PHI_MODE"), default=True),
            port=_parse_int(environ.get("PORT"), DEFAULT_PORT, "PORT"),
            mock_fail_run=_parse_bool(environ.get("MOCK_FAIL_RUN"), default=False),
            policy_conserve_waiting=_parse_float(
                environ.get("POLICY_CONSERVE_WAITING"),
                DEFAULT_CONSERVE_WAITING,
                "POLICY_CONSERVE_WAITING",
            ),
            policy_conserve_kv=_parse_float(
                environ.get("POLICY_CONSERVE_KV"), DEFAULT_CONSERVE_KV, "POLICY_CONSERVE_KV"
            ),
            policy_conserve_ttft=_parse_float(
                environ.get("POLICY_CONSERVE_TTFT"), DEFAULT_CONSERVE_TTFT, "POLICY_CONSERVE_TTFT"
            ),
            policy_surge_waiting=_parse_float(
                environ.get("POLICY_SURGE_WAITING"), DEFAULT_SURGE_WAITING, "POLICY_SURGE_WAITING"
            ),
            policy_surge_kv=_parse_float(
                environ.get("POLICY_SURGE_KV"), DEFAULT_SURGE_KV, "POLICY_SURGE_KV"
            ),
            policy_surge_ttft=_parse_float(
                environ.get("POLICY_SURGE_TTFT"), DEFAULT_SURGE_TTFT, "POLICY_SURGE_TTFT"
            ),
        )

    @property
    def is_mock_llm(self) -> bool:
        """True when completions are served by the in-process canned provider."""
        return self.llm_base_url.lower() == MOCK_ENDPOINT

    @property
    def is_mock_metrics(self) -> bool:
        """True when telemetry comes from the synthetic load curve, not a vLLM server."""
        return self.vllm_metrics_url.lower() == MOCK_ENDPOINT


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load `.env` via python-dotenv and parse the process environment into Settings."""
    if env_file is None:
        load_dotenv()
    else:
        load_dotenv(dotenv_path=env_file)
    return Settings.from_env(os.environ)


#: The process-wide frozen configuration. Import this: `from config import Config`.
Config: Settings = load_settings()
