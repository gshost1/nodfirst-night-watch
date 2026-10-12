"""config.py is the single source of env parsing: mock defaults, egress allowlist, policy table."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from dotenv import dotenv_values

from config import (
    CONSERVE_PARAMS,
    DEFAULT_PORT,
    ENV_VAR_NAMES,
    NORMAL_PARAMS,
    SURGE_PARAMS,
    URGENT_PARAMS,
    Config,
    ModeParams,
    Settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# A line that *assigns* an env var, optionally commented out (the documented-default
# style in .env.example).
ASSIGNMENT = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


def _settings_from_dotenv_file() -> Settings:
    """Parse the repo's real .env through config's own parser (no process-env dependency)."""
    values = dotenv_values(REPO_ROOT / ".env")
    return Settings.from_env({k: v for k, v in values.items() if v is not None})


def _assigns(line: str, name: str) -> bool:
    """True when `line` sets `name` (whether or not it is commented out)."""
    return line.strip().lstrip("#").strip().startswith(f"{name}=")


def test_env_example_lists_exactly_the_variables_config_reads() -> None:
    declared = set(ASSIGNMENT.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))
    assert declared == set(ENV_VAR_NAMES)


def test_env_example_keeps_the_policy_and_fail_run_knobs_commented_out() -> None:
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    optional = [name for name in ENV_VAR_NAMES if name.startswith("POLICY_")] + ["MOCK_FAIL_RUN"]
    for name in optional:
        matching = [line for line in lines if _assigns(line, name)]
        assert len(matching) == 1, f"{name} should appear exactly once"
        assert matching[0].strip().startswith("#"), f"{name} should be commented out"


def test_config_loads_mock_defaults_from_env_file() -> None:
    """The mission's .env is mock-only (locked budget rule) and config must read it."""
    settings = _settings_from_dotenv_file()
    assert settings.llm_base_url == "mock"
    assert settings.vllm_metrics_url == "mock"
    assert settings.llm_model == "Qwen/Qwen3-4B"
    assert settings.port == 8765
    assert settings.egress_allowlist == frozenset({"localhost", "127.0.0.1"})
    assert settings.mock_fail_run is False
    assert isinstance(settings.phi_mode, bool)


def test_env_example_template_parses_to_the_same_mock_defaults() -> None:
    """The committed template is the durable mock contract (survives the on-site swap)."""
    values = dotenv_values(ENV_EXAMPLE)
    settings = Settings.from_env({k: v for k, v in values.items() if v is not None})
    assert settings.llm_base_url == "mock"
    assert settings.vllm_metrics_url == "mock"
    assert settings.llm_model == "Qwen/Qwen3-4B"
    assert settings.port == DEFAULT_PORT
    assert settings.mock_fail_run is False
    assert settings.policy_surge_kv == 0.85  # commented-out defaults still apply


def test_module_singleton_exposes_the_dotenv_backed_config() -> None:
    assert isinstance(Config, Settings)
    assert Config.llm_base_url == "mock"
    assert Config.vllm_metrics_url == "mock"
    assert Config.is_mock_llm and Config.is_mock_metrics
    assert Config.port == DEFAULT_PORT


def test_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        Config.port = 1  # type: ignore[misc]


def test_bare_environment_falls_back_to_mock_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.llm_base_url == "mock"
    assert settings.vllm_metrics_url == "mock"
    assert settings.llm_model == "Qwen/Qwen3-4B"
    assert settings.port == DEFAULT_PORT
    assert settings.mock_fail_run is False
    assert settings.phi_mode is True  # privacy-first default when unset


def test_policy_threshold_defaults_come_from_config() -> None:
    settings = Settings.from_env({})
    assert settings.policy_conserve_waiting == 3
    assert settings.policy_conserve_kv == 0.70
    assert settings.policy_conserve_ttft == 2.0
    assert settings.policy_surge_waiting == 8
    assert settings.policy_surge_kv == 0.85
    assert settings.policy_surge_ttft == 5.0


def test_policy_thresholds_are_overridable_via_env() -> None:
    settings = Settings.from_env(
        {
            "POLICY_CONSERVE_WAITING": "5",
            "POLICY_CONSERVE_KV": "0.5",
            "POLICY_CONSERVE_TTFT": "1.25",
            "POLICY_SURGE_WAITING": "12",
            "POLICY_SURGE_KV": "0.9",
            "POLICY_SURGE_TTFT": "6.5",
        }
    )
    assert settings.policy_conserve_waiting == 5
    assert settings.policy_conserve_kv == 0.5
    assert settings.policy_conserve_ttft == 1.25
    assert settings.policy_surge_waiting == 12
    assert settings.policy_surge_kv == 0.9
    assert settings.policy_surge_ttft == 6.5


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
def test_mock_fail_run_truthy_values(value: str) -> None:
    assert Settings.from_env({"MOCK_FAIL_RUN": value}).mock_fail_run is True


@pytest.mark.parametrize("value", ["0", "false", "No", "off", ""])
def test_mock_fail_run_falsy_values(value: str) -> None:
    assert Settings.from_env({"MOCK_FAIL_RUN": value}).mock_fail_run is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("localhost,127.0.0.1", frozenset({"localhost", "127.0.0.1"})),
        (" Sandbox.Example.COM , localhost ,,", frozenset({"sandbox.example.com", "localhost"})),
        ("", frozenset()),
    ],
)
def test_egress_allowlist_parsing(raw: str, expected: frozenset[str]) -> None:
    assert Settings.from_env({"EGRESS_ALLOWLIST": raw}).egress_allowlist == expected


def test_real_endpoint_is_not_reported_as_mock() -> None:
    settings = Settings.from_env(
        {"LLM_BASE_URL": "http://vllm.local:8000/v1", "VLLM_METRICS_URL": "http://vllm.local:8000/metrics"}
    )
    assert not settings.is_mock_llm
    assert not settings.is_mock_metrics


def test_invalid_numeric_override_raises_with_the_variable_name() -> None:
    with pytest.raises(ValueError, match="POLICY_SURGE_KV"):
        Settings.from_env({"POLICY_SURGE_KV": "not-a-number"})
    with pytest.raises(ValueError, match="PORT"):
        Settings.from_env({"PORT": "eighty"})


def _shape(params: ModeParams) -> tuple[int, int, int]:
    return (params.max_tokens, params.context_items, params.batch_size)


def test_mode_params_match_the_architecture_table() -> None:
    assert _shape(NORMAL_PARAMS) == (512, 12, 1)
    assert _shape(CONSERVE_PARAMS) == (256, 6, 4)
    assert _shape(SURGE_PARAMS) == (128, 3, 8)
    assert (URGENT_PARAMS.max_tokens, URGENT_PARAMS.context_items) == (160, 3)
    for params in (NORMAL_PARAMS, CONSERVE_PARAMS, SURGE_PARAMS, URGENT_PARAMS):
        assert params.enable_thinking is False
