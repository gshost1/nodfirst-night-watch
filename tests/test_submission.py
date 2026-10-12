"""README + submission hygiene — the judged-submission surface (VAL-README-001, VAL-SUB-001).

Everything here is offline and read-only: the README text, `.env.example`, and the
git index of this repo (no network, no services, no ports).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from config import ENV_VAR_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"

AI_INFERENCE_REPO = "https://github.com/akamai-developers/akamai-workshop-ai-inference"
SOLUTION_ARCHITECT_REPO = (
    "https://github.com/akamai-developers/akamai-workshop-solution-architect-agent"
)

# The DigitalOcean host may appear ONLY as the documented config-only fallback
# (VAL-SUB-001): docs may name it, executable code may never reference it.
FALLBACK_HOST = "do-ai.run"
CODE_GLOBS = ("*.py", "*.html", "*.js")

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
BOX_DRAWING = re.compile(r"[┌└│├─▶▼▲┐┘┤]")


def _read_readme() -> str:
    assert README_PATH.is_file(), "README.md is missing from the repo root"
    text = README_PATH.read_text(encoding="utf-8")
    assert text.strip(), "README.md is empty"
    return text


def _section(text: str, heading_pattern: str) -> str:
    """Return the markdown section whose heading matches `heading_pattern`."""
    matches = [m for m in HEADING.finditer(text) if re.search(heading_pattern, m.group(2), re.I)]
    assert matches, f"README has no heading matching {heading_pattern!r}"
    start = matches[0]
    level = len(start.group(1))
    for nxt in HEADING.finditer(text, start.end()):
        if len(nxt.group(1)) <= level:
            return text[start.end() : nxt.start()]
    return text[start.end() :]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout


def _git_paths(*args: str) -> list[str]:
    """Like `_git`, but tolerates exit 1 (git grep's "no matches" status)."""
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode in (0, 1), result.stderr
    return result.stdout.splitlines()


@pytest.fixture(scope="module")
def readme() -> str:
    return _read_readme()


def test_readme_architecture_overview_covers_the_three_owned_inference_pieces(readme: str) -> None:
    """Architecture overview = pipeline + egress guard + self-tuning loop, with a diagram."""
    architecture = _section(readme, r"architecture").lower()
    for term in ("pipeline", "egress", "guard", "self-tuning", "normal", "conserve", "surge"):
        assert term in architecture, f"architecture section does not mention {term!r}"
    assert BOX_DRAWING.search(architecture), "architecture overview needs an ASCII diagram"


def test_readme_states_the_privacy_pitch_verbatim(readme: str) -> None:
    assert "elder health data stays on our gpu on akamai cloud" in readme.lower()


def test_readme_run_instructions_cover_uv_sync_and_the_mock_mode_start(readme: str) -> None:
    quickstart = _section(readme, r"quick ?start|run it|running")
    assert "uv sync" in quickstart
    assert "uv run uvicorn app.app:app" in quickstart
    assert "127.0.0.1:8765" in quickstart
    assert "mock" in quickstart.lower()


def test_readme_lists_the_three_dashboard_buttons(readme: str) -> None:
    for button in ("Run Night Watch", "Simulate Load", "Attempt PHI Egress"):
        assert button in readme, f"README does not document the {button!r} button"


def test_readme_demo_script_keeps_the_fall_alert_and_sibling_tasks(readme: str) -> None:
    demo = _section(readme, r"demo").lower()
    for fragment in ("03:12", "anna", "pavel"):
        assert fragment in demo, f"demo section does not mention {fragment!r}"


def test_readme_carries_the_measured_live_demo_timing_tips(readme: str) -> None:
    """The M2 timing tips: load click must follow the run click immediately; runs shorten."""
    tips = _section(readme, r"demo tips|timing")
    assert "Run Night Watch" in tips and "Simulate Load" in tips
    assert re.search(r"immediatel", tips, re.I), "tips must say to click load immediately"
    assert re.search(r"conserve.{0,160}?\b2\s*s", tips, re.I | re.S), (
        "tips must note the ~2s CONSERVE frame on the way up"
    )
    assert re.search(r"\b2\s*[-–]\s*4\s*s", tips), (
        "tips must note that a SURGE run finishes in ~2-4s"
    )
    assert re.search(r"defer", tips, re.I), "tips must explain why SURGE runs are short"


def test_readme_notes_the_metrics_history_cap(readme: str) -> None:
    assert re.search(r"300[- ]sample", readme, re.I), "README must note the 300-sample cap"
    assert re.search(r"10\s*(min|minutes)", readme, re.I), "cap should be called out as ~10 min"


def test_readme_credits_both_workshop_repos(readme: str) -> None:
    assert AI_INFERENCE_REPO in readme
    assert SOLUTION_ARCHITECT_REPO in readme


def test_readme_discloses_the_prep_per_the_build_on_site_rule(readme: str) -> None:
    lower = readme.lower()
    assert "sunday" in lower, "the prep disclosure must state when the code was written"
    assert "build-on-site" in lower, "the disclosure must name the build-on-site rule"
    assert re.search(r"disclos", lower), "the prep section must be an explicit disclosure"


def test_readme_documents_every_environment_variable(readme: str) -> None:
    for name in ENV_VAR_NAMES:
        assert name in readme, f"README does not document the {name} env var"


def test_readme_env_swap_checklist_names_the_five_variables_and_verification_steps(
    readme: str,
) -> None:
    swap = _section(readme, r"env.swap")
    for name in ("LLM_BASE_URL", "LLM_MODEL", "VLLM_METRICS_URL", "EGRESS_ALLOWLIST", "PHI_MODE"):
        assert name in swap, f"env-swap checklist does not name {name}"
    assert "PHI_MODE=true" in swap
    assert "/v1/models" in swap, "checklist must read the model id from GET /v1/models"
    assert "/metrics" in swap, "checklist must verify the Prometheus metrics endpoint"
    assert re.search(r"screenshot", swap, re.I), "checklist must capture the on-site screenshot"
    assert re.search(r"no code changes", swap, re.I), "the swap must be config-only"


def test_readme_env_swap_flags_the_mock_config_the_test_suite_needs(readme: str) -> None:
    swap = _section(readme, r"env.swap")
    assert "uv run pytest -q" in swap
    assert "mock" in swap.lower()
    assert ".env.example" in swap, "the mock-config recipe must point at .env.example"


def test_readme_documents_the_config_only_fallback(readme: str) -> None:
    lower = readme.lower()
    assert FALLBACK_HOST in lower
    assert "config-only" in lower or "config only" in lower
    assert "never called" in lower, "the fallback must be documented as never called"


def test_readme_references_a_committed_dashboard_screenshot(readme: str) -> None:
    referenced = re.findall(r"docs/[\w./-]+\.png", readme)
    assert referenced, "README should show the submission screenshot"
    tracked = set(_git("ls-files").splitlines())
    for relative in referenced:
        assert (REPO_ROOT / relative).is_file(), f"{relative} is referenced but missing"
        assert relative in tracked, f"{relative} must be committed (not a local artifact)"


def test_env_is_gitignored_and_untracked() -> None:
    tracked = _git("ls-files").splitlines()
    assert not [path for path in tracked if Path(path).name == ".env"], ".env must not be tracked"
    subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO_ROOT, check=True)
    assert ".env.example" in tracked, "the committed template must stay tracked"


@pytest.mark.parametrize("name", [".env", ".env.onsite", ".env.bak"])
def test_local_env_copies_are_gitignored(name: str) -> None:
    """Local copies of the real env (the on-site swap keeps one) must never be committable."""
    subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO_ROOT, check=True)


def test_the_env_template_is_not_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".env.example"], cwd=REPO_ROOT, capture_output=True
    )
    assert result.returncode == 1, ".env.example must stay committable (negated in .gitignore)"


def test_no_secret_bearing_env_file_is_tracked() -> None:
    tracked = _git("ls-files").splitlines()
    extra_env_files = [p for p in tracked if p.startswith(".env.") and p != ".env.example"]
    assert extra_env_files == [], f"unexpected env files tracked: {extra_env_files}"


def test_the_fallback_host_appears_only_as_documentation() -> None:
    """inference.do-ai.run is config-only: no module, page, or script may reference it."""
    # This test module itself names the host inside its assertion, hence the exclusion.
    code_hits = _git_paths(
        "grep",
        "-l",
        "-F",
        FALLBACK_HOST,
        "--",
        *CODE_GLOBS,
        f":(exclude){Path(__file__).resolve().relative_to(REPO_ROOT)}",
    )
    assert code_hits == [], f"the fallback host leaked into code: {sorted(code_hits)}"
    template = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert f"FALLBACK_BASE_URL=https://inference.{FALLBACK_HOST}" in template, (
        ".env.example must document the fallback (config-only)"
    )


def test_readme_documents_the_mock_only_budget_rule(readme: str) -> None:
    lower = readme.lower()
    assert "mock" in lower
    assert re.search(r"no outbound|zero outbound|no.*network", lower), (
        "README must state that the repo makes no outbound inference calls"
    )
