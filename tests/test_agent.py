"""Offline contract tests for the caregiver agent (persona, tool protocol, store).

Covers VAL-AGENT-001 (persona safety lines asserted as substrings of the exposed
system prompt), VAL-AGENT-002 (graceful JSON tool-protocol degradation), and
VAL-AGENT-003 (seeded in-memory store mutated/read by all four tools).
"""

from __future__ import annotations

from typing import Any

import pytest

from core.agent import (
    SYSTEM_PROMPT,
    TOOL_ADD_APPOINTMENT,
    TOOL_ASSIGN_SIBLING_TASK,
    TOOL_GET_PARENT_SUMMARY,
    TOOL_LOG_MED,
    TOOL_NAMES,
    ReplyKind,
    Store,
    apply_reply,
    parse_agent_reply,
    run_tool,
)

# --- VAL-AGENT-001: the system prompt carries the caregiver safety lines ---


def test_system_prompt_never_diagnoses() -> None:
    """The no-diagnosis line must survive as a substring of the persona."""
    assert "never diagnose" in SYSTEM_PROMPT
    assert "not a doctor" in SYSTEM_PROMPT


def test_system_prompt_escalates_to_911_and_mom_s_doctor() -> None:
    """Emergency wording: call 911 for emergencies, call Mom's doctor otherwise."""
    assert "call 911" in SYSTEM_PROMPT
    assert "call Mom's doctor" in SYSTEM_PROMPT


def test_system_prompt_is_written_for_the_sibling_audience() -> None:
    """Audience framing: adult children — specifically siblings Anna and Pavel."""
    assert "adult children" in SYSTEM_PROMPT
    assert "siblings Anna and Pavel" in SYSTEM_PROMPT


def test_system_prompt_describes_the_json_tool_protocol() -> None:
    """The protocol block names all four tools and the single-JSON-block contract."""
    for tool_name in (
        TOOL_LOG_MED,
        TOOL_ADD_APPOINTMENT,
        TOOL_ASSIGN_SIBLING_TASK,
        TOOL_GET_PARENT_SUMMARY,
    ):
        assert tool_name in SYSTEM_PROMPT
    assert '"tool"' in SYSTEM_PROMPT
    assert '"args"' in SYSTEM_PROMPT


# --- VAL-AGENT-002: the JSON tool protocol degrades gracefully ---


def test_parse_returns_the_tool_call_for_a_tool_action_json_block() -> None:
    content = (
        "Sure thing. "
        '{"tool": "log_med", "args": {"med": "metformin", "time": "21:15", "taken": false}, '
        '"reply": "Noted - worth a check-in with Mom."} '
        "That is everything."
    )
    reply = parse_agent_reply(content)
    assert reply.kind is ReplyKind.TOOL
    assert reply.tool == "log_med"
    assert reply.args == {"med": "metformin", "time": "21:15", "taken": False}
    assert reply.text == "Noted - worth a check-in with Mom."


def test_parse_returns_the_parsed_dict_for_a_bare_json_block() -> None:
    content = '{"urgency": "urgent", "rationale": "possible fall at 03:12"}'
    reply = parse_agent_reply(content)
    assert reply.kind is ReplyKind.JSON
    assert reply.data == {"urgency": "urgent", "rationale": "possible fall at 03:12"}


def test_parse_returns_plain_text_for_prose() -> None:
    prose = "Mom sounded cheerful on tonight's call and mentioned she slept well."
    reply = parse_agent_reply(prose)
    assert reply.kind is ReplyKind.TEXT
    assert reply.tool is None
    assert reply.data is None
    assert reply.text == prose


def test_parse_tolerates_code_fences_around_tool_json() -> None:
    fenced = (
        "```json\n"
        '{"tool": "assign_sibling_task", "args": {"sibling": "Pavel", '
        '"task": "Check the bathroom night light"}}\n'
        "```"
    )
    reply = parse_agent_reply(fenced)
    assert reply.kind is ReplyKind.TOOL
    assert reply.tool == "assign_sibling_task"
    assert reply.args == {"sibling": "Pavel", "task": "Check the bathroom night light"}
    assert reply.text == ""


def test_parse_accepts_the_action_key_alias() -> None:
    reply = parse_agent_reply(
        '{"action": "add_appointment", "arguments": {"title": "Eye exam", '
        '"datetime": "2026-10-16T08:30"}}'
    )
    assert reply.kind is ReplyKind.TOOL
    assert reply.tool == "add_appointment"
    assert reply.args == {"title": "Eye exam", "datetime": "2026-10-16T08:30"}


def test_parse_defaults_to_empty_args_when_a_tool_block_omits_them() -> None:
    reply = parse_agent_reply('{"tool": "get_parent_summary"}')
    assert reply.kind is ReplyKind.TOOL
    assert reply.tool == "get_parent_summary"
    assert reply.args == {}


def test_parse_treats_a_non_string_tool_field_as_bare_json() -> None:
    reply = parse_agent_reply('{"tool": 5, "note": "not a call"}')
    assert reply.kind is ReplyKind.JSON
    assert reply.data == {"tool": 5, "note": "not a call"}


def test_parse_falls_back_to_text_for_an_unknown_tool() -> None:
    content = '{"tool": "book_flight", "args": {"to": "Prague"}}'
    reply = parse_agent_reply(content)
    assert reply.kind is ReplyKind.TEXT
    assert reply.tool is None
    assert reply.text == content


def test_parse_falls_back_to_text_for_malformed_json() -> None:
    content = '{"tool": "log_med", "args": {"med": "metformin"'
    reply = parse_agent_reply(content)
    assert reply.kind is ReplyKind.TEXT
    assert reply.text == content


def test_parse_falls_back_to_text_for_single_quoted_json() -> None:
    content = "{'tool': 'log_med', 'args': {}}"
    reply = parse_agent_reply(content)
    assert reply.kind is ReplyKind.TEXT
    assert reply.text == content


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_parse_falls_back_to_text_for_empty_or_blank_replies(content: str) -> None:
    reply = parse_agent_reply(content)
    assert reply.kind is ReplyKind.TEXT
    assert reply.tool is None


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "}",
        '{"a": {',
        '{"tool": "log_med", "args": "just a sentence"}',
        "[]",
        '[{"tool": "get_parent_summary", "args": {}}]',
        '{"tool": "log_med", "args": {"med": "metformin", "time": "9pm", "taken": "maybe"}}',
        'he said "{hello}" and nothing else',
        42,
        None,
    ],
)
def test_parse_never_raises_on_adversarial_input(content: Any) -> None:
    """No malformed or mistyped reply can crash the protocol parser."""
    reply = parse_agent_reply(content)
    assert reply.kind in {ReplyKind.TOOL, ReplyKind.JSON, ReplyKind.TEXT}


# --- VAL-AGENT-003: seeded store, mutated and read by all four tools ---


def test_fresh_store_is_seeded_for_the_demo() -> None:
    store = Store()
    assert store.parent.name == "Ludmila"
    assert store.parent.age == 81
    assert [sibling.name for sibling in store.siblings] == ["Anna", "Pavel"]
    schedule = [(slot.med, slot.time_of_day) for slot in store.med_schedule]
    assert schedule == [("metformin", "evening")]
    assert len(store.appointments) >= 2
    assert all(appointment.title and appointment.datetime for appointment in store.appointments)
    assert store.med_log == []
    assert store.sibling_tasks == []


def test_two_fresh_stores_do_not_share_state() -> None:
    first, second = Store(), Store()
    first.log_med(med="metformin", time="21:15", taken=True)
    assert second.med_log == []


def test_log_med_appends_an_entry_to_the_store() -> None:
    store = Store()
    entry = store.log_med(med="metformin", time="21:15", taken=False)
    assert (entry.med, entry.time, entry.taken) == ("metformin", "21:15", False)
    logged = [(m.med, m.time, m.taken) for m in store.med_log]
    assert logged == [("metformin", "21:15", False)]


def test_add_appointment_appends_to_the_store() -> None:
    store = Store()
    before = len(store.appointments)
    created = store.add_appointment(title="Podiatry consult", datetime="2026-10-14T10:00")
    assert created.title == "Podiatry consult"
    assert len(store.appointments) == before + 1
    assert store.appointments[-1].datetime == "2026-10-14T10:00"


def test_assign_sibling_task_appends_and_normalizes_the_sibling_name() -> None:
    store = Store()
    created = store.assign_sibling_task(sibling="pavel", task="Check the bathroom night light")
    assert (created.sibling, created.task) == ("Pavel", "Check the bathroom night light")
    assert store.sibling_tasks[-1].sibling == "Pavel"


def test_assign_sibling_task_rejects_an_unknown_sibling() -> None:
    store = Store()
    with pytest.raises(ValueError, match="unknown sibling"):
        store.assign_sibling_task(sibling="Cousin Ira", task="Bring soup")


def test_get_parent_summary_reflects_mutations_made_earlier() -> None:
    store = Store()
    store.log_med(med="metformin", time="21:15", taken=False)
    store.add_appointment(title="Podiatry consult", datetime="2026-10-14T10:00")
    store.assign_sibling_task(
        sibling="Anna", task="Call Mom's doctor about the missed metformin dose"
    )

    summary = store.get_parent_summary()

    assert "Ludmila" in summary
    assert "metformin at 21:15" in summary
    assert "NOT taken" in summary
    assert "Podiatry consult" in summary
    assert "Anna: Call Mom's doctor about the missed metformin dose" in summary


def test_get_parent_summary_on_a_fresh_store_renders_the_seed_and_empty_sections() -> None:
    store = Store()
    summary = store.get_parent_summary()
    assert "Ludmila" in summary
    assert "81" in summary
    assert "Anna" in summary
    assert "Pavel" in summary
    for appointment in store.appointments:
        assert appointment.title in summary
    assert "none" in summary  # empty med-log and task sections are named, not hidden


# --- tool execution surfaces (run_tool / apply_reply) ---


def test_tool_names_registry_lists_exactly_the_four_tools() -> None:
    assert TOOL_NAMES == frozenset(
        {"log_med", "add_appointment", "assign_sibling_task", "get_parent_summary"}
    )


def test_run_tool_executes_each_known_tool_and_mutates_the_store() -> None:
    store = Store()

    logged = run_tool(store, "log_med", {"med": "metformin", "time": "21:15", "taken": False})
    assert logged["ok"] is True
    assert logged["tool"] == "log_med"
    assert logged["result"]["taken"] is False
    assert len(store.med_log) == 1

    appointment = run_tool(
        store, "add_appointment", {"title": "Eye exam", "datetime": "2026-10-16T08:30"}
    )
    assert appointment["ok"] is True
    assert appointment["result"]["title"] == "Eye exam"

    task = run_tool(store, "assign_sibling_task", {"sibling": "anna", "task": "Confirm the ride"})
    assert task["ok"] is True
    assert task["result"]["sibling"] == "Anna"

    summary = run_tool(store, "get_parent_summary", {})
    assert summary["ok"] is True
    assert "metformin at 21:15" in summary["result"]


def test_run_tool_reports_unknown_tool_as_an_error_result() -> None:
    outcome = run_tool(Store(), "book_flight", {"to": "Prague"})
    assert outcome["ok"] is False
    assert outcome["tool"] == "book_flight"
    assert "unknown tool" in outcome["error"]


def test_run_tool_reports_invalid_args_as_an_error_result_without_raising() -> None:
    outcome = run_tool(Store(), "log_med", {"med": "metformin", "time": "21:15", "taken": "maybe"})
    assert outcome["ok"] is False
    assert outcome["tool"] == "log_med"
    assert outcome["error"]


def test_run_tool_accepts_missing_args_for_the_no_argument_tool() -> None:
    outcome = run_tool(Store(), "get_parent_summary", None)
    assert outcome["ok"] is True
    assert "Ludmila" in outcome["result"]


def test_apply_reply_executes_a_parsed_tool_action_against_the_store() -> None:
    store = Store()
    reply = parse_agent_reply(
        '{"tool": "log_med", "args": {"med": "metformin", "time": "21:15", "taken": false}, '
        '"reply": "Noted - worth a check-in with Mom."}'
    )

    applied = apply_reply(reply, store)

    assert applied.result is not None
    assert applied.result["ok"] is True
    assert store.med_log[0].med == "metformin"
    assert store.med_log[0].taken is False


def test_apply_reply_passes_non_tool_replies_through_unchanged() -> None:
    store = Store()
    reply = parse_agent_reply("Just a warm check-in message.")

    applied = apply_reply(reply, store)

    assert applied.kind is ReplyKind.TEXT
    assert applied.result is None
    assert store.med_log == []
