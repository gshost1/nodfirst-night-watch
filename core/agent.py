"""Caregiver agent: persona prompt, JSON tool protocol, in-memory family store.

The persona is written for Mom's adult children — siblings Anna and Pavel — and is
warm, concise, and safe: it never diagnoses and escalates emergencies to 911 and
Mom's doctor. Tools run over a JSON protocol (never SDK tool-calls, per the
Qwen3-4B reliability note): the model is asked to reply with a single
``{"tool": ..., "args": ..., "reply": ...}`` block, and the parser degrades
gracefully — tool JSON → bare JSON → plain text — so a malformed reply never
crashes a caller.

The `Store` is the in-memory family state, seeded for the demo: Mom ("Ludmila",
81), siblings Anna and Pavel, the evening metformin schedule, and two
appointments. All four tools are `Store` methods so every surface (app, MCP
server) executes the same underlying code.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# In-package reuse of the balanced-JSON extractor that core.llm already ships:
# both modules parse "first JSON object inside prose/fences" the same way.
from core.llm import _parse_first_json_object

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_NAMES",
    "TOOL_LOG_MED",
    "TOOL_ADD_APPOINTMENT",
    "TOOL_ASSIGN_SIBLING_TASK",
    "TOOL_GET_PARENT_SUMMARY",
    "AgentReply",
    "ReplyKind",
    "Store",
    "parse_agent_reply",
    "apply_reply",
    "run_tool",
    "Parent",
    "Sibling",
    "MedSchedule",
    "MedLogEntry",
    "Appointment",
    "SiblingTask",
    "LogMedArgs",
    "AddAppointmentArgs",
    "AssignSiblingTaskArgs",
]

TOOL_LOG_MED = "log_med"
TOOL_ADD_APPOINTMENT = "add_appointment"
TOOL_ASSIGN_SIBLING_TASK = "assign_sibling_task"
TOOL_GET_PARENT_SUMMARY = "get_parent_summary"

#: The complete tool registry; the parser treats any other name as plain text.
TOOL_NAMES: frozenset[str] = frozenset(
    {
        TOOL_LOG_MED,
        TOOL_ADD_APPOINTMENT,
        TOOL_ASSIGN_SIBLING_TASK,
        TOOL_GET_PARENT_SUMMARY,
    }
)

SYSTEM_PROMPT = """\
You are Night Watch, the private caregiver assistant for one family: \
Ludmila — "Mom", 81 — and the adult children who share her care, \
siblings Anna and Pavel. You are writing to those adult children, \
never to medical staff.

How you speak:
- Warm, plain, and concise — a short paragraph or a few bullets, no jargon.
- You are not a doctor and you never diagnose or suggest treatments, doses, or \
medication changes.
- For a possible medical emergency — chest pain, trouble breathing, a fall, stroke signs — \
tell the family to call 911 right away.
- For changes that worry but are not life-threatening, tell the family to call Mom's doctor \
before trying anything on their own.
- When the situation is unclear, say what to watch for and who to call — never guess a cause.

What you can do:
- log_med(med, time, taken) — record that Mom took or missed a medication.
- add_appointment(title, datetime) — add an upcoming appointment.
- assign_sibling_task(sibling, task) — assign a task to Anna or Pavel.
- get_parent_summary() — render the current summary of Mom's care from the store.

When a family message calls for one of those actions, reply with a single JSON block \
and nothing else:
{"tool": "<tool name>", "args": {...}, "reply": "<one short, warm line for the family>"}
When no action is needed, reply with just the warm, short message for the family.
"""


class Parent(BaseModel):
    """Mom — the person the family cares for together."""

    name: str
    age: int


class Sibling(BaseModel):
    """One of Mom's adult children sharing her care."""

    name: str


class MedSchedule(BaseModel):
    """A planned medication slot (seeded: the evening metformin)."""

    med: str
    time_of_day: str


class MedLogEntry(BaseModel):
    """One taken/missed medication record produced by the log_med tool."""

    med: str
    time: str
    taken: bool


class Appointment(BaseModel):
    """An upcoming care appointment."""

    title: str
    datetime: str


class SiblingTask(BaseModel):
    """A task assigned to one sibling by the assign_sibling_task tool."""

    sibling: str
    task: str


class LogMedArgs(BaseModel):
    """Arguments accepted by the log_med tool action (extra keys ignored)."""

    med: str = Field(min_length=1)
    time: str = Field(min_length=1)
    taken: bool


class AddAppointmentArgs(BaseModel):
    """Arguments accepted by the add_appointment tool action (extra keys ignored)."""

    title: str = Field(min_length=1)
    datetime: str = Field(min_length=1)


class AssignSiblingTaskArgs(BaseModel):
    """Arguments accepted by the assign_sibling_task tool action (extra keys ignored)."""

    sibling: str = Field(min_length=1)
    task: str = Field(min_length=1)


class ReplyKind(StrEnum):
    """How a model reply was interpreted by the JSON tool protocol."""

    TOOL = "tool"
    JSON = "json"
    TEXT = "text"


@dataclasses.dataclass(frozen=True, slots=True)
class AgentReply:
    """A parsed model reply: a tool call, a bare JSON object, or plain text.

    `text` carries the family-facing message: the `reply` field of a tool block,
    the raw content of a text reply, or "" when absent. `result` is filled by
    `apply_reply` after executing a tool action against a store.
    """

    kind: ReplyKind
    tool: str | None = None
    args: dict[str, Any] | None = None
    text: str = ""
    data: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class Store:
    """In-memory family state, seeded for the Night Watch demo.

    All four agent tools are `Store` methods so every surface — the app, the MCP
    server — executes the same underlying code. Fresh `Store` instances share
    nothing, so a new run starts from a clean state.
    """

    def __init__(self) -> None:
        self.parent = Parent(name="Ludmila", age=81)
        self.siblings: list[Sibling] = [Sibling(name="Anna"), Sibling(name="Pavel")]
        self.med_schedule: list[MedSchedule] = [MedSchedule(med="metformin", time_of_day="evening")]
        self.appointments: list[Appointment] = [
            Appointment(
                title="Endocrinology follow-up - metformin review", datetime="2026-10-13T09:30"
            ),
            Appointment(title="Cardiology check-up", datetime="2026-10-20T11:00"),
        ]
        self.med_log: list[MedLogEntry] = []
        self.sibling_tasks: list[SiblingTask] = []

    def log_med(self, med: str, time: str, taken: bool) -> MedLogEntry:
        """Record that Mom took (or missed) a medication."""
        entry = MedLogEntry(med=med, time=time, taken=taken)
        self.med_log.append(entry)
        return entry

    def add_appointment(self, title: str, datetime: str) -> Appointment:
        """Add an upcoming appointment to the family calendar."""
        appointment = Appointment(title=title, datetime=datetime)
        self.appointments.append(appointment)
        return appointment

    def assign_sibling_task(self, sibling: str, task: str) -> SiblingTask:
        """Assign a task to one of the seeded siblings (name case-insensitive)."""
        entry = SiblingTask(sibling=self._match_sibling(sibling), task=task)
        self.sibling_tasks.append(entry)
        return entry

    def get_parent_summary(self) -> str:
        """Render Mom's current care picture from the store (mutations included)."""
        header = (
            f"Mom ({self.parent.name}, {self.parent.age})"
            f" — care summary for {self._sibling_names()}"
        )
        lines = [header, ""]

        if self.med_log:
            lines.append("Medications logged:")
            for entry in self.med_log:
                if entry.taken:
                    lines.append(f"- {entry.med} at {entry.time} — taken")
                else:
                    lines.append(f"- {entry.med} at {entry.time} — NOT taken, check in with Mom")
        else:
            lines.append("Medications logged: none yet tonight.")

        lines.append("")
        lines.append("Medication schedule:")
        for slot in self.med_schedule:
            lines.append(f"- {slot.med} in the {slot.time_of_day}")

        lines.append("")
        lines.append("Upcoming appointments:")
        for appointment in sorted(self.appointments, key=lambda item: item.datetime):
            lines.append(f"- {appointment.title} ({appointment.datetime})")

        lines.append("")
        lines.append("Sibling tasks:")
        if self.sibling_tasks:
            for task in self.sibling_tasks:
                lines.append(f"- {task.sibling}: {task.task}")
        else:
            lines.append("- none assigned yet")

        return "\n".join(lines)

    def _match_sibling(self, sibling: str) -> str:
        """Canonical sibling name for a caller-supplied name; unknown names raise."""
        candidate = (sibling or "").strip()
        for known in self.siblings:
            if known.name.lower() == candidate.lower():
                return known.name
        raise ValueError(
            f"unknown sibling: {sibling!r} (expected one of: "
            f"{', '.join(s.name for s in self.siblings)})"
        )

    def _sibling_names(self) -> str:
        """Human join of the sibling names, e.g. 'siblings Anna and Pavel'."""
        names = [sibling.name for sibling in self.siblings]
        if len(names) > 1:
            return f"siblings {', '.join(names[:-1])} and {names[-1]}"
        if names:
            return f"siblings {names[0]}"
        return "the family"


def parse_agent_reply(content: Any) -> AgentReply:
    """Interpret a model reply; degrade to plain text instead of ever raising.

    Precedence: a tool action JSON block wins, then any bare JSON object; anything
    else — malformed JSON, unknown tool names, empty input — is plain text.
    """
    text = content if isinstance(content, str) else ""
    try:
        payload = _parse_first_json_object(text)
    except (ValueError, TypeError):
        return AgentReply(kind=ReplyKind.TEXT, text=text)
    if not isinstance(payload, dict):
        return AgentReply(kind=ReplyKind.TEXT, text=text)

    tool = _tool_name_from(payload)
    if tool is None:
        return AgentReply(kind=ReplyKind.JSON, data=payload, text=text)
    if tool not in TOOL_NAMES:
        # Unknown tool: the safest action is none — surface the reply as plain text.
        return AgentReply(kind=ReplyKind.TEXT, text=text)

    reply_line = payload.get("reply")
    return AgentReply(
        kind=ReplyKind.TOOL,
        tool=tool,
        args=_tool_args_from(payload),
        text=reply_line.strip() if isinstance(reply_line, str) else "",
    )


def apply_reply(reply: AgentReply, store: Store) -> AgentReply:
    """Execute a parsed tool-action reply against the store, filling `result`.

    Non-tool replies pass through unchanged. Invalid tool arguments surface as
    `result["ok"] is False` — this too never raises.
    """
    if reply.kind is not ReplyKind.TOOL or reply.tool is None:
        return reply
    outcome = run_tool(store, reply.tool, reply.args)
    return dataclasses.replace(reply, result=outcome)


def run_tool(store: Store, tool: str, args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Execute one tool by name against the store; never raises on bad input.

    Returns `{"ok": True, "tool", "result"}` on success and
    `{"ok": False, "tool", "error"}` when the tool name or arguments are invalid.
    """
    clean_args = dict(args) if isinstance(args, Mapping) else {}
    try:
        result: Any
        if tool == TOOL_LOG_MED:
            parsed = LogMedArgs(**clean_args)
            result = store.log_med(parsed.med, parsed.time, parsed.taken)
        elif tool == TOOL_ADD_APPOINTMENT:
            parsed_appointment = AddAppointmentArgs(**clean_args)
            result = store.add_appointment(
                parsed_appointment.title, parsed_appointment.datetime
            )
        elif tool == TOOL_ASSIGN_SIBLING_TASK:
            parsed_task = AssignSiblingTaskArgs(**clean_args)
            result = store.assign_sibling_task(parsed_task.sibling, parsed_task.task)
        elif tool == TOOL_GET_PARENT_SUMMARY:
            result = store.get_parent_summary()
        else:
            return {"ok": False, "tool": tool, "error": f"unknown tool: {tool}"}
    except (ValueError, TypeError) as exc:
        # Covers pydantic ValidationError (a ValueError) and Store's own checks.
        return {"ok": False, "tool": tool, "error": str(exc)}
    return {"ok": True, "tool": tool, "result": _json_ready(result)}


def _tool_name_from(payload: dict[str, Any]) -> str | None:
    """Tool name from the canonical 'tool' key (accepts the 'action' alias)."""
    for key in ("tool", "action"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _tool_args_from(payload: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments from the canonical 'args' key (accepts 'arguments')."""
    for key in ("args", "arguments"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _json_ready(value: Any) -> Any:
    """Return a JSON-serializable version of a tool result."""
    return value.model_dump() if isinstance(value, BaseModel) else value
