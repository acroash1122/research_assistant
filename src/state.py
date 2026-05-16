"""
State schema for the LangGraph Research Assistant.

Field roles:
  messages            — full conversation history; reduced by add_messages (append-only)
  clarity_status      — whether the user's input is unambiguous ("clear"), needs a follow-up
                        question ("needs_clarification"), or has not yet been evaluated ("pending")
  company_name        — the target company extracted from the user's message; persists across
                        turns so follow-up answers can resolve an earlier ambiguity
  user_question       — the specific question being asked about the company (e.g. "who is the CEO")
  research_findings   — list of dicts, each with keys: query, results, summary
  confidence_score    — 0-10 integer; how confident the validator is that findings are sufficient
  validation_result   — whether gathered research is "sufficient", "insufficient", or "pending"
  validation_reason   — free-text explanation of why the validator made its call (for debugging)
  research_attempts   — number of research rounds completed; routing logic caps this at 3
  final_summary       — the synthesised answer returned to the user
  clarification_question — the question the clarity agent wants to ask the user before proceeding
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    clarity_status: Literal["clear", "needs_clarification", "pending"]
    company_name: str | None
    user_question: str
    research_findings: list[dict]
    confidence_score: int
    validation_result: Literal["sufficient", "insufficient", "pending"]
    validation_reason: str
    research_attempts: int
    final_summary: str
    clarification_question: str


def initial_state(user_message: str) -> ConversationState:
    """Return a fresh state with only the first user message and all other fields reset."""
    return ConversationState(
        messages=[HumanMessage(content=user_message)],
        clarity_status="pending",
        company_name=None,
        user_question="",
        research_findings=[],
        confidence_score=0,
        validation_result="pending",
        validation_reason="",
        research_attempts=0,
        final_summary="",
        clarification_question="",
    )
