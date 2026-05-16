"""Validator Agent — judges whether accumulated research findings sufficiently answer the user's question.

Reads all findings gathered so far (not just the latest) so it can make a
cumulative judgement across retry rounds. Returns "sufficient" or "insufficient"
with a one-sentence reason.
"""

from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

load_dotenv()

from src.llm import get_llm
from src.state import ConversationState

_SYSTEM_PROMPT = """You are a research quality validator. Your job is to decide whether
the accumulated search findings genuinely and specifically answer the user's question.

## Judgement criteria

**sufficient** — the findings directly, specifically, and recently answer the question.
  - Example: question="who is the CEO of Tesla" → findings contain "Elon Musk is CEO" → sufficient
  - Example: question="Apple Q1 2026 revenue" → findings contain the exact revenue figure → sufficient

**insufficient** — the findings are vague, off-topic, stale, or only tangentially related.
  - Example: question="who is the CEO of Tesla" → findings only discuss Tesla stock price → insufficient
  - Example: question="Apple Q1 2026 revenue" → findings discuss Apple products but not Q1 revenue → insufficient
  - Example: findings are empty or "No results found" → insufficient

## Rules
- Be strict. If the answer is buried in vague text and hard to extract, that is insufficient.
- Consider ALL accumulated findings together, not just the latest round.
- If the findings contain directly relevant, specific information that a person could use
  to answer the question, that is sufficient.

Return a JSON object with:
  validation_result: "sufficient" or "insufficient"
  validation_reason: one concise sentence explaining your decision"""


class ValidationDecision(BaseModel):
    validation_result: Literal["sufficient", "insufficient"]
    validation_reason: str


_llm = None  # initialised lazily on first call

def _get_llm():
    """Return the lazily-initialised structured-output LLM for validation decisions."""
    global _llm
    if _llm is None:
        _llm = get_llm().with_structured_output(ValidationDecision)
    return _llm


def validator_agent(state: ConversationState) -> dict:
    """Judge whether the accumulated findings sufficiently answer the user's question."""
    company: str = state.get("company_name") or "unknown company"
    question: str = state.get("user_question") or ""
    findings: list[dict] = state.get("research_findings") or []

    if not findings:
        return {
            "validation_result": "insufficient",
            "validation_reason": "No research findings have been gathered yet.",
        }

    findings_text = ""
    for i, f in enumerate(findings, 1):
        findings_text += f"\n--- Round {i} (query: {f['query']}) ---\n"
        findings_text += f"Summary: {f['summary']}\n"
        for j, r in enumerate(f.get("results", [])[:3], 1):
            findings_text += f"  [{j}] {r['title']}: {r['content'][:200]}\n"

    user_msg = (
        f"Company: {company}\n"
        f"User question: {question}\n\n"
        f"Accumulated findings:\n{findings_text}"
    )

    decision: ValidationDecision = _get_llm().invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    )

    return {
        "validation_result": decision.validation_result,
        "validation_reason": decision.validation_reason,
    }
