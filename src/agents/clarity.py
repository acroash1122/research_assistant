"""Clarity Agent — decides whether the user's query is unambiguous enough to research.

Extracts the target company and the specific question being asked. If the
company cannot be identified from the message history, it returns a
clarification question to ask the user instead of proceeding.
"""

from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

load_dotenv()

from src.llm import get_llm
from src.state import ConversationState

_SYSTEM_PROMPT = """You are the Clarity Agent for a company research assistant.

Your job is to read the conversation and determine:
1. Which company the user is asking about.
2. What specific question they want answered about that company.
3. Whether you have enough information to proceed, or whether you need to ask for clarification.

## Rules

**Resolving pronouns from prior context**
If the latest user message uses pronouns or vague references such as "they",
"their", "it", "the company", "that company", "them", or similar — AND a
previous turn in the conversation already established a company name — treat
the query as CLEAR and reuse that company name. Do NOT ask for clarification
in this case.

**When to set needs_clarification**
Only set clarity_status = "needs_clarification" when there is genuinely no
way to identify the company from the full message history. Set
clarification_question to a concise, friendly question to ask the user.

**Extracting user_question**
Always populate user_question with the specific thing the user wants to know
(e.g. "who is the CEO", "what is the stock price", "recent news"). If the
user asked something general like "tell me about X", use "general overview".

## Examples

- "tell me about Tesla" → clear, company_name="Tesla", user_question="general overview"
- "what about their CEO" (previous turn mentioned Tesla) → clear, company_name="Tesla", user_question="who is the CEO"
- "tell me about that company" (no prior context) → needs_clarification, clarification_question="Which company are you asking about?"
- "what is Apple's revenue?" → clear, company_name="Apple", user_question="revenue"

## Output format

Return a JSON object with:
  clarity_status: "clear" or "needs_clarification"
  company_name: the company name as a string, or null if unknown
  user_question: what the user wants to know (always set, even if needs_clarification)
  clarification_question: your question for the user, or null if clear
"""


class ClarityDecision(BaseModel):
    clarity_status: Literal["clear", "needs_clarification"]
    company_name: str | None
    user_question: str
    clarification_question: str | None


_llm = None  # initialised lazily on first call

def _get_llm():
    """Return the lazily-initialised structured-output LLM for clarity decisions."""
    global _llm
    if _llm is None:
        _llm = get_llm().with_structured_output(ClarityDecision)
    return _llm


def clarity_agent(state: ConversationState) -> dict:
    """Determine whether the query is clear enough to research and extract the company name."""
    messages: list[BaseMessage] = state["messages"]
    prior_company: str | None = state.get("company_name")

    context_note = ""
    if prior_company:
        context_note = f"\n\nNote: The previous turn established the company as \"{prior_company}\". Use this if the latest message refers to it implicitly."

    decision: ClarityDecision = _get_llm().invoke(
        [SystemMessage(content=_SYSTEM_PROMPT + context_note)] + list(messages)
    )

    return {
        "clarity_status": decision.clarity_status,
        "company_name": decision.company_name,
        "user_question": decision.user_question,
        "clarification_question": decision.clarification_question or "",
    }
