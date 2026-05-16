"""Synthesis Agent — produces the final Markdown answer from accumulated research findings.

Combines all research findings, conversation context, and confidence metadata
into a single well-structured Markdown response. Appends an AIMessage to the
conversation history via the add_messages reducer.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

from src.llm import get_llm
from src.state import ConversationState

_SYSTEM_PROMPT = """You are a research assistant that writes clear, well-structured Markdown answers.

## Format rules

1. **Headline**: open with a single bold line that directly answers the question.
2. **Body**: use Markdown headings (## Recent Developments, ## Key Facts, etc.)
   only when they add clarity. Omit headings for simple factual answers.
3. **Citations**: cite sources inline as [source](url) using the URLs provided in
   the research findings. Every factual claim should have at least one citation.
4. **Conversational context**: if previous turns in the conversation are relevant,
   reference them naturally (e.g. "As we discussed earlier, Tesla's CEO is...").
5. **Disclaimer**: if confidence_score < 6 OR (research_attempts == 3 AND
   validation_result == "insufficient"), end with:
   > **Note:** I had limited information on this topic; treat this answer with care.

Write for a professional audience. Be concise — prefer one good sentence over three vague ones."""


_llm = None  # initialised lazily on first call

def _get_llm():
    """Return the lazily-initialised plain LLM for Markdown answer generation."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def synthesis_agent(state: ConversationState) -> dict:
    """Produce the final Markdown answer by combining all research findings."""
    company: str = state.get("company_name") or "the company"
    question: str = state.get("user_question") or ""
    findings: list[dict] = state.get("research_findings") or []
    confidence: int = state.get("confidence_score") or 0
    validation: str = state.get("validation_result") or "pending"
    attempts: int = state.get("research_attempts") or 0
    messages = list(state.get("messages") or [])

    # Build findings block
    findings_text = ""
    for i, f in enumerate(findings, 1):
        findings_text += f"\n### Round {i} findings (query: \"{f['query']}\")\n"
        findings_text += f"Summary: {f['summary']}\n"
        for r in f.get("results", []):
            findings_text += f"- [{r['title']}]({r['url']}): {r['content'][:300]}\n"

    needs_disclaimer = confidence < 6 or (attempts >= 3 and validation == "insufficient")

    user_content = (
        f"Company: {company}\n"
        f"Question: {question}\n"
        f"Confidence score: {confidence}/10\n"
        f"Validation result: {validation}\n"
        f"Research attempts: {attempts}\n"
        f"Add disclaimer: {'yes' if needs_disclaimer else 'no'}\n"
        f"\n## Research findings\n{findings_text if findings_text else 'No findings available.'}"
    )

    response = _get_llm().invoke(
        [SystemMessage(content=_SYSTEM_PROMPT)]
        + list(messages)
        + [HumanMessage(content=user_content)]
    )

    final_summary: str = response.content

    return {
        "final_summary": final_summary,
        "messages": [AIMessage(content=final_summary)],
    }
