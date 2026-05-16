"""Research Agent — searches for information about a company and synthesises findings.

Given a resolved company_name and user_question from state, it:
1. Templates a search query.
2. Calls the Tavily search tool.
3. Passes results to an LLM for a structured summary + confidence score.
4. Appends the findings to state and increments the attempt counter.
"""

from __future__ import annotations

import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

load_dotenv()

from src.llm import get_llm

from src.state import ConversationState
from src.tools.tavily_search import search_company

_SYNTHESIS_SYSTEM = """You are a research synthesis agent. You are given a search query
and a list of web search results. Your job is to:
1. Write a concise 2-4 sentence summary that directly answers the user's question.
2. Assign a confidence score (0-10) based on how well the results answer the question.
3. Give a one-sentence reason for your score.

Confidence scale:
  0-3  : few results, irrelevant, or contradictory
  4-5  : some results but stale, off-topic, or only partially answer the question
  6-7  : solid coverage, recent sources, directly answers the question
  8-10 : comprehensive, multiple corroborating sources, highly recent and specific

Be honest — if the results are weak, give a low score."""


class ResearchOutput(BaseModel):
    summary: str
    confidence_score: int
    confidence_reason: str


_llm = None  # initialised lazily on first call

def _get_llm():
    """Return the lazily-initialised structured-output LLM for research synthesis."""
    global _llm
    if _llm is None:
        _llm = get_llm().with_structured_output(ResearchOutput)
    return _llm


def _build_query(company: str, question: str) -> str:
    """Append the current year to the query so results skew towards recent sources."""
    year = datetime.datetime.now().year
    return f"{company} {question} {year}"


async def research_agent(state: ConversationState) -> dict:
    """Search for company information and synthesise findings with a confidence score."""
    company: str = state["company_name"] or ""
    question: str = state["user_question"] or ""
    current_findings: list[dict] = list(state.get("research_findings") or [])
    attempts: int = state.get("research_attempts") or 0

    query = _build_query(company, question)

    results = await search_company(query, max_results=5)

    if not results:
        new_finding = {"query": query, "results": [], "summary": "No results found."}
        return {
            "research_findings": current_findings + [new_finding],
            "confidence_score": 0,
            "research_attempts": attempts + 1,
        }

    results_text = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['url']}\n{r['content'][:400]}"
        for i, r in enumerate(results)
    )
    user_msg = f"Search query: {query}\n\nSearch results:\n{results_text}"

    output: ResearchOutput = _get_llm().invoke(
        [SystemMessage(content=_SYNTHESIS_SYSTEM), HumanMessage(content=user_msg)]
    )

    new_finding = {
        "query": query,
        "results": results,
        "summary": output.summary,
    }

    return {
        "research_findings": current_findings + [new_finding],
        "confidence_score": output.confidence_score,
        "research_attempts": attempts + 1,
    }
