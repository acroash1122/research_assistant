"""LangGraph graph definition for the Research Assistant.

Routing logic:
  START → clarity
    clarity  : if needs_clarification → clarification; else → research
    clarification : always → clarity  (loops until question is clear)
    research : if confidence >= 6 → synthesis (fast path); else → validator
    validator: if sufficient → synthesis;
               elif attempts >= 3 → synthesis (force-stop);
               else → research  (retry loop, max 3 rounds)
    synthesis → END
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.clarity import clarity_agent
from src.agents.research import research_agent
from src.agents.synthesis import synthesis_agent
from src.agents.validator import validator_agent
from src.state import ConversationState

load_dotenv()

DEFAULT_CONFIG: dict = {"configurable": {"thread_id": "default"}}


# ---------------------------------------------------------------------------
# Clarification node
# ---------------------------------------------------------------------------

def clarification_node(state: ConversationState) -> dict:
    """Interrupt the graph and ask the user a clarification question; resume with their answer."""
    question = state.get("clarification_question") or "Could you clarify your question?"
    human_response: str = interrupt(question)
    return {
        "messages": [HumanMessage(content=human_response)],
        "clarity_status": "pending",
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_clarity(state: ConversationState) -> str:
    """Route to clarification if the query is ambiguous, otherwise proceed to research."""
    if state.get("clarity_status") == "needs_clarification":
        return "clarification"
    return "research"


def _route_research(state: ConversationState) -> str:
    """Skip validation and go straight to synthesis if confidence is already high."""
    if (state.get("confidence_score") or 0) >= 6:
        return "synthesis"
    return "validator"


def _route_validator(state: ConversationState) -> str:
    """Synthesise if findings are sufficient or attempts are exhausted, else retry research."""
    if state.get("validation_result") == "sufficient":
        return "synthesis"
    if (state.get("research_attempts") or 0) >= 3:
        return "synthesis"
    return "research"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the research assistant graph with a MemorySaver checkpointer."""
    graph = StateGraph(ConversationState)

    graph.add_node("clarity", clarity_agent)
    graph.add_node("clarification", clarification_node)
    graph.add_node("research", research_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("synthesis", synthesis_agent)

    graph.add_edge(START, "clarity")

    graph.add_conditional_edges("clarity", _route_clarity, {
        "clarification": "clarification",
        "research": "research",
    })

    graph.add_edge("clarification", "clarity")

    graph.add_conditional_edges("research", _route_research, {
        "synthesis": "synthesis",
        "validator": "validator",
    })

    graph.add_conditional_edges("validator", _route_validator, {
        "synthesis": "synthesis",
        "research": "research",
    })

    graph.add_edge("synthesis", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["clarification"])
