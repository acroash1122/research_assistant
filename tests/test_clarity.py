"""Standalone smoke-test for the Clarity Agent. Run with: python tests/test_clarity.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.clarity import clarity_agent
from src.state import initial_state


def run(label: str, state: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Scenario: {label}")
    print(f"Messages: {[m.content for m in state['messages']]}")
    if state.get("company_name"):
        print(f"Prior company in state: {state['company_name']}")
    result = clarity_agent(state)
    print(f"  clarity_status        : {result['clarity_status']}")
    print(f"  company_name          : {result['company_name']}")
    print(f"  user_question         : {result['user_question']}")
    print(f"  clarification_question: {result['clarification_question'] or '(none)'}")


if __name__ == "__main__":
    # Scenario 1: clear query — company named explicitly
    run("Clear query (explicit company)", initial_state("Tell me about Tesla's latest earnings."))

    # Scenario 2: ambiguous — no company, no prior context
    run("Ambiguous (no company, no prior context)", initial_state("Tell me about that company."))

    # Scenario 3: follow-up with pronoun — prior company in state
    follow_up = initial_state("What about their CEO?")
    follow_up["messages"] = [
        HumanMessage(content="Tell me about Tesla."),
        AIMessage(content="Tesla is an electric vehicle company..."),
        HumanMessage(content="What about their CEO?"),
    ]
    follow_up["company_name"] = "Tesla"
    run("Follow-up with pronoun (prior company = Tesla)", follow_up)

    print("\nDone.")
