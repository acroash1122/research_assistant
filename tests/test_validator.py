"""Standalone smoke-test for the Validator Agent. Run with: python tests/test_validator.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not set.")
    sys.exit(1)

from src.agents.validator import validator_agent
from src.state import initial_state


def run(label: str, state: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Scenario : {label}")
    print(f"Question : {state['user_question']}")
    result = validator_agent(state)
    print(f"  validation_result : {result['validation_result']}")
    print(f"  validation_reason : {result['validation_reason']}")


if __name__ == "__main__":
    # Scenario 1: mismatched findings — stock price results for a CEO question
    mismatch = initial_state("Who is the CEO of Tesla?")
    mismatch["company_name"] = "Tesla"
    mismatch["user_question"] = "who is the CEO"
    mismatch["research_findings"] = [
        {
            "query": "Tesla stock price 2026",
            "results": [
                {"title": "TSLA Stock", "url": "https://example.com", "content": "Tesla stock is trading at $180.", "score": 1.0},
                {"title": "Tesla market cap", "url": "https://example.com/2", "content": "Tesla's market cap reached $600B.", "score": 0.9},
            ],
            "summary": "Tesla stock is currently trading around $180 with a market cap of $600B.",
        }
    ]
    run("Mismatched findings (stock price vs CEO question)", mismatch)

    # Scenario 2: directly relevant findings
    relevant = initial_state("Who is the CEO of Tesla?")
    relevant["company_name"] = "Tesla"
    relevant["user_question"] = "who is the CEO"
    relevant["research_findings"] = [
        {
            "query": "Tesla CEO 2026",
            "results": [
                {"title": "Tesla Leadership", "url": "https://example.com", "content": "Elon Musk is the CEO and largest shareholder of Tesla, Inc.", "score": 1.0},
            ],
            "summary": "Elon Musk is the CEO of Tesla. He has led the company since 2008 and remains its largest individual shareholder.",
        }
    ]
    run("Relevant findings (correct CEO info)", relevant)

    # Scenario 3: empty findings
    empty = initial_state("What is Apple's revenue?")
    empty["company_name"] = "Apple"
    empty["user_question"] = "Q1 2026 revenue"
    empty["research_findings"] = []
    run("Empty findings", empty)

    print("\nDone.")
