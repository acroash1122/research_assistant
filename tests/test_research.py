"""Standalone smoke-test for the Research Agent. Run with: python tests/test_research.py"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    print("ERROR: TAVILY_API_KEY not set.")
    sys.exit(1)
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not set.")
    sys.exit(1)


async def main() -> None:
    from src.agents.research import research_agent
    from src.state import initial_state

    state = initial_state("Tell me recent news about Tesla.")
    state["company_name"] = "Tesla"
    state["user_question"] = "recent news"
    state["clarity_status"] = "clear"

    print("Running Research Agent for Tesla / recent news …\n")
    updates = await research_agent(state)

    print(f"research_attempts : {updates['research_attempts']}")
    print(f"confidence_score  : {updates['confidence_score']}/10")
    print(f"findings count    : {len(updates['research_findings'])}")
    print()

    finding = updates["research_findings"][-1]
    print(f"Query : {finding['query']}")
    print(f"Summary:\n{finding['summary']}")
    print(f"\nTop result: {finding['results'][0]['title']} — {finding['results'][0]['url']}")


if __name__ == "__main__":
    asyncio.run(main())
