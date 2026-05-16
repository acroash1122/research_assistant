"""Standalone smoke-test for the Tavily search tool. Run with: python tests/test_tavily.py"""

import asyncio
import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    print("ERROR: TAVILY_API_KEY is not set. Copy .env.example to .env and fill in your key.")
    sys.exit(1)


async def main() -> None:
    import src.tools.tavily_search as ts

    print("Searching for 'Tesla recent news' (max_results=3) …\n")
    results = await ts.search_company("Tesla recent news", max_results=3)

    for r in results:
        print(f"[{r['score']:.2f}] {r['title']} — {r['url']}")

    print(f"\nBackend used: {ts.BACKEND}")


if __name__ == "__main__":
    asyncio.run(main())
