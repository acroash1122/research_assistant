"""Tavily search tool with MCP primary path and direct-SDK fallback.

Primary path: spawns the official Tavily MCP server via
  npx -y tavily-mcp  (stdio transport, Node.js required)

Fallback: uses the tavily-python SDK directly if MCP fails or Node.js is
absent. The active backend is logged at WARNING level so it is visible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any

import certifi
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL installs REQUESTS_CA_BUNDLE / SSL_CERT_FILE pointing at its own
# bundle. Override both to certifi's known-good path so every outbound HTTPS
# request in this process uses the right CA store.
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# None = not yet probed; True/False cached after the first call
_mcp_available: bool | None = None
BACKEND: str = "unknown"


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw search result dict to the standard {title, url, content, score} shape."""
    return {
        "title": raw.get("title", ""),
        "url": raw.get("url", ""),
        "content": raw.get("content", raw.get("snippet", "")),
        "score": float(raw.get("score", 0.0)),
    }


def _parse_mcp_text(text: str) -> list[dict]:
    """Parse the formatted text block that tavily-mcp returns into result dicts."""
    import re
    results = []
    # Split on blank line followed by "Title:" to separate each result entry
    blocks = re.split(r"\n\n(?=Title:)", text)
    for block in blocks:
        if not block.strip() or block.startswith("Detailed Results"):
            continue
        title = url = content = ""
        for line in block.splitlines():
            if line.startswith("Title: "):
                title = line[7:].strip()
            elif line.startswith("URL: "):
                url = line[5:].strip()
            elif line.startswith("Content: "):
                content = line[9:].strip()
        if title or url:
            results.append({"title": title, "url": url, "content": content, "score": 1.0})
    return results


async def _search_via_mcp(query: str, max_results: int) -> list[dict]:
    """Run a Tavily search via the stdio MCP server spawned by npx."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # stdio transport: langchain-mcp-adapters spawns the npx process itself.
    # Pass the full environment so npx can resolve Node modules and PATH.
    client = MultiServerMCPClient({
        "tavily": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "transport": "stdio",
            "env": {**os.environ, "TAVILY_API_KEY": TAVILY_API_KEY},
        }
    })
    tools = await client.get_tools()

    search_tool = next((t for t in tools if "search" in t.name.lower()), None)
    if search_tool is None:
        raise RuntimeError("No search tool found on Tavily MCP server")

    raw = await search_tool.ainvoke({"query": query, "max_results": max_results})

    # tavily-mcp returns a list of MCP content blocks: [{"type": "text", "text": "..."}]
    # The text is formatted as:
    #   Detailed Results:
    #   Title: ...
    #   URL: ...
    #   Content: ...
    #   (repeated per result)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        return _parse_mcp_text(raw[0]["text"])[:max_results]

    # Fallback: try JSON or list-of-dicts format
    import json
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return [_normalize(r) for r in raw[:max_results]]
    return [_normalize(r) for r in raw.get("results", [])[:max_results]]


async def _search_via_sdk(query: str, max_results: int) -> list[dict]:
    """Run a Tavily search via the tavily-python SDK (fallback path)."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.search(query=query, max_results=max_results),
    )
    results: list[dict] = response.get("results", [])
    return [_normalize(r) for r in results]


async def search_company(query: str, max_results: int = 5) -> list[dict]:
    """Return up to *max_results* normalized dicts: title, url, content, score."""
    global _mcp_available, BACKEND

    if _mcp_available is None:
        # Only attempt MCP if npx is on PATH
        if shutil.which("npx") is None:
            logger.warning("npx not found — Tavily MCP unavailable, using SDK")
            _mcp_available = False
            BACKEND = "sdk"
        else:
            try:
                results = await asyncio.wait_for(
                    _search_via_mcp(query, max_results), timeout=45.0
                )
                _mcp_available = True
                BACKEND = "mcp"
                logger.info("Tavily MCP (stdio/npx) active")
                return results
            except Exception as exc:
                logger.warning(
                    "Tavily MCP unavailable (%s: %s) — falling back to SDK",
                    type(exc).__name__,
                    exc,
                )
                _mcp_available = False
                BACKEND = "sdk"

    if _mcp_available:
        return await _search_via_mcp(query, max_results)

    return await _search_via_sdk(query, max_results)
