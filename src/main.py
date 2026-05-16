"""CLI driver for the LangGraph Research Assistant.

Runs an async REPL that:
  - Sends each user message through the compiled graph.
  - Streams node transitions so routing is visible.
  - Pauses and prompts the user when the Clarity Agent needs clarification.
  - Renders the final Markdown answer via the rich library.

Usage:
  python -m src.main

Commands:
  /quit   — exit
  /reset  — start a fresh conversation (new thread_id)
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

# Force UTF-8 on Windows so Rich box-drawing characters encode correctly.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("GOOGLE_API_KEY", "TAVILY_API_KEY")

console = Console(force_terminal=True, highlight=False)


def _check_env() -> None:
    """Exit with a clear message if required environment variables are missing."""
    missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        console.print(f"[bold red]Error:[/bold red] missing environment variables: {', '.join(missing)}")
        console.print("Copy [bold].env.example[/bold] to [bold].env[/bold] and fill in your keys.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Print the welcome panel."""
    console.print(Panel(
        "[bold cyan]Research Assistant[/bold cyan]\n"
        "[dim]Ask me anything about a company.[/dim]\n\n"
        "[dim]/quit[/dim] — exit  │  [dim]/reset[/dim] — clear context",
        expand=False,
    ))


def _print_transition(node: str) -> None:
    # Escape brackets so Rich doesn't treat the node name as a markup tag
    console.print(f"  [dim]\\[{node}][/dim]")


def _print_assistant(text: str) -> None:
    """Render the assistant's Markdown answer to the terminal."""
    console.print("\n[bold green]Assistant:[/bold green]")
    console.print(Markdown(text))
    console.print()


def _print_question(question: str) -> None:
    """Print a clarification question from the assistant."""
    console.print(f"\n[bold green]Assistant:[/bold green] {question}")


# ---------------------------------------------------------------------------
# Core turn runner
# ---------------------------------------------------------------------------

async def _run_turn(graph, graph_input: dict, config: dict) -> None:
    """Stream a graph turn, handling clarification interrupts, and print the result."""
    current_input = graph_input

    while True:
        nodes_seen: list[str] = []

        async for event in graph.astream(current_input, config, stream_mode="updates"):
            for node_name in event:
                if node_name.startswith("__"):
                    continue
                nodes_seen.append(node_name)
                _print_transition(node_name)

        # Check whether the graph paused on a clarification interrupt
        state = graph.get_state(config)
        if state.next and "clarification" in state.next:
            question = state.values.get("clarification_question") or "Could you clarify your question?"
            _print_question(question)
            try:
                user_answer = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                sys.exit(0)

            if not user_answer or user_answer == "/quit":
                console.print("[dim]Goodbye![/dim]")
                sys.exit(0)

            # Resume the graph with the user's answer
            current_input = Command(resume=user_answer)
            continue  # re-enter the stream loop

        # Graph ran to completion — print the answer
        final_summary = state.values.get("final_summary", "")
        if final_summary:
            _print_assistant(final_summary)
        break


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the interactive REPL loop."""
    _check_env()

    from src.graph import build_graph
    from src.state import initial_state

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    _print_banner()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input == "/reset":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            console.print("[dim]Context cleared. Starting fresh.[/dim]\n")
            continue

        # Build input — initial_state wraps the message and resets research fields;
        # the add_messages reducer ensures prior messages accumulate across turns.
        graph_input = initial_state(user_input)

        try:
            await _run_turn(graph, graph_input, config)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type /quit to exit.[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())
