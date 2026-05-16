"""Shared LLM factory for the Research Assistant.

Tries models in priority order and returns the first one with quota.
Change MODEL_PREFERENCE to adjust the order.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Ordered by preference; first model with remaining quota wins
MODEL_PREFERENCE = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

_active_model: str | None = None


def get_model() -> str:
    """Return the first model in MODEL_PREFERENCE that still has quota."""
    global _active_model
    if _active_model:
        return _active_model

    for model in MODEL_PREFERENCE:
        try:
            ChatGoogleGenerativeAI(model=model).invoke(
                [HumanMessage(content="hi")], config={"max_retries": 0}
            )
            _active_model = model
            return model
        except Exception:
            continue

    raise RuntimeError(
        "All Gemini models are quota-exhausted. "
        "Wait for the daily limit to reset or enable billing at console.cloud.google.com."
    )


def get_llm(**kwargs) -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance using the active quota model."""
    return ChatGoogleGenerativeAI(model=get_model(), **kwargs)
